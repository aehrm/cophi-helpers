import pandas
import torch
import os
import sys
import more_itertools
from tqdm.auto import tqdm

from transformers import BertTokenizer, BertModel, BertConfig
from torch.nn.utils.rnn import pad_sequence

if len(sys.argv) < 4:
    print("Usage: python makecache.py cacheword_file processed_dir output_file", file=sys.stderr)
    sys.exit(-1)

cachewordfile = sys.argv[1]
processed_dir = sys.argv[2]
outputfile = sys.argv[3]


print('Initializing model', file=sys.stderr)

# adjust your model here
config = BertConfig.from_pretrained("bert-base-german-cased", output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
model = BertModel.from_pretrained("bert-base-german-cased", config=config)
assert model.config.output_hidden_states
model.to('cuda')

model.eval()
torch.set_grad_enabled(False)
max_piecelength = 8

with open(cachewordfile) as f:
    cachewords = [x.strip() for x in f.readlines()]

word_index_map = dict(zip(cachewords, range(len(cachewords))))
tokenized_docs = [os.path.join(processed_dir, f) for f in os.listdir(processed_dir)]

frequencies = torch.zeros([len(cachewords)], dtype=torch.int)
accumulated_layers = torch.zeros([len(cachewords), max_piecelength, 13, 768], dtype=torch.float32)


def read_doc(path):
    fileobj = open(path, "r")
    content = fileobj.read()
    fileobj.close()

    return [line.split(' ') for line in content.split('\n')]


def make_sequences(sents):
    acc = []
    for sent in sents:
        if len(sent) > 510:
            print("Sentence longer than 510 tokens", file=sys.stderr)
            continue

        if len(acc) + len(sent) > 510:
            yield acc
            acc = sent
        else:
            acc += sent

    if len(acc) > 0:
        yield acc


def forward_pass(batch):
    indexed_tokens = [tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(seq)) for seq in batch]
    tokens_tensor = pad_sequence(list(map(torch.tensor, indexed_tokens)), batch_first=True)
    attention_tensor = (tokens_tensor != 0).int()
    attention_tensor = attention_tensor.to('cuda')
    tokens_tensor = tokens_tensor.to('cuda')

    outputs = model(tokens_tensor, attention_mask=attention_tensor)
    inner_layers = outputs[2]
    permuted = torch.stack(list(inner_layers)).permute(1, 2, 0, 3).to('cpu')
    return permuted


def token_ranges(seq):
    for t in more_itertools.split_before(enumerate(seq), lambda x: not x[1].startswith('#')):
        token = ''.join(map(lambda x: x[1], t)).replace('#', '')
        yield (token, t[0][0], t[-1][0])


def process_output(batch, output):
    for i, seq in enumerate(batch):
        for token, begin, end in token_ranges(seq):
            if token not in cachewords:
                continue

            # shift by one due to [CLS]
            tensor = torch.cat(
                [output[i].narrow(0, 1 + begin, end - begin + 1), torch.zeros(max_piecelength - end + begin - 1, 13, 768)], 0)
            frequencies[word_index_map[token]] += 1
            accumulated_layers[word_index_map[token]] += tensor


num_sents = sum(map(len, map(read_doc, tokenized_docs)))
sents = tqdm(more_itertools.flatten(map(read_doc, tokenized_docs)), total=num_sents)
batches = more_itertools.chunked(make_sequences(sents), 10)

for batch in batches:
    process_output(batch, forward_pass(batch))

torch.save({'frequencies': frequencies, 'accumulated_layers': accumulated_layers, 'words': word_index_map}, outputfile)
