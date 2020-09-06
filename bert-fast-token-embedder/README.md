# Fast word embedding from BERT-processed contexts

## Summary 

As a context-sensitive language model, [BERT](https://arxiv.org/abs/1810.04805) takes as input a larger text of limited size (e.g. sentence) and provides a contextualized vector embedding for each token of the text.
This is in contrast to context-free language models (e.g. [fasttext](https://github.com/facebookresearch/fastText)), which map tokens to a vector without further information on the context given.

We present a context-free word embedding generated from averaged BERT-processed embeddings of word contexts, in order to analyze and compare the structural properties of the resulting vector space.

## Model

Given a corups, a concordance lists for each word from the corpus the immediate contexts of the respective word.
We expect for an (at least non-ambiguous) word that their respective occurences in the corups are surrounded by semantically similar contexts.

In a BERT embedding, we expect that the vector to each token is influenced by the tokens surrounding it.
Hence, given a focus word, and embedding each of the contexts from the concordance for that focus word,  the set of vectors corresponding to the focus word describes that word taking into consideration the contexts given from the corpus.

We finalize by aggregating that set of vectors by taking the arithmetic mean.

## Implementation Design

We face two difficulties in implementing that approach:

* The BERT language model uses a [WordPiece](https://arxiv.org/abs/1609.08144) sequence as input tokens (further referred to as *piece*).
This embedding maps each word to at least one piece from a fixed vocabulary (size approx. 30k).
Note that given the embedding of multiple pieces constituting a single word, we cannot infer a sensible embedding of that word.

* Especially with a high-frequency focus word, the above outlined aggregation can only be performed with significant computation time.
Hence, we wish to preprocess and cache as many word-vector associations as possible.
In general, we cannot apply this to every unique word of the corpus due to the large number.
This implies that we face a space-time tradeoff.

We continue by presenting the implementation in three steps: first, a preprocessing accelerating the following algorithms, second, a description of the implementation of above outlined procedure (i.e. given an out-of-cache word, compute embedding), third, an efficient procedure to generate the cache.

### Concordance Generation

As it is central to efficiently determine the BERT-prepared context sequences of a given focus word, we preprocess given corpus.
Assume the corpus as a set of documents consisting only text.

First, for each document, split the text into sentences (here, it is implemented using [Syntok](https://github.com/fnl/syntok)).
Then, each sentence can be transformed to a piece sequence.
This results in a document representation consisting of a piece sequence, that also allows the sentence boundaries to be read.
The preprocessed documents can subsequently written on disk.

(In our implementation, the pieces are stored in their ASCII respresentation, separated by whitespace, and sentences separated by newline.)

Now, the preprocessed documents can be indexed to allow an efficient search for focus words:
Generate a reverse index that maps each piece to a set of (document, position) tuples specifying the occurrences.
By indexing on the pieces (and not words), the resulting index remains relatively small (due to the small number of unique pieces).

Storing the positions of the occurrences allows the efficient search for term phrases (i.e. words tokenized into >1 pieces) using the reverse index.

(We implement the reverse index using the Python [Whoosh](https://github.com/mchaput/whoosh) package)

### Cache Miss Procedure

Assuming the given word is not present in the cache, we outline the implementation generating the aggregated embedding.

* Compute the piece sequence corresponding to the focus word.
* Determine the positions of that sequence in the preprocessed corpus using the reverse index.
* In order to generate contexts from occurences, start with the sentence containing the position, and then extend the context window on the left and right side, by adding preceding and succeeding tokenized sentences, until BERT's input size limit is reached.
Hence, the tokenized focus word is approximately in the center of the context.
* Add the `[CLS]` and `[SEP]` tokens to each context to finalize the input for BERT.
Keeping track of the indices, we have, for each context sequence 𝑠, the range [𝑙<sub>𝑠</sub>, 𝑟<sub>𝑠</sub>] that indicates the boundary of the focus word subsequence.
* Forward-pass each context sequence 𝑠 through the BERT model, and then extract the embedding of the tokens in the range [𝑙<sub>𝑠</sub>, 𝑟<sub>𝑠</sub>].
Assuming that each piece is embedded into a [13, 768]-tensor (12 hidden layers + 1 output layer, 768 parameters per layer), we obtain a [𝑛, 𝑙, 13, 768]-tensor, where 𝑛 refers to the number of contexts, 𝑙 the length of the focus word sequence.
* Taking the mean along the first axis, we obtain a [𝑙, 13, 768]-tensor as embedding for the initial focus word.

### Cache Generation

From the preprocessed documents, the cache can easily be generated by a single forward pass of the entire corpus.
For an optimal cache, determine the frequency of each word in the corpus, and take the 𝑛 most frequent ones.
These tokens require the most processing time if they were not cached.

Generate input sequences by adding subsequent sentences until reaching size limit.
Perform the forward passes.
To determine the embeddings of each word (not piece), we exploit the reversability of the WordPiece tokenization.
That is, we can partition each input sequence into subsequences representing the individual words.  

Taking the same subsequences on the output yields embeddings of the respective words. Accumulating the embeddings of the chosen cache words, and dividing by frequency, results in the desired cache.


## Usage

### Preparation

```bash
export CORPUS=./corpus                     # directory of plain text files
export PROCESSED_CORPUS=./processed_corpus # empty dir
export INDEXDIR=./indexdir                 # empty dir
export CACHE=./cachefile.pl                # name of cachefile to generate

pip install -r requirements.txt

# preprocess corpus
python makeindex.py $CORPUS $PROCESSED_CORPUS $INDEXDIR

# determine frequencies
python frequencies.py $PROCESSED_CORPUS > frequencies.tsv

# get cache words (10k most frequent ones)
head -n 10000 frequencies.tsv | cut -f1 > cachewords

# generate cache
python makecache.py cachewords $PROCESSED_CORPUS $CACHE
```

### Python API

See `embedder.py` file:
* `BertEmbedder(indexdir, docdir, cache_picle, bert_model, bert_tokenizer)`  
  Embedder constructor, give `indexdir`, `docdir`, `cache_pickle` as strings; `bert_model` resp. `bert_tokenizer` as [`transformers.BertModel`](https://huggingface.co/transformers/model_doc/bert.html#bertmodel) resp. [`transformers.BertTokenizer`](https://huggingface.co/transformers/model_doc/bert.html#berttokenizer).
  
* `BertEmbedder.compute_embedding(self, word)`  
  Given string `word`, computes the output embedding as outlined in Sec. *Cache Miss Procedure*, using the index and processed corpus.
  
* `BertEmbedder.read_cached_embedding(self, word)`  
  Given string `word`, reads the cached embedding `word`, or returns `None` if not in the cache
  
* `BertEmbedder.embed(self, word)`  
  Determines if string `word` is present in the cache, and invokes one of the above respective functions. 
  
### Example

```python
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from embedder import BertEmbedder

config = BertConfig.from_pretrained("bert-base-german-cased", output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
model = BertModel.from_pretrained("bert-base-german-cased", config=config)
model.to('cuda')

model.eval()
torch.set_grad_enabled(False)

embedder = BertEmbedder(indexdir='indexdir', docdir='processed_corpus',
        cache_pickle='cachefile.pl', bert_model=model, bert_tokenizer=tokenizer)

# embed word
embedder.embed('Unmöglichkeit')
# observe that the token is split into two pieces: ['Un', '##möglichkeit']
# returns tensor of shape [2, 13, 768]:
# tensor([[[-0.3489,  0.5044,  0.3678,  ..., -1.5239, -0.9979, -0.3289],
#          [-0.2173,  0.7550,  0.1045,  ..., -0.9457, -0.7756, -0.2617],
#          [ 0.2841,  0.3570, -0.4009,  ..., -0.6682, -0.4641, -0.5449],
#          ...,
#          [ 0.6555,  0.6270, -0.7339,  ...,  1.4157,  0.3457, -0.3370],
#          [ 0.4427,  0.3215, -0.5317,  ...,  1.0440,  0.4267, -0.2955],
#          [ 0.6560,  0.3177,  0.2431,  ...,  0.7445,  0.6516, -0.4647]],
#
#         [[ 0.6231, -0.1010,  0.9392,  ...,  0.4587, -0.5650, -0.7968],
#          [-0.0180, -0.0901,  0.2420,  ...,  0.6077, -0.1154, -0.3848],
#          [ 0.1130, -0.0888, -0.1231,  ...,  0.3450,  0.5666, -0.1898],
#          ...,
#          [-0.1236,  0.8450, -0.7257,  ...,  0.4887,  0.2776, -0.5623],
#          [-0.4489,  0.5725, -0.5801,  ...,  0.4962,  0.5719, -0.6368],
#          [-0.2603,  0.7459,  0.0532,  ...,  0.5482,  0.6203, -0.6134]]])
```
