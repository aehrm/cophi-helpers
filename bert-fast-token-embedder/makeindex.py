import os.path
import sys
from os import listdir
from tqdm import tqdm

import syntok.segmenter as segmenter
from transformers import BertTokenizer
from whoosh import index
from whoosh.analysis import RegexTokenizer
from whoosh.fields import Schema, ID, TEXT

# change tokenizer if desired
tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")


def get_schema():
    return Schema(filename=ID(unique=True, stored=True), content=TEXT(phrase=True, analyzer=RegexTokenizer(r"[^ \n]+")))


def add_doc(writer, path, processed_doc_path):
    fileobj = open(path, "r")
    content = fileobj.read()
    fileobj.close()

    # tokenize
    tokenized_str = ''
    for sent in [sent for para in segmenter.analyze(content) for sent in para]:
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer([t.value for t in sent], is_pretokenized=True, add_special_tokens=False)['input_ids'])
        tokenized_str += ' '.join(tokens) + '\n'

    filename = os.path.basename(path)
    out = open(processed_doc_path, 'w')
    print(tokenized_str, file=out)
    out.close()

    writer.add_document(filename=filename, content=tokenized_str)


def clean_index(index_dir, corpus_path, processed_path):
    # Always create the index from scratch
    ix = index.create_in(index_dir, schema=get_schema())
    writer = ix.writer()

    for path in tqdm([os.path.join(corpus_path, f) for f in listdir(corpus_path) if
                 os.path.isfile(os.path.join(corpus_path, f))]):
        filename = os.path.basename(path)
        processed_doc_path = os.path.join(processed_path, filename)
        add_doc(writer, path, processed_doc_path)
        tqdm.write(f"{path} tokenized and written to {processed_doc_path}")

    writer.commit()


if len(sys.argv) < 4:
    print("Usage: python makeindex.py corpus_dir processed_corpus_dir index_dir", file=sys.stderr)
    sys.exit(-1)

corpus_dir = sys.argv[1]
processed_corpus_dir = sys.argv[2]
index_dir = sys.argv[3]

if not os.path.exists(index_dir):
    os.mkdir(index_dir)

if not os.path.exists(processed_corpus_dir):
    os.mkdir(processed_corpus_dir)

clean_index(index_dir, corpus_dir, processed_corpus_dir)
