import collections
import csv
import os.path
import sys
from os import listdir

import more_itertools
import re
from tqdm import tqdm

frequencies = collections.defaultdict(int)


def insert_pieces(pieces):
    word = pieces[0] + ''.join(list(map(lambda p: p[2:], pieces[1:])))
    frequencies[word] += 1


def process_file(f):
    fileobj = open(f, 'r')
    content = fileobj.read().split()
    fileobj.close()
    for pieces in more_itertools.split_before(content, lambda x: not x.startswith('##')):
        insert_pieces(pieces)


if len(sys.argv) < 2:
    print("Usage: python frequencies.py processed_corpus", file=sys.stderr)
    sys.exit(-1)

path = sys.argv[1]

for f in tqdm(list(os.path.join(path, f) for f in listdir(path) if os.path.isfile(os.path.join(path, f)))):
    process_file(f)

entries = list(frequencies.items())
entries.sort(key=lambda x: x[1], reverse=True)

writer = csv.writer(sys.stdout, delimiter='\t')

punct_regex = re.compile(r"(\W|[0-9])")
for e in entries:
    if punct_regex.search(e[0]):  # remove punctuation
        continue
    writer.writerow(e)
