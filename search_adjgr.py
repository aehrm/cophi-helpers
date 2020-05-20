#!/usr/bin/env python3
import csv
import sys
import re
import itertools
import collections

def groupstring(adj, k, sent):
    lines = []
    queue = []
    queue.append(k)
    while queue:
        i = queue.pop(0)
        lines.append(sent[i])
        for j in adj[i]:
            queue.append(j[1])
            
    return " ".join([l['token-syntok'] for l in sorted(lines, key=lambda l: l['index'])])

def parsetree(sent, field):
    kwindices = set()
    tree = collections.defaultdict(set)
    for i, line in enumerate(sent):
        if line['lemma-rnntagger'] in searchwords:
            kwindices.add(i)

        edge = eval(line[field])
        tree[i+edge[1]].add((edge[0], i))
    
    return tree, kwindices

def extract_adjgr_parzu(sent, searchwords):
    tree, kwindices = parsetree(sent, 'syntax-parzu')

    for k in kwindices:
        for x in [x for x in tree[k] if x[0] == 'attr']:
            groupstr = groupstring(tree, x[1], sent)
            yield (sent[k]['index'], sent[k]['token-syntok'], groupstr)
            
def extract_adjgr_spacy(sent, searchwords):
    tree, kwindices = parsetree(sent, 'syntax-spacy')

    for k in kwindices:
        for x in [x for x in tree[k] if x[0] == 'nk' and sent[x[1]]['pos-spacy'].startswith("ADJ")]:
            groupstr = groupstring(tree, x[1], sent)
            yield (sent[k]['index'], sent[k]['token-syntok'], groupstr)
                        
def extract_adjgr_corenlp(sent, searchwords):
    tree, kwindices = parsetree(sent, 'syntax-corenlp')

    for k in kwindices:
        for x in [x for x in tree[k] if x[0] == 'amod' and sent[x[1]]['pos-corenlp'].startswith("ADJ")]:
            groupstr = groupstring(tree, x[1], sent)
            yield (sent[k]['index'], sent[k]['token-syntok'], groupstr)

writer = csv.DictWriter(sys.stdout, fieldnames=['file', 'index', 'parser', 'token', 'adjgr'])

try:
    split_index = sys.argv.index('--')
except:
    print(f"Usage: {sys.argv[0]} searchword1 searchword2 ... -- file1 file2 ...", file=sys.stderr)
    sys.exit(1)

searchwords = sys.argv[1:split_index]

writer.writeheader()
            
for f in sys.argv[split_index+1:]:
    with open(f) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
        table = list(reader)
        for i, line in enumerate(table): line['index'] = i+1
        
        sents = []
        # find relevant sentences
        for s in [list(g) for k, g in itertools.groupby(table, lambda x: int(x['sentence-syntok']))]:
            for line in s:
                if line['lemma-rnntagger'] in searchwords:
                    sents.append(s)
                    break
                
        if len(sents) == 0: continue

        for sent in sents:
            for i, token, groupstr in extract_adjgr_corenlp(sent, searchwords):
                writer.writerow({'file': f, 'index': i, 'parser': 'corenlp', 'token': token, 'adjgr': groupstr})
            for i, token, groupstr in extract_adjgr_spacy(sent, searchwords):
                writer.writerow({'file': f, 'index': i, 'parser': 'spacy', 'token': token, 'adjgr': groupstr})
            for i, token, groupstr in extract_adjgr_parzu(sent, searchwords):
                writer.writerow({'file': f, 'index': i, 'parser': 'parzu', 'token': token, 'adjgr': groupstr})
                    
