#!/usr/bin/env python3

try:
    from lxml import etree
except ImportError:
    print("abort: lxml not installed", file=sys.stderr)
    sys.exit(1)
    
try:
    import syntok.segmenter as segmenter
except ImportError:
    print("abort: syntok not installed", file=sys.stderr)
    sys.exit(1)

import re
#from somajo import SoMaJo
import random
import json
import sys

def strip_ns_prefix(tree):
    #xpath query for selecting all element nodes in namespace
    query = "descendant-or-self::*[namespace-uri()!='']"
    #for each element returned by the above xpath query...
    for element in tree.xpath(query):
        #replace element name with its local name
        element.tag = etree.QName(element).localname
    return tree

def merge_whitespace(a, b):
    b = re.sub('\s+', ' ', b)
    if len(a) > 0 and a[-1] == ' ':
        return a + re.sub('^\s+', '', b)
    else:
        return a + b

def readtree(tree):
    body = tree.find('//body/ab')
    string = re.sub('^\s+', '', re.sub('\s+$', ' ', body.text)) # start with unenclosed text
    segments = []
    cursegment = None
    for el in body:
        if cursegment == None:
            cursegment = {'id': el.get('ana'), 'start': len(string)}


        string = merge_whitespace(string, el.text)

        if el.getnext() is None or el.getnext().get('ana') != cursegment['id']:
            cursegment['end'] = len(string)
            segments.append(cursegment)
            cursegment = None

        if el.tail is not None:
            string = merge_whitespace(string, el.tail)

    return (string, segments)

if len(sys.argv) < 3:
    print("usage: prepare_file.py <left file> <right file>", file=sys.stderr)
    sys.exit(1)

parser = etree.XMLParser(remove_blank_text=False)
tree_left = strip_ns_prefix(etree.parse(sys.argv[1], parser))
tree_right = strip_ns_prefix(etree.parse(sys.argv[2], parser))
(lstring, lsegments) = readtree(tree_left)
(rstring, rsegments) = readtree(tree_right)

for i in range(len(lstring)):
    if lstring[i] != rstring[i]:
        print("abort: normalized text missmatches on char index " + i, file=sys.stderr)
        sys.exit(1)

string = lstring
sentences = [{'start': sentence[0]._offset, 'end': sentence[-1]._offset + len(sentence[-1].value)}
        for paragraph in segmenter.analyze(string) for sentence in paragraph]

# TODO implement alignments
alignment_scores = [{'name': 'random', 'scores': [random.random() for s in sentences]},
                    {'name': 'random2', 'scores': [random.random() for s in sentences]}]

output = {'sentences': sentences,
        'alignment_scores': alignment_scores,
        'text': string,
        'left_segments': {seg['id']:seg for seg in lsegments},
        'right_segments': {seg['id']:seg for seg in rsegments}
        }

json.dump(output, sys.stdout)
