import codecs
import itertools
import os
import sys

import more_itertools
import torch
import whoosh.index
import whoosh.query
from torch.nn.utils.rnn import pad_sequence


class BertEmbedder:

    def __init__(self, indexdir, docdir, cache_pickle, bert_model, bert_tokenizer):
        self.index = whoosh.index.open_dir(indexdir)
        self.docdir = docdir
        self.model = bert_model
        self.tokenizer = bert_tokenizer

        assert self.model.config.output_hidden_states

        self.cache = torch.load(cache_pickle)
        assert 'frequencies' in self.cache.keys()
        assert 'accumulated_layers' in self.cache.keys()
        assert 'words' in self.cache.keys()

        self.localcache = dict()

    def embed(self, word):
        if word in self.localcache:
            return self.localcache[word]
        elif word in self.cache['words']:
            return self.read_cached_embedding(word)
        else:
            ret = self.compute_embedding(word)
            self.localcache[word] = ret
            return ret

    def read_cached_embedding(self, word):
        wordidx = self.cache['words'][word]
        cached = self.cache['accumulated_layers'][wordidx] / self.cache['frequencies'][wordidx]

        lastnonzero = torch.nonzero(cached)[-1][0].item()
        return cached[0:lastnonzero + 1]

    def compute_embedding(self, word):
        return torch.stack(list(
            more_itertools.collapse(map(self.forwardpass, more_itertools.chunked(self.query_index(word), 10)),
                                    levels=1))).mean(0)

    def tokenize(self, word):
        return self.tokenizer.tokenize(word)

    def query_index(self, word):
        searchtokens = self.tokenize(word)
        terms = [whoosh.query.Term('content', x) for x in searchtokens]

        whoosh.query.Or.matcher_type = 1
        q = whoosh.query.spans.SpanNear2(terms)

        with self.index.searcher() as s:
            matcher = q.matcher(s)

            while matcher.is_active():
                filename = s.stored_fields(matcher.id())['filename']
                doc = self._read_doc(filename)
                for span in matcher.spans():
                    out = self._process_context(doc, span.start, span.end)
                    if out is not None:
                        yield out

                matcher.next()

    def _read_doc(self, filename):
        path = os.path.join(self.docdir, filename)
        fileobj = codecs.open(path, "rb", "utf-8")
        content = fileobj.read()
        fileobj.close()

        return [line.split(' ') for line in content.split('\n')]

    def _process_context(self, doc, tokenstart, tokenend):
        sentidxs = [0] + list(itertools.accumulate(len(sent) for sent in doc))[:-1]
        sentidx = [i for i, j in enumerate(sentidxs) if j <= tokenstart][-1]

        context = doc[sentidx]
        tokenstart = tokenstart - sentidxs[sentidx]
        tokenend = tokenend - sentidxs[sentidx]

        if tokenend + 1 < len(context) and context[tokenend + 1].startswith('##'):
            return None

        if len(context) > 510:
            # exceptional behavior: center context independent of sentence boundaries
            print("context sentence longer than 510 tokens, centering on focus word", file=sys.stderr)

            # extend context on the left
            i = sentidx - 1
            while i > 0 and tokenstart < 260:
                context = doc[i] + context
                tokenstart += len(doc[i])
                tokenend += len(doc[i])
                i = i - 1
            # extend context on the right
            i = sentidx + 1
            while i - 1 < len(doc) and len(context) - tokenend < 260:
                context = context + doc[i]
                i = i + 1

            # extract context window
            contextstart = tokenstart - 250
            contextend = tokenstart + 250
            return (context[contextstart:contextend], 250, 250 + tokenend - tokenstart)

        # regular behavior: add sentences left and right until max size reached
        breakleft = False
        breakright = False
        indices = [(-1) ** ((j + 1) % 2) * (j // 2) for j in range(0, len(doc))]  # -1, 1, -2, 2, ...
        for j, i in enumerate(indices):
            if breakleft and breakright:
                break
            if j % 2 == 0 and breakleft:
                continue
            if j % 2 == 1 and breakright:
                continue

            if i not in range(0, len(doc)):
                if j % 2 == 0:
                    breakleft = True
                    continue
                else:
                    breakright = True
                    continue

            if len(context) + len(doc[i]) <= 510:
                if j % 2 == 0:
                    context = doc[i] + context
                    tokenstart += len(doc[i])
                    tokenend += len(doc[i])
                else:
                    context = context + doc[i]
            else:
                if j % 2 == 0:
                    breakleft = True
                else:
                    breakright = True

        return (context, tokenstart, tokenend)

    def forwardpass(self, contexts):
        indexed_tokens = [
            self.tokenizer.build_inputs_with_special_tokens(self.tokenizer.convert_tokens_to_ids(context[0]))
            for context in contexts]
        tokens_tensor = pad_sequence(list(map(torch.tensor, indexed_tokens)), batch_first=True)
        attention_tensor = (tokens_tensor != 0).int().to(self.model.device)
        tokens_tensor = tokens_tensor.to(self.model.device)

        outputs = self.model(tokens_tensor, attention_mask=attention_tensor)
        inner_layers = outputs[2]
        permuted = torch.stack(list(inner_layers)).permute(1, 2, 0, 3)

        # offset by one due to [CLS]
        return [permuted[i].narrow(0, 1 + contexts[i][1], contexts[i][2] - contexts[i][1] + 1) for i in
                range(len(contexts))]
