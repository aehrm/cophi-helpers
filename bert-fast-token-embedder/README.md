# Fast word embedding from BERT-processed contexts

## Summary 

As a context-sensitive language model, [BERT]() takes as input a larger text of limited size (e.g. sentence) and provides a contextualized vector embedding for each token of the text.
This is in contrast to context-free language models (e.g. [fasttext]()), which map tokens to a vector without further information on the context given.

We present a context-free word embedding generated from averaged BERT-processed embeddings of word contexts, in order to analyze and compare the structural properties of the resulting vector space.

## Model

Given a corups, a concordance lists for each word from the corpus the immediate contexts of the respective word.
We expect for an (at least non-ambiguous) word that their respective occurences in the corups are surrounded by semantically similar contexts.

In a BERT embedding, we expect that the vector to each token is influenced by the tokens surrounding it.
Hence, given a focus word, and embedding each of the contexts from the concordance for that focus word,  the set of vectors corresponding to the focus word describes that word taking into consideration the contexts given from the corpus.

We finalize by aggregating that set of vectors by taking the arithmetic mean.

## Implementation Design

We face two difficulties in implementing that approach:

(a) The BERT language model uses a [WordPiece]() sequence as input tokens (further referred to as *piece*).
This embedding maps each word to at least one piece from a fixed vocabulary (size approx. 30k).
Note that given the embedding of multiple pieces constituting a single word, we cannot infer a sensible embedding of that word.

(b) Especially with a high-frequency focus word, the above outlined aggregation can only be performed with significant computation time.
Hence, we wish to preprocess and cache as many word-vector associations as possible.
In general, we cannot apply this to every unique word of the corpus due to the large number.
This implies that we face a space-time tradeoff.

We continue by presenting the implementation in three steps: first, a preprocessing accelerating the following algorithms, second, a description of the implementation of above outlined procedure (i.e. given an out-of-cache word, compute embedding), third, an efficient procedure to generate the cache.

### Concordance Generation

As it is central to efficiently determine the BERT-prepared context sequences of a given focus word, we preprocess given corpus.
Assume the corpus as a set of documents consisting only text.

First, for each document, split the text into sentences (here, it is implemented using [Syntok]()).
Then, each sentence can be transformed to a piece sequence.
This results in a document representation consisting of a piece sequence, that also allows the sentence boundaries to be read.
The preprocessed documents can subsequently written on disk.

(In our implementation, the pieces are stored in their ASCII respresentation, separated by whitespace, and sentences separated by newline.)

Now, the preprocessed documents can be indexed to allow an efficient search for focus words:
Generate a reverse index that maps each piece to a set of (document, position) tuples specifying the occurrences.
By indexing on the pieces (and not words), the resulting index remains relatively small (due to the small number of unique pieces).

Storing the positions of the occurrences allows the efficient search for term phrases (i.e. words tokenized into >1 pieces) using the reverse index.

TODO N-gram space-time tradeoff

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
For an optimal cache, determine the frequency of each token in the (unprocessed) corpus, and take the 𝑛 most frequent ones.
These tokens require the most processing time if they were not cached.

Generate input sequences by adding subsequent sentences until reaching size limit.
Perform the forward passes.
To determine the embeddings of each word (not piece), we exploit the reversability of the WordPiece tokenization.
That is, we can partition each input sequence into subsequences representing the individual words.  

Taking the same subsequences on the output yields embeddings of the respective words. Accumulating the embeddings of the chosen cache words, and dividing by frequency, results in the desired cache.

