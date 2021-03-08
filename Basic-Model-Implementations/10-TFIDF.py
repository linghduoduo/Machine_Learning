import string
from math import log10


def term_frequency(term: str, document: str) -> int:
    """
    Return the number of times a term occurs within
    a given document.
    @params: term, the term to search a document for, and document,
            the document to search within
    @returns: an integer representing the number of times a term is
            found within the document
    @examples:
    >>> term_frequency("to", "To be, or not to be")
    2
    """
    # strip all punctuation and newlines and replace it with ''
    document_without_punctuation = document.translate(
        str.maketrans("", "", string.punctuation)
    ).replace("\n", "")
    tokenize_document = document_without_punctuation.split(" ")  # word tokenization
    return len([word for word in tokenize_document if word.lower() == term.lower()])


def document_frequency(term: str, corpus: str) -> int:
    """
    Calculate the number of documents in a corpus that contain a
    given term
    @params : term, the term to search each document for, and corpus, a collection of
             documents. Each document should be separated by a newline.
    @returns : the number of documents in the corpus that contain the term you are
               searching for and the number of documents in the corpus
    @examples :
    >>> document_frequency("first", 
    "This is the first document in the corpus.\\n
    ThIs is the second document in the corpus.\\n
    THIS is the third document in the corpus.")
    (1, 3)
    """
    corpus_without_punctuation = corpus.lower().translate(
        str.maketrans("", "", string.punctuation)
    )  # strip all punctuation and replace it with ''
    docs = corpus_without_punctuation.split("\n")
    term = term.lower()
    return (len([doc for doc in docs if term in doc]), len(docs))


def inverse_document_frequency(df: int, N: int, smoothing=False) -> float:
    """
    Return an integer denoting the importance
    of a word. This measure of importance is
    calculated by log10(N/df), where N is the
    number of documents and df is
    the Document Frequency.
    @params : df, the Document Frequency, N,
    the number of documents in the corpus and
    smoothing, if True return the idf-smooth
    @returns : log10(N/df) or 1+log10(N/1+df)
    @examples :
    >>> inverse_document_frequency(3, 0)
    Traceback (most recent call last):
     ...
    ValueError: log10(0) is undefined.
    >>> inverse_document_frequency(1, 3)
    0.477
    >>> inverse_document_frequency(0, 3)
    Traceback (most recent call last):
     ...
    ZeroDivisionError: df must be > 0
    >>> inverse_document_frequency(0, 3,True)
    1.477
    """
    if smoothing:
        if N == 0:
            raise ValueError("log10(0) is undefined.")
        return round(1 + log10(N / (1 + df)), 3)

    if df == 0:
        raise ZeroDivisionError("df must be > 0")
    elif N == 0:
        raise ValueError("log10(0) is undefined.")
    return round(log10(N / df), 3)


def tf_idf(tf: int, idf: int) -> float:
    """
    Combine the term frequency
    and inverse document frequency functions to
    calculate the originality of a term. This
    'originality' is calculated by multiplying
    the term frequency and the inverse document
    frequency : tf-idf = TF * IDF
    @params : tf, the term frequency, and idf, the inverse document
    frequency
    @examples :
    >>> tf_idf(2, 0.477)
    0.954
    """
    return round(tf * idf, 3)


class TFIDFEncoder:
    def __init__(
        self,
        vocab=None,
        lowercase=True,
        min_count=0,
        smooth_idf=True,
        max_tokens=None,
        input_type="filename",
        filter_stopwords=True,
    ):
        r"""
        An object for compiling and encoding the term-frequency
        inverse-document-frequency (TF-IDF) representation of the tokens in a
        text corpus.
        Notes
        -----
        TF-IDF is intended to reflect how important a word is to a document in
        a collection or corpus. For a word token `w` in a document `d`, and a
        corpus, :math:`D = \{d_1, \ldots, d_N\}`, we have:
        .. math::
            \text{TF}(w, d)  &=  \text{num. occurences of }w \text{ in document }d \\
            \text{IDF}(w, D)  &=  \log \frac{|D|}{|\{ d \in D: t \in d \}|}
        Parameters
        ----------
        vocab : :class:`Vocabulary` object or list-like
            An existing vocabulary to filter the tokens in the corpus against.
            Default is None.
        lowercase : bool
            Whether to convert each string to lowercase before tokenization.
            Default is True.
        min_count : int
            Minimum number of times a token must occur in order to be included
            in vocab. Default is 0.
        smooth_idf : bool
            Whether to add 1 to the denominator of the IDF calculation to avoid
            divide-by-zero errors. Default is True.
        max_tokens : int
            Only add the `max_tokens` most frequent tokens that occur more
            than `min_count` to the vocabulary.  If None, add all tokens
            greater that occur more than than `min_count`. Default is None.
        input_type : {'filename', 'strings'}
            If 'files', the sequence input to `fit` is expected to be a list
            of filepaths. If 'strings', the input is expected to be a list of
            lists, each sublist containing the raw strings for a single
            document in the corpus. Default is 'filename'.
        filter_stopwords : bool
            Whether to remove stopwords before encoding the words in the
            corpus. Default is True.
        """
        # create a function to filter against words in the vocab
        self._filter_vocab = lambda words: words
        if isinstance(vocab, Vocabulary):
            self._filter_vocab = vocab.filter
        elif isinstance(vocab, (list, np.ndarray, set)):
            vocab = set(vocab)
            self._filter_vocab = lambda words: [
                w if w in vocab else "<unk>" for w in words
            ]

        if input_type not in ["files", "strings"]:
            fstr = "`input_type` must be either 'files' or 'strings', but got {}"
            raise ValueError(fstr.format(input_type))

        self._tokens = None
        self._idx2doc = None
        self.term_freq = None
        self.idx2token = None
        self.token2idx = None
        self.inv_doc_freq = None

        self.hyperparameters = {
            "id": "TFIDFEncoder",
            "encoding": None,
            "vocab": vocab
            if not isinstance(vocab, Vocabulary)
            else vocab.hyperparameters,
            "lowercase": lowercase,
            "min_count": min_count,
            "input_type": input_type,
            "max_tokens": max_tokens,
            "smooth_idf": smooth_idf,
            "filter_stopwords": filter_stopwords
            if not isinstance(vocab, Vocabulary)
            else vocab.hyperparameters["filter_stopwords"],
        }

    def fit(self, corpus_seq, encoding="utf-8-sig"):
        """
        Compute term-frequencies and inverse document frequencies on a
        collection of documents.
        Parameters
        ----------
        corpus_seq : str or list of strs
            The filepath / list of filepaths / raw string contents of the
            document(s) to be encoded, in accordance with the `input_type`
            parameter passed to the :meth:`__init__` method. Each document is
            expected to be a newline-separated strings of text, with adjacent
            tokens separated by a whitespace character.
        encoding : str
            Specifies the text encoding for corpus if `input_type` is `files`.
            Common entries are either 'utf-8' (no header byte), or 'utf-8-sig'
            (header byte). Default is 'utf-8-sig'.
        """
        H = self.hyperparameters

        if isinstance(corpus_seq, str):
            corpus_seq = [corpus_seq]

        if H["input_type"] == "files":
            for corpus_fp in corpus_seq:
                assert op.isfile(corpus_fp), "{} does not exist".format(corpus_fp)

        tokens = []
        idx2token, token2idx = {}, {}

        # encode special tokens
        for tt in ["<bol>", "<eol>", "<unk>"]:
            token2idx[tt] = len(tokens)
            idx2token[len(tokens)] = tt
            tokens.append(Token(tt))

        min_count = H["min_count"]
        max_tokens = H["max_tokens"]
        H["encoding"] = encoding

        bol_ix = token2idx["<bol>"]
        eol_ix = token2idx["<eol>"]
        idx2doc, term_freq = {}, {}

        # encode the text in `corpus_fps` without any filtering ...
        for d_ix, doc in enumerate(corpus_seq):
            doc_count = {}
            idx2doc[d_ix] = doc if H["input_type"] == "files" else None
            token2idx, idx2token, tokens, doc_count = self._encode_document(
                doc, token2idx, idx2token, tokens, doc_count, bol_ix, eol_ix,
            )
            term_freq[d_ix] = doc_count

        self._tokens = tokens
        self._idx2doc = idx2doc
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.term_freq = term_freq

        # ... retain only the top `max_tokens` most frequent tokens, coding
        # everything else as <unk> ...
        if max_tokens is not None and len(tokens) > max_tokens:
            self._keep_top_n_tokens()

        # ... replace all words occurring less than `min_count` by <unk> ...
        if min(self._tokens, key=lambda t: t.count).count < min_count:
            self._drop_low_freq_tokens()

        # ... sort tokens alphabetically and reindex ...
        self._sort_tokens()

        # ... finally, calculate inverse document frequency
        self._calc_idf()

    def _encode_document(
        self, doc, word2idx, idx2word, tokens, doc_count, bol_ix, eol_ix,
    ):
        """Perform tokenization and compute token counts for a single document"""
        H = self.hyperparameters
        lowercase = H["lowercase"]
        filter_stop = H["filter_stopwords"]

        if H["input_type"] == "files":
            with open(doc, "r", encoding=H["encoding"]) as handle:
                doc = handle.read()

        n_words = 0
        lines = doc.split("\n")
        for line in lines:
            words = tokenize_words(line, lowercase, filter_stop)
            words = self._filter_vocab(words)
            n_words += len(words)

            for ww in words:
                if ww not in word2idx:
                    word2idx[ww] = len(tokens)
                    idx2word[len(tokens)] = ww
                    tokens.append(Token(ww))

                t_idx = word2idx[ww]
                tokens[t_idx].count += 1
                doc_count[t_idx] = doc_count.get(t_idx, 0) + 1

            # wrap line in <bol> and <eol> tags
            tokens[bol_ix].count += 1
            tokens[eol_ix].count += 1

            doc_count[bol_ix] = doc_count.get(bol_ix, 0) + 1
            doc_count[eol_ix] = doc_count.get(eol_ix, 0) + 1
        return word2idx, idx2word, tokens, doc_count

    def _keep_top_n_tokens(self):
        N = self.hyperparameters["max_tokens"]
        doc_counts, word2idx, idx2word = {}, {}, {}
        tokens = sorted(self._tokens, key=lambda x: x.count, reverse=True)

        # reindex the top-N tokens...
        unk_ix = None
        for idx, tt in enumerate(tokens[:N]):
            word2idx[tt.word] = idx
            idx2word[idx] = tt.word

            if tt.word == "<unk>":
                unk_ix = idx

        # ... if <unk> isn't in the top-N, add it, replacing the Nth
        # most-frequent word and adjust the <unk> count accordingly ...
        if unk_ix is None:
            unk_ix = self.token2idx["<unk>"]
            old_count = tokens[N - 1].count
            tokens[N - 1] = self._tokens[unk_ix]
            tokens[N - 1].count += old_count
            word2idx["<unk>"] = N - 1
            idx2word[N - 1] = "<unk>"

        # ... and recode all dropped tokens as "<unk>"
        for tt in tokens[N:]:
            tokens[unk_ix].count += tt.count

        # ... finally, reindex the word counts for each document
        doc_counts = {}
        for d_ix in self.term_freq.keys():
            doc_counts[d_ix] = {}
            for old_ix, d_count in self.term_freq[d_ix].items():
                word = self.idx2token[old_ix]
                new_ix = word2idx.get(word, unk_ix)
                doc_counts[d_ix][new_ix] = doc_counts[d_ix].get(new_ix, 0) + d_count

        self._tokens = tokens[:N]
        self.token2idx = word2idx
        self.idx2token = idx2word
        self.term_freq = doc_counts

        assert len(self._tokens) <= N

    def _drop_low_freq_tokens(self):
        """
        Replace all tokens that occur less than `min_count` with the `<unk>`
        token.
        """
        H = self.hyperparameters
        unk_token = self._tokens[self.token2idx["<unk>"]]
        eol_token = self._tokens[self.token2idx["<eol>"]]
        bol_token = self._tokens[self.token2idx["<bol>"]]
        tokens = [unk_token, eol_token, bol_token]

        unk_idx = 0
        word2idx = {"<unk>": 0, "<eol>": 1, "<bol>": 2}
        idx2word = {0: "<unk>", 1: "<eol>", 2: "<bol>"}
        special = {"<eol>", "<bol>", "<unk>"}

        for tt in self._tokens:
            if tt.word not in special:
                if tt.count < H["min_count"]:
                    tokens[unk_idx].count += tt.count
                else:
                    word2idx[tt.word] = len(tokens)
                    idx2word[len(tokens)] = tt.word
                    tokens.append(tt)

        # reindex document counts
        doc_counts = {}
        for d_idx in self.term_freq.keys():
            doc_counts[d_idx] = {}
            for old_idx, d_count in self.term_freq[d_idx].items():
                word = self.idx2token[old_idx]
                new_idx = word2idx.get(word, unk_idx)
                doc_counts[d_idx][new_idx] = doc_counts[d_idx].get(new_idx, 0) + d_count

        self._tokens = tokens
        self.token2idx = word2idx
        self.idx2token = idx2word
        self.term_freq = doc_counts

    def _sort_tokens(self):
        # sort tokens alphabetically and recode
        ix = 0
        token2idx, idx2token, = {}, {}
        special = ["<eol>", "<bol>", "<unk>"]
        words = sorted(self.token2idx.keys())
        term_freq = {d: {} for d in self.term_freq.keys()}

        for w in words:
            if w not in special:
                old_ix = self.token2idx[w]
                token2idx[w], idx2token[ix] = ix, w
                for d in self.term_freq.keys():
                    if old_ix in self.term_freq[d]:
                        count = self.term_freq[d][old_ix]
                        term_freq[d][ix] = count
                ix += 1

        for w in special:
            token2idx[w] = len(token2idx)
            idx2token[len(idx2token)] = w

        self.token2idx = token2idx
        self.idx2token = idx2token
        self.term_freq = term_freq
        self.vocab_counts = Counter({t.word: t.count for t in self._tokens})

    def _calc_idf(self):
        """
        Compute the (smoothed-) inverse-document frequency for each token in
        the corpus.
        For a word token `w`, the IDF is simply
            IDF(w) = log ( |D| / |{ d in D: w in d }| ) + 1
        where D is the set of all documents in the corpus,
            D = {d1, d2, ..., dD}
        If `smooth_idf` is True, we perform additive smoothing on the number of
        documents containing a given word, equivalent to pretending that there
        exists a final D+1st document that contains every word in the corpus:
            SmoothedIDF(w) = log ( |D| + 1 / [1 + |{ d in D: w in d }|] ) + 1
        """
        inv_doc_freq = {}
        smooth_idf = self.hyperparameters["smooth_idf"]
        tf, doc_idxs = self.term_freq, self._idx2doc.keys()

        D = len(self._idx2doc) + int(smooth_idf)
        for word, w_ix in self.token2idx.items():
            d_count = int(smooth_idf)
            d_count += np.sum([1 if w_ix in tf[d_ix] else 0 for d_ix in doc_idxs])
            inv_doc_freq[w_ix] = 1 if d_count == 0 else np.log(D / d_count) + 1
        self.inv_doc_freq = inv_doc_freq

    def transform(self, ignore_special_chars=True):
        """
        Generate the term-frequency inverse-document-frequency encoding of a
        text corpus.
        Parameters
        ----------
        ignore_special_chars : bool
            Whether to drop columns corresponding to "<eol>", "<bol>", and
            "<unk>" tokens from the final tfidf encoding. Default is True.
        Returns
        -------
        tfidf : numpy array of shape `(D, M [- 3])`
            The encoded corpus, with each row corresponding to a single
            document, and each column corresponding to a token id. The mapping
            between column numbers and tokens is stored in the `idx2token`
            attribute IFF `ignore_special_chars` is False. Otherwise, the
            mappings are not accurate.
        """
        D, N = len(self._idx2doc), len(self._tokens)
        tf = np.zeros((D, N))
        idf = np.zeros((D, N))

        for d_ix in self._idx2doc.keys():
            words, counts = zip(*self.term_freq[d_ix].items())
            docs = np.ones(len(words), dtype=int) * d_ix
            tf[docs, words] = counts

        words = sorted(self.idx2token.keys())
        idf = np.tile(np.array([self.inv_doc_freq[w] for w in words]), (D, 1))
        tfidf = tf * idf

        if ignore_special_chars:
            idxs = [
                self.token2idx["<unk>"],
                self.token2idx["<eol>"],
                self.token2idx["<bol>"],
            ]
            tfidf = np.delete(tfidf, idxs, 1)

        return tfidf
