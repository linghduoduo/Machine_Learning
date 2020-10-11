import string
from math import log10

"""
    tf-idf Wikipedia: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    tf-idf and other word frequency algorithms are often used
    as a weighting factor in information retrieval and text
    mining. 83% of text-based recommender systems use
    tf-idf for term weighting. In Layman's terms, tf-idf
    is a statistic intended to reflect how important a word
    is to a document in a corpus (a collection of documents)
    Here I've implemented several word frequency algorithms
    that are commonly used in information retrieval: Term Frequency,
    Document Frequency, and TF-IDF (Term-Frequency*Inverse-Document-Frequency)
    are included.
    Term Frequency is a statistical function that
    returns a number representing how frequently
    an expression occurs in a document. This
    indicates how significant a particular term is in
    a given document.
    Document Frequency is a statistical function that returns
    an integer representing the number of documents in a
    corpus that a term occurs in (where the max number returned
    would be the number of documents in the corpus).
    Inverse Document Frequency is mathematically written as
    log10(N/df), where N is the number of documents in your
    corpus and df is the Document Frequency. If df is 0, a
    ZeroDivisionError will be thrown.
    Term-Frequency*Inverse-Document-Frequency is a measure
    of the originality of a term. It is mathematically written
    as tf*log10(N/df). It compares the number of times
    a term appears in a document with the number of documents
    the term appears in. If df is 0, a ZeroDivisionError will be thrown.
"""


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


    ### Spark implementation
    # Step 1: Load the data
    # Create and RDD of textFiles
    # Using wholeTextFiles the created RDD has filename as Key and content as value
    # The method takes numpartitions = 8 to set the parallelism of the application ﻿ ﻿
    tech_text = sc.wholeTextFiles("/mnt/dataset/public/bbcnews/tech/",8) 


    #Step 2: Compute TF-IDF - one variable at a time
    number_of_docs = tech_text.count()


    # term frequency
    import re

    def tokenize(s):
      return re.split("\\W+", s.lower())

    #We Tokenize the text
    tokenized_text = tech_text.map(lambda (text,title): (title, tokenize(text)) )

    #Count Words in each document
    term_frequency = tokenized_text.flatMapValues(lambda x: x).countByValue()
    term_frequency.items()[:20] # Display 20 lines

    # document frequency

    document_frequency = tokenized_text.flatMapValues(lambda x: x).distinct()\
                        .map(lambda (title,word): (word,title)).countByKey()


    document_frequency.items()[:10]

    # compute tfidf
    import numpy as np


    def tf_idf(number_of_docs, term_frequency, document_frequency):
        result = []
        for key, value in tf.items():
            doc = key[0]
            term = key[1]
            df = document_frequency[term]
            if (df>0):
              tf_idf = float(value)*np.log(number_of_docs/df)
            
            result.append({"doc":doc, "term":term, "score":tf_idf})
        return result


    tf_idf_output = tf_idf(number_of_docs, term_frequency, document_frequency)

    tf_idf_output[:10]





