"""
## Overview
This Python module provides tools for natural language processing (NLP) using the NLTK library.
It includes functionality for initializing stop words, creating a thread-safe lemmatizer,
and ensuring necessary resources are downloaded and loaded.
"""

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.data import find
from nltk.stem import WordNetLemmatizer

# Ensure necessary resources are downloaded and loaded
nltk.download("wordnet", quiet=True)
_ = wordnet._morphy  # Trigger the main-thread load for WordNet


# building a lemmatizer that points at your reader
class ThreadSafeLemmatizer(WordNetLemmatizer):
    """
    A thread-safe lemmatizer that uses a custom WordNet reader.

    Attributes:
        _wordnet (WordNetCorpusReader): The WordNet reader used for lemmatization.
    """

    def __init__(self, reader):
        """
        Initializes the ThreadSafeLemmatizer with a custom WordNet reader.

        Args:
            reader (WordNetCorpusReader): The WordNet reader to be used.
        """
        super().__init__()
        self._wordnet = reader


def initialize_nlp_tools():
    """
    Initializes necessary NLP tools by downloading required resources if not already available.

    This function ensures that the stopwords corpus is available and loads it into memory.

    Returns:
        set: A set of English stopwords.
    """
    # Downloads necessary NLP resources if not available
    #   - already included in docker file

    try:
        find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    stop_words = set(stopwords.words("english"))
    return stop_words


# Initialize stop words globally
stop_words = initialize_nlp_tools()


def get_stop_words():
    """
    Retrieves the set of English stopwords.

    Returns:
        set: A set of English stopwords.
    """
    return stop_words


def get_lemmatizer():
    """
    Creates and retrieves a thread-safe lemmatizer instance.

    Returns:
        ThreadSafeLemmatizer: A thread-safe lemmatizer using the WordNet reader.
    """
    lemm = ThreadSafeLemmatizer(wordnet)
    return lemm
