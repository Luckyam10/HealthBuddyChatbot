"""
preprocess.py
--------------
Text preprocessing module for the Health Information Chatbot.

Implements:
  1. Lowercasing
  2. Tokenization
  3. Stopword removal
  4. Light suffix-stripping stemming

This module is used by scikit-learn's TfidfVectorizer.
"""

import re


STOPWORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "to", "of", "in", "on", "at", "for", "with", "about", "as", "by", "and", "or",
    "but", "if", "than", "so", "do", "does", "did", "will", "would", "could",
    "should", "please", "just", "very", "really", "let", "know", "there", "here",
}


_SUFFIXES = ["ing", "edly", "edly", "ed", "es", "s", "ly"]


def simple_stem(word: str) -> str:

    if len(word) <= 4:
        return word

    for suf in _SUFFIXES:

        if (
            word.endswith(suf)
            and len(word) - len(suf) >= 3
        ):
            return word[: -len(suf)]

    return word


def tokenize(text: str):

    text = text.lower()

    tokens = re.findall(
        r"[a-z']+",
        text,
    )

    return tokens


def preprocess(text: str):

    tokens = tokenize(text)

    tokens = [
        t
        for t in tokens
        if t not in STOPWORDS
        and len(t) > 1
    ]

    tokens = [
        simple_stem(t)
        for t in tokens
    ]

    return tokens


if __name__ == "__main__":

    samples = [
        "What are the symptoms of the flu?",
        "I can't breathe, help!",
        "How many hours of sleep do I need?",
    ]

    for s in samples:

        print(
            f"{s!r:45} -> {preprocess(s)}"
        )
