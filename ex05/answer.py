from __future__ import annotations

import sys
from pathlib import Path

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nlp_utils import ensure_nltk_data

TEXT = """
The interviewer interviews the president in an interview
"""


def main() -> None:
    ensure_nltk_data()
    stemmer = PorterStemmer()
    tokens = word_tokenize(TEXT)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    print(stemmed_tokens)


if __name__ == "__main__":
    main()
