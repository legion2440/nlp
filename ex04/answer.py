from __future__ import annotations

import sys
from pathlib import Path

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nlp_utils import ensure_nltk_data

TEXT = """
The goal of this exercise is to learn to remove stop words with NLTK.
Stop words usually refers to the most common words in a language.
"""


def main() -> None:
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(TEXT)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    print(filtered_tokens)


if __name__ == "__main__":
    main()
