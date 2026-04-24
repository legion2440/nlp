from __future__ import annotations

import string
import sys
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nlp_utils import ensure_nltk_data

TEXT = (
    "01 Edu System presents an innovative curriculum in software engineering "
    "and programming. With a renowned industry-leading reputation, the "
    "curriculum has been rigorously designed for learning skills of the digital "
    "world and technology industry. Taking a different approach than the "
    "classic teaching methods today, learning is facilitated through a "
    "collective and co-creative process in a professional environment."
)

TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)
STEMMER = PorterStemmer()


def preprocess_text(text: str) -> list[str]:
    ensure_nltk_data()
    stop_words = set(stopwords.words("english"))

    lowered_text = text.lower()
    text_without_punctuation = lowered_text.translate(TRANSLATION_TABLE)
    tokens = word_tokenize(text_without_punctuation)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return [STEMMER.stem(token) for token in filtered_tokens]


def main() -> None:
    print(preprocess_text(TEXT))


if __name__ == "__main__":
    main()
