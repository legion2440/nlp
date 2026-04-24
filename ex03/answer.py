from __future__ import annotations

import sys
from pathlib import Path

from nltk.tokenize import sent_tokenize, word_tokenize

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nlp_utils import ensure_nltk_data

TEXT = (
    "Bitcoin is a cryptocurrency invented in 2008 by an unknown person or group "
    "of people using the name Satoshi Nakamoto. The currency began use in 2009 "
    "when its implementation was released as open-source software."
)


def main() -> None:
    ensure_nltk_data()
    print(sent_tokenize(TEXT))
    print()
    print(word_tokenize(TEXT))


if __name__ == "__main__":
    main()
