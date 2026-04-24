from __future__ import annotations

import string

TEXT = 'Remove, this from .? the sentence !!!! !"#&\'()*+,-./:;<=>_'


def main() -> None:
    translation_table = str.maketrans("", "", string.punctuation)
    print(TEXT.translate(translation_table))


if __name__ == "__main__":
    main()
