from __future__ import annotations

import pandas as pd

TEXTS = [
    "This is my first NLP exercise",
    "wtf!!!!!",
]


def main() -> None:
    series_data = pd.Series(TEXTS, name="text")
    print(series_data.str.lower())
    print()
    print(series_data.str.upper())


if __name__ == "__main__":
    main()
