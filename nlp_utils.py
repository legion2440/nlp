from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import nltk

REPO_ROOT = Path(__file__).resolve().parent
DATASET_PATH = REPO_ROOT / "resources" / "tweets_train.txt"
NLTK_DATA_DIR = REPO_ROOT / "nltk_data"
LABEL_MAP = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}

if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))


@lru_cache(maxsize=1)
def ensure_nltk_data() -> None:
    required_resources = (
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    )

    for resource_path, package_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(
                package_name,
                download_dir=str(NLTK_DATA_DIR),
                quiet=True,
                raise_on_error=True,
            )

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download(
                "punkt_tab",
                download_dir=str(NLTK_DATA_DIR),
                quiet=True,
                raise_on_error=True,
            )
        except Exception:
            pass


def load_labeled_tweets(path: Path | str = DATASET_PATH) -> tuple[list[str], list[str]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    labels: list[str] = []
    tweets: list[str] = []

    with dataset_path.open(encoding="utf-8") as dataset_file:
        for raw_line in dataset_file:
            line = raw_line.strip()
            if not line:
                continue

            label, tweet = line.split(",", 1)
            normalized_label = label.strip().lower()
            if normalized_label not in LABEL_MAP:
                raise ValueError(f"Unsupported label: {normalized_label}")

            labels.append(normalized_label)
            tweets.append(tweet.strip())

    return labels, tweets
