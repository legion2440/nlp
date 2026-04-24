from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ex06.answer import preprocess_text
from nlp_utils import DATASET_PATH, LABEL_MAP, load_labeled_tweets


def main() -> None:
    labels, tweets = load_labeled_tweets(DATASET_PATH)

    vectorizer = CountVectorizer(
        max_features=500,
        lowercase=False,
        tokenizer=preprocess_text,
        token_pattern=None,
    )
    word_count_matrix = vectorizer.fit_transform(tweets)
    feature_names = vectorizer.get_feature_names_out()
    count_vectorized_df = pd.DataFrame.sparse.from_spmatrix(
        word_count_matrix,
        columns=feature_names,
    )

    print(f"Shape: {word_count_matrix.shape}")
    print()
    print("Fourth tweet token counts:")
    fourth_tweet_counts = count_vectorized_df.iloc[3]
    print(fourth_tweet_counts[fourth_tweet_counts > 0])
    print()
    print("Top 15 tokens:")
    print(count_vectorized_df.sum(axis=0).sort_values(ascending=False).head(15))
    print()
    count_vectorized_df["label"] = [LABEL_MAP[label] for label in labels]
    print("Label preview:")
    print(count_vectorized_df[["label"]].head())


if __name__ == "__main__":
    main()
