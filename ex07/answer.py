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


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    try:
        return dataframe.to_markdown()
    except ImportError as error:
        if "tabulate" not in str(error):
            raise
        return dataframe_to_basic_markdown(dataframe)


def dataframe_to_basic_markdown(dataframe: pd.DataFrame) -> str:
    rows = [["", *map(str, dataframe.columns)]]
    for index, row in dataframe.iterrows():
        rows.append([str(index), *[str(value) for value in row]])

    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    separator = ["-" * width for width in widths]
    table_rows = [rows[0], separator, *rows[1:]]
    return "\n".join(
        "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
        for row in table_rows
    )


def print_sparse_matrix_preview(matrix, limit: int = 10) -> None:
    coo_matrix = matrix.tocoo()
    format_name = {
        "csr": "Compressed Sparse Row",
        "csc": "Compressed Sparse Column",
        "coo": "COOrdinate",
    }.get(getattr(matrix, "format", ""), matrix.__class__.__name__)
    print(
        f"<{format_name} sparse matrix of dtype '{matrix.dtype}'"
        f"\n\twith {matrix.nnz} stored elements and shape {matrix.shape}>"
    )
    print("  Coords\tValues")

    items = list(zip(coo_matrix.row, coo_matrix.col, coo_matrix.data))
    preview_items = items[:limit]
    if len(items) > limit * 2:
        preview_items += [(None, None, None)]
    preview_items += items[-limit:] if len(items) > limit else []

    for row, column, value in preview_items:
        if row is None:
            print("  :\t:")
        else:
            print(f"  ({row}, {column})\t{value}")


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

    print_sparse_matrix_preview(word_count_matrix)
    print()
    print(f"Shape: {word_count_matrix.shape}")
    print()
    print("DataFrame slice before label:")
    print(dataframe_to_markdown(count_vectorized_df.iloc[:3, 400:403]))
    print()
    print("Fourth tweet token counts:")
    fourth_tweet_counts = count_vectorized_df.iloc[3]
    print(fourth_tweet_counts[fourth_tweet_counts > 0])
    print()
    print("Top 15 tokens:")
    print(count_vectorized_df.sum(axis=0).sort_values(ascending=False).head(15))
    print()
    count_vectorized_df["label"] = [LABEL_MAP[label] for label in labels]
    print("DataFrame slice after label:")
    print(dataframe_to_markdown(count_vectorized_df.iloc[350:354][["your", "label"]]))


if __name__ == "__main__":
    main()
