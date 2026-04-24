#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$ROOT_DIR/resources/tweets_train.txt"
NLTK_DATA_DIR="$ROOT_DIR/nltk_data"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}

trap cleanup EXIT

git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/01-edu/public "$TMP_DIR/public"
git -C "$TMP_DIR/public" sparse-checkout set --no-cone /subjects/ai/nlp/resources/tweets_train.txt
cp "$TMP_DIR/public/subjects/ai/nlp/resources/tweets_train.txt" "$TARGET"

git clone --quiet --depth 1 --filter=blob:none --sparse https://github.com/nltk/nltk_data "$TMP_DIR/nltk_data_repo"
git -C "$TMP_DIR/nltk_data_repo" sparse-checkout set --no-cone \
  /packages/corpora/stopwords.zip \
  /packages/tokenizers/punkt.zip \
  /packages/tokenizers/punkt_tab.zip

mkdir -p "$NLTK_DATA_DIR/corpora" "$NLTK_DATA_DIR/tokenizers"
cp "$TMP_DIR/nltk_data_repo/packages/corpora/stopwords.zip" "$NLTK_DATA_DIR/corpora/"
cp "$TMP_DIR/nltk_data_repo/packages/tokenizers/punkt.zip" "$NLTK_DATA_DIR/tokenizers/"
cp "$TMP_DIR/nltk_data_repo/packages/tokenizers/punkt_tab.zip" "$NLTK_DATA_DIR/tokenizers/"

python3 - <<PY
from pathlib import Path
from zipfile import ZipFile

data_dir = Path(r"$NLTK_DATA_DIR")
archives = (
    data_dir / "corpora" / "stopwords.zip",
    data_dir / "tokenizers" / "punkt.zip",
    data_dir / "tokenizers" / "punkt_tab.zip",
)

for archive_path in archives:
    with ZipFile(archive_path) as archive:
        archive.extractall(archive_path.parent)
PY

rm -f \
  "$NLTK_DATA_DIR/corpora/stopwords.zip" \
  "$NLTK_DATA_DIR/tokenizers/punkt.zip" \
  "$NLTK_DATA_DIR/tokenizers/punkt_tab.zip"

echo "Saved $TARGET"
echo "Prepared local NLTK data in $NLTK_DATA_DIR"
