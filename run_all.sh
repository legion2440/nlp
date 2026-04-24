#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -x "$ROOT_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON="$ROOT_DIR/.venv/Scripts/python.exe"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  PYTHON="$(command -v python)"
fi

scripts=(
  "ex00/check_env.py"
  "ex01/answer.py"
  "ex02/answer.py"
  "ex03/answer.py"
  "ex04/answer.py"
  "ex05/answer.py"
  "ex06/answer.py"
  "ex07/answer.py"
)

for script in "${scripts[@]}"; do
  echo "Running $script"
  if [[ "$PYTHON" == *.exe ]]; then
    SCRIPT_PATH="$(wslpath -w "$ROOT_DIR/$script")"
  else
    SCRIPT_PATH="$ROOT_DIR/$script"
  fi
  "$PYTHON" "$SCRIPT_PATH"
  echo
done
