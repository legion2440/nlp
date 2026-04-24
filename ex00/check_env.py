from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from nlp_utils import ensure_nltk_data

MODULES = ("jupyter", "nltk", "pandas", "sklearn")


def main() -> None:
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version < (3, 9):
        raise RuntimeError("Python 3.9 or newer is required.")

    for module_name in MODULES:
        importlib.import_module(module_name)
        print(f"{module_name}: ok")

    ensure_nltk_data()
    print("nltk_data: ok")


if __name__ == "__main__":
    main()
