"""Запуск tests.evaluator из любого cwd: переключается в корень rag-bseu."""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    os.chdir(ROOT)
    raise SystemExit(subprocess.call([sys.executable, "-m", "tests.evaluator", *sys.argv[1:]]))


if __name__ == "__main__":
    main()
