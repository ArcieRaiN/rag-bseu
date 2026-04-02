"""Запуск usage/rebuild_index.py из корня rag-bseu."""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    os.chdir(ROOT)
    script = os.path.join(ROOT, "usage", "rebuild_index.py")
    raise SystemExit(subprocess.call([sys.executable, script]))


if __name__ == "__main__":
    main()
