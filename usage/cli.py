"""
CLI для rag-bseu: единый вход для query и подготовки векторного хранилища.

Использует подпакеты:
- prepare_vector_store.py
- query.py
"""

import argparse
from pathlib import Path
import subprocess

def main():
    parser = argparse.ArgumentParser(description="RAG-BSEU CLI")
    parser.add_argument(
        "--prepare-vector-store",
        action="store_true",
        help="Запустить подготовку базы знаний и FAISS индекса",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Интерактивный режим запросов",
    )

    args = parser.parse_args()
    root_dir = Path(__file__).resolve().parent

    if args.prepare_vector_store:
        subprocess.run(["python", root_dir / "prepare_vector_store.py"])
    elif args.query:
        subprocess.run(["python", root_dir / "query.py"])
    else:
        print("❌ Нужно указать флаг: --prepare-vector-store или --query")

if __name__ == "__main__":
    main()
