"""
CLI для rag-bseu: единый вход для всех pipeline-кнопок.

Доступные команды:
- parse_documents.py        → загрузка / парсинг источников
- prepare_vector_store.py   → построение базы знаний и FAISS
- query.py                  → интерактивный RAG-запрос
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-BSEU CLI")

    parser.add_argument(
        "--parse-documents",
        action="store_true",
        help="Скачать и распарсить документы (PDF / сайты)",
    )
    parser.add_argument(
        "--prepare-vector-store",
        action="store_true",
        help="Построить базу знаний и FAISS индекс",
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="Интерактивный режим запросов",
    )

    args = parser.parse_args()
    usage_dir = Path(__file__).resolve().parent

    if args.parse_documents:
        subprocess.run([sys.executable, usage_dir / "parse_documents.py"])

    elif args.prepare_vector_store:
        subprocess.run([sys.executable, usage_dir / "prepare_vector_store.py"])

    elif args.query:
        subprocess.run([sys.executable, usage_dir / "query.py"])

    else:
        print("❌ Укажите режим запуска:")
        print("   --parse-documents")
        print("   --prepare-vector-store")
        print("   --query")


if __name__ == "__main__":
    main()
