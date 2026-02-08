"""
USAGE: кнопка "Parse Documents".

Запускает pipeline парсинга документов
и сохраняет PDF в usage/documents.
"""

from pathlib import Path

from src.pipelines.parse_documents import ParseDocumentsPipeline


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent  # rag-bseu

    documents_dir = root_dir / "usage" / "documents"

    pipeline = ParseDocumentsPipeline(
        output_dir=documents_dir,
        # source_url можно добавить позже, если нужно
        # source_url="https://example.com/statistics"
    )

    pipeline.run()


if __name__ == "__main__":
    main()
