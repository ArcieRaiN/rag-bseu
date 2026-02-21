"""
USAGE: кнопка "Parse Documents".

Запускает pipeline парсинга документов
и сохраняет PDF в usage/documents.
"""

from pathlib import Path

from src.pipelines.parse_documents_pipeline import ParseDocumentsPipeline


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent  # rag-bseu

    documents_dir = root_dir / "usage" / "archive_documents"

    pipeline = ParseDocumentsPipeline(
        output_dir=documents_dir,
        max_pages=1,
    )

    pipeline.run()


if __name__ == "__main__":
    main()
