from __future__ import annotations

"""
PIPELINE: парсинг и загрузка документов (PDF).

Назначение:
- Скачать / собрать PDF-документы
- Сохранить их в usage/documents
"""

from pathlib import Path
from typing import Optional

from src.ingestion.site_parser import SiteParser


class ParseDocumentsPipeline:
    """
    Pipeline для подготовки документов.

    SRC-уровень:
    - не знает про CLI
    - не парсит аргументы
    - только orchestrates ingestion
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        source_url: Optional[str] = None,
    ):
        """
        Args:
            output_dir: куда сохранять PDF (usage/documents)
            source_url: опционально — источник (сайт / каталог)
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.site_parser = SiteParser(
            output_dir=self.output_dir,
            source_url=source_url,
        )

    def run(self) -> None:
        """
        Запуск пайплайна.
        """
        print("📥 Запуск ParseDocumentsPipeline...")
        print(f"📂 Директория документов: {self.output_dir}")

        self.site_parser.parse()

        print("✅ Парсинг документов завершён")
