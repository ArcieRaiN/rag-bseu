from __future__ import annotations

"""
PIPELINE: парсинг и загрузка документов (PDF).

Назначение:
- Скачать статистические сборники (compilation) с Белстата
- Сохранить PDF в usage/documents
"""

from pathlib import Path

from src.ingestion.site_parser import SiteParser


class ParseDocumentsPipeline:
    """Pipeline для подготовки документов.

    SRC-уровень:
    - не знает про CLI
    - не парсит аргументы
    - только orchestrates ingestion
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        max_pages: int = 2,
    ) -> None:
        """
        Args:
            output_dir: куда сохранять PDF (usage/documents).
            max_pages:  сколько страниц каталога обработать (по умолчанию 2).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.site_parser = SiteParser(
            output_dir=self.output_dir,
            max_pages=max_pages,
        )

    def run(self) -> list[Path]:
        """Запуск пайплайна. Возвращает список скачанных PDF."""
        print("Запуск ParseDocumentsPipeline…")
        print(f"Директория документов: {self.output_dir}")

        downloaded = self.site_parser.parse()

        print("Парсинг документов завершён.")
        return downloaded
