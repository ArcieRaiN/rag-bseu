from __future__ import annotations

"""
Модуль для чанкинга PDF-документов с использованием LlamaIndex.

Стратегия:
- ЧАНК = СТРАНИЦА PDF

Отвечает за:
- Чтение PDF через LlamaIndex PDFReader
- Группировку текста по страницам
- Извлечение и нормализацию номера страницы
"""

from pathlib import Path
from typing import List, Dict
import shutil
from collections import defaultdict

from llama_index.readers.file import PDFReader
from llama_index.core.schema import BaseNode

from src.main.models import Chunk


class PDFChunker:
    """
    Класс для разбиения PDF-документов на чанки.

    Стратегия:
    - один Chunk соответствует одной странице PDF
    - идеально подходит для статистических таблиц и отчётов
    """

    def chunk_pdf(self, pdf_path: Path) -> List[Chunk]:
        """
        Разбивает PDF на чанки по страницам.

        Args:
            pdf_path: Путь к PDF-файлу

        Returns:
            Список Chunk, где:
            - text = полный текст страницы
            - page = номер страницы
            - source = имя PDF
            - остальные поля заполняются позже
        """
        # LlamaIndex PDFReader работает с директориями,
        # поэтому создаём временную папку с одним PDF
        temp_dir = pdf_path.parent / f"_temp_{pdf_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        temp_pdf = temp_dir / pdf_path.name

        try:
            shutil.copy2(pdf_path, temp_pdf)

            pdf_reader = PDFReader()
            documents = pdf_reader.load_data(file=str(temp_pdf))

            # page_number -> list[text fragments]
            pages: Dict[int, List[str]] = defaultdict(list)

            for doc in documents:
                # PDFReader уже разбивает документ на page-level ноды
                nodes: List[BaseNode] = getattr(doc, "nodes", None)

                # fallback: если nodes нет, работаем через text + metadata
                if not nodes:
                    page = self._extract_page_number(doc)
                    pages[page].append(doc.text or "")
                    continue

                for node in nodes:
                    page = self._extract_page_number(node)
                    if node.text:
                        pages[page].append(node.text)

            # Собираем чанки: один чанк = одна страница
            chunks: List[Chunk] = []

            for page_num in sorted(pages.keys()):
                page_text = "\n".join(pages[page_num]).strip()

                if not page_text:
                    continue

                chunk = Chunk(
                    id="",              # будет назначен позже
                    context="",         # заполнится в LLMEnricher
                    text=page_text,
                    source=pdf_path.name,
                    page=page_num,
                    geo=None,
                    metrics=None,
                    years=None,
                    time_granularity=None,
                    oked=None,
                )
                chunks.append(chunk)

            return chunks

        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _extract_page_number(node: BaseNode) -> int:
        """
        Извлекает номер страницы из метаданных ноды или документа.

        Args:
            node: BaseNode или Document от LlamaIndex

        Returns:
            Номер страницы (0 если не найден)
        """
        metadata = getattr(node, "metadata", None)
        if not isinstance(metadata, dict):
            return 0

        page_num = (
            metadata.get("page_label")
            or metadata.get("page_number")
            or metadata.get("page")
            or 0
        )

        try:
            return int(page_num) if page_num else 0
        except (ValueError, TypeError):
            return 0
