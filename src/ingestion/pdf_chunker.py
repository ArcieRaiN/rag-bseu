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
import tempfile
from collections import defaultdict
import logging

from llama_index.readers.file import PDFReader
from llama_index.core.schema import BaseNode

from src.core.models import Chunk

logger = logging.getLogger(__name__)


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
        chunks: List[Chunk] = []

        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_pdf = Path(tmpdirname) / pdf_path.name
            shutil.copy2(pdf_path, temp_pdf)

            pdf_reader = PDFReader()
            documents = pdf_reader.load_data(file=str(temp_pdf))

            # page_number -> list[text fragments]
            page_texts: Dict[int, List[str]] = defaultdict(list)

            for doc in documents:
                # PDFReader разбивает документ на page-level ноды
                nodes: List[BaseNode] = getattr(doc, "nodes", [])

                if nodes:
                    for node in nodes:
                        page_number = self._extract_page_number(node)
                        if node.text:
                            page_texts[page_number].append(node.text)
                else:
                    page_number = self._extract_page_number(doc)
                    page_texts[page_number].append(doc.text or "")

            # Собираем чанки: один чанк = одна страница
            for page_number in sorted(page_texts.keys()):
                page_text = "\n".join(page_texts[page_number]).strip()

                if not page_text:
                    continue

                chunk = Chunk(
                    id="",              # будет назначен позже
                    context="",         # заполнится в LLMEnricher
                    text=page_text,
                    source=pdf_path.name,
                    page=page_number,
                    geo=None,
                    metrics=None,
                    years=None,
                    time_granularity=None,
                    oked=None,
                )
                chunks.append(chunk)

            logger.info(
                "PDF %s: %d страниц разбито на %d чанков",
                pdf_path.name, len(documents), len(chunks)
            )

        return chunks

    @staticmethod
    def _extract_page_number(node: BaseNode) -> int:
        """
        Извлекает номер страницы из метаданных ноды или документа.
        Args:
            node: BaseNode или Document от LlamaIndex
        Returns:
            Номер страницы (0 если не найден)
        """
        metadata = getattr(node, "metadata", {}) or {}
        for key in ("page_label", "page_number", "page"):
            val = metadata.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return 0
        return 0
