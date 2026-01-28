from __future__ import annotations

"""
Модуль для чанкинга PDF-документов с использованием LlamaIndex.

Отвечает за:
- Чтение PDF через LlamaIndex PDFReader
- Разбиение на чанки с настраиваемыми параметрами
- Извлечение метаданных страниц
"""

from pathlib import Path
from typing import List
import shutil

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

from src.main.models import Chunk


class PDFChunker:
    """
    Класс для разбиения PDF-документов на чанки.
    
    Использует LlamaIndex для чтения PDF и парсинга на ноды.
    """

    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        paragraph_separator: str = "\n\n",
    ):
        """
        Инициализация чанкера.
        
        Args:
            chunk_size: Максимальный размер чанка в символах
            chunk_overlap: Перекрытие между чанками
            paragraph_separator: Разделитель параграфов
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator
        
        self._node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator,
        )

    def chunk_pdf(self, pdf_path: Path) -> List[Chunk]:
        """
        Разбивает PDF на чанки.
        
        Args:
            pdf_path: Путь к PDF-файлу
            
        Returns:
            Список Chunk с заполненными:
            - text (текст чанка)
            - source (имя PDF)
            - page (номер страницы)
            - context и метаданные пустые (заполнятся в LLMEnricher)
        """
        # Создаём временную директорию с одним PDF для LlamaIndex
        # (PDFReader работает с директориями)
        temp_dir = pdf_path.parent / f"_temp_{pdf_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        temp_pdf = temp_dir / pdf_path.name

        try:
            # Копируем PDF во временную директорию
            shutil.copy2(pdf_path, temp_pdf)

            # Читаем PDF через LlamaIndex
            pdf_reader = PDFReader()
            documents = pdf_reader.load_data(file=str(temp_pdf))

            # Парсим документы на ноды (чанки)
            chunks: List[Chunk] = []
            for doc in documents:
                nodes = self._node_parser.get_nodes_from_documents([doc])

                for node in nodes:
                    # Извлекаем номер страницы из метаданных
                    page_num = self._extract_page_number(node)

                    # Создаём Chunk с пустыми полями для enrichment
                    chunk = Chunk(
                        id="",  # будет присвоен в KnowledgeBaseBuilder
                        context="",  # заполнится в LLMEnricher
                        text=node.text or "",
                        source=pdf_path.name,
                        page=page_num,
                        geo=None,
                        metrics=None,
                        years=None,
                        time_granularity=None,
                        oked=None,
                    )
                    chunks.append(chunk)

        finally:
            # Удаляем временную директорию
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        return chunks

    @staticmethod
    def _extract_page_number(node) -> int:
        """
        Извлекает номер страницы из метаданных ноды.
        
        Args:
            node: Нода от LlamaIndex
            
        Returns:
            Номер страницы (0 если не найден)
        """
        if not hasattr(node, "metadata"):
            return 0
            
        metadata = node.metadata
        page_num = (
            metadata.get("page_label") or
            metadata.get("page_number") or
            metadata.get("page") or
            0
        )
        
        try:
            return int(page_num) if page_num else 0
        except (ValueError, TypeError):
            return 0
