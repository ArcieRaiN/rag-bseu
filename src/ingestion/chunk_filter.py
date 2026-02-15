from __future__ import annotations

"""
Модуль для фильтрации и классификации чанков.

Отвечает за:
- Определение типа чанка (data, metadata, skip)
- Фильтрацию служебных чанков (предисловие, содержание, контакты)
- Обнаружение двуязычных дубликатов
"""

from typing import List
import re

from src.core.models import Chunk


class ChunkType:
    """Типы чанков."""
    DATA = "data"       # Данные для обогащения
    METADATA = "metadata"  # Служебная информация (предисловие, содержание)
    SKIP = "skip"       # Пропустить полностью


class ChunkFilter:
    """
    Фильтр для классификации и фильтрации чанков.

    Определяет, какие чанки нужно обогащать, а какие пропустить или обработать упрощённо.
    """

    # Ключевые слова для служебных чанков (русский и английский)
    SKIP_KEYWORDS = {
        "предисловие", "foreword", "preface",
        "содержание", "contents", "table of contents",
        "ответственные", "responsible", "authors",
        "контакты", "contacts", "contact information",
        "оглавление", "table of contents",
        "список", "list",
    }

    # Паттерны для определения служебных чанков
    SKIP_PATTERNS = [
        re.compile(r"^ПРЕДИСЛОВИЕ", re.IGNORECASE),
        re.compile(r"^СОДЕРЖАНИЕ", re.IGNORECASE),
        re.compile(r"^ОГЛАВЛЕНИЕ", re.IGNORECASE),
        re.compile(r"^CONTENTS", re.IGNORECASE),
        re.compile(r"^FOREWORD", re.IGNORECASE),
        re.compile(r"^PREFACE", re.IGNORECASE),
        re.compile(r"ответственные за", re.IGNORECASE),
        re.compile(r"responsible for", re.IGNORECASE),
        re.compile(r"телефон|phone|email|@", re.IGNORECASE),
    ]

    def __init__(self, skip_first_pages: int = 3):
        """
        Инициализация фильтра.

        Args:
            skip_first_pages: Количество первых страниц для упрощённой обработки
        """
        self.skip_first_pages = skip_first_pages

    def classify_chunk(self, chunk: Chunk) -> str:
        """
        Классифицирует чанк по типу.

        Args:
            chunk: Чанк для классификации

        Returns:
            Тип чанка: ChunkType.DATA, ChunkType.METADATA или ChunkType.SKIP
        """
        if chunk.page <= self.skip_first_pages:
            return ChunkType.METADATA

        text = (chunk.text or "").strip()
        if not text:
            return ChunkType.SKIP

        # Служебные чанки по паттернам
        for pattern in self.SKIP_PATTERNS:
            if pattern.search(text[:200]):
                return ChunkType.METADATA

        # Ключевые слова в начале текста
        first_words = text.lower().split()[:5]
        if any(word in self.SKIP_KEYWORDS for word in first_words):
            return ChunkType.METADATA

        # Двуязычные дубликаты
        if self._is_bilingual_duplicate(text):
            return ChunkType.METADATA

        return ChunkType.DATA

    def filter_chunks(self, chunks: List[Chunk]) -> tuple[List[Chunk], List[Chunk], List[Chunk]]:
        """
        Фильтрует чанки на три категории.

        Args:
            chunks: Список чанков для фильтрации

        Returns:
            Кортеж (data_chunks, metadata_chunks, skip_chunks)
        """
        data_chunks, metadata_chunks, skip_chunks = [], [], []

        for chunk in chunks:
            chunk_type = self.classify_chunk(chunk)
            if chunk_type == ChunkType.DATA:
                data_chunks.append(chunk)
            elif chunk_type == ChunkType.METADATA:
                metadata_chunks.append(chunk)
            else:
                skip_chunks.append(chunk)

        return data_chunks, metadata_chunks, skip_chunks

    @staticmethod
    def _is_bilingual_duplicate(text: str) -> bool:
        """
        Определяет, является ли текст двуязычным дубликатом (RU + EN заголовки).

        Args:
            text: Текст для проверки

        Returns:
            True если это двуязычный дубликат
        """
        lines = text[:200].split("\n")[:3]
        if len(lines) < 2:
            return False

        first_line, second_line = lines[0].strip(), lines[1].strip()

        # Кириллица в первой строке
        has_cyrillic = any("\u0400" <= c <= "\u04FF" for c in first_line)
        # Латиница в заглавных буквах второй строки
        has_latin_upper = second_line.isupper() and any("A" <= c <= "Z" for c in second_line) and len(second_line.split()) <= 5

        return has_cyrillic and has_latin_upper

    @staticmethod
    def normalize_chunk_id(chunk_id: str) -> str:
        """
        Нормализует chunk_id: удаляет NBSP, нормализует пробелы, убирает префиксы.

        Args:
            chunk_id: Исходный chunk_id

        Returns:
            Нормализованный chunk_id
        """
        if not chunk_id:
            return ""

        normalized = chunk_id.strip()
        prefixes = ["ID:", "id:", "ID: ", "id: ", "chunk_id:", "chunk_id: "]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()

        for nbspace in ["\u00A0", "\u2009", "\u2000", "\u2001"]:
            normalized = normalized.replace(nbspace, " ")

        # Множественные пробелы → один
        normalized = " ".join(normalized.split())
        return normalized
