from __future__ import annotations

"""
Модуль для фильтрации и классификации чанков.

Отвечает за:
- Определение типа чанка (data, metadata, skip)
- Фильтрацию служебных чанков (предисловие, содержание, контакты)
- Обнаружение двуязычных дубликатов
"""

from typing import List, Optional
import re

from src.main.models import Chunk


class ChunkType:
    """Типы чанков."""
    DATA = "data"  # Данные для обогащения
    METADATA = "metadata"  # Служебная информация (предисловие, содержание)
    SKIP = "skip"  # Пропустить полностью


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

    # Паттерны для двуязычных дубликатов (RU + EN заголовки)
    BILINGUAL_PATTERN = re.compile(
        r"^([А-ЯЁ\s]+)\s*\n\s*([A-Z\s]+)$",
        re.MULTILINE | re.DOTALL
    )

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
        # Первые страницы обычно содержат обложку и содержание
        if chunk.page <= self.skip_first_pages:
            return ChunkType.METADATA

        text = (chunk.text or "").strip()
        if not text:
            return ChunkType.SKIP

        # Проверка на служебные чанки по паттернам
        for pattern in self.SKIP_PATTERNS:
            if pattern.search(text[:200]):  # Проверяем первые 200 символов
                return ChunkType.METADATA

        # Проверка на ключевые слова в начале текста
        text_lower = text.lower()
        first_words = text_lower.split()[:5]  # Первые 5 слов
        for word in first_words:
            if word in self.SKIP_KEYWORDS:
                return ChunkType.METADATA

        # Проверка на двуязычные дубликаты (RU заголовок + EN заголовок)
        if self._is_bilingual_duplicate(text):
            return ChunkType.METADATA

        return ChunkType.DATA

    def filter_chunks(self, chunks: List[Chunk]) -> tuple[List[Chunk], List[Chunk], List[Chunk]]:
        """
        Фильтрует чанки на три категории.
        
        Args:
            chunks: Список чанков для фильтрации
            
        Returns:
            Кортеж (data_chunks, metadata_chunks, skip_chunks):
            - data_chunks: чанки для полного обогащения
            - metadata_chunks: служебные чанки (упрощённое обогащение)
            - skip_chunks: чанки для пропуска
        """
        data_chunks: List[Chunk] = []
        metadata_chunks: List[Chunk] = []
        skip_chunks: List[Chunk] = []

        for chunk in chunks:
            chunk_type = self.classify_chunk(chunk)
            
            if chunk_type == ChunkType.DATA:
                data_chunks.append(chunk)
            elif chunk_type == ChunkType.METADATA:
                metadata_chunks.append(chunk)
            else:  # ChunkType.SKIP
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
        # Проверяем первые 200 символов
        preview = text[:200].strip()
        
        # Паттерн: русский заголовок, затем английский заголовок
        lines = preview.split('\n')[:3]  # Первые 3 строки
        
        if len(lines) < 2:
            return False
        
        # Проверяем, есть ли русский текст в первой строке и английский во второй
        first_line = lines[0].strip()
        second_line = lines[1].strip()
        
        # Проверка на кириллицу в первой строке
        has_cyrillic_first = any('\u0400' <= char <= '\u04FF' for char in first_line)
        
        # Проверка на латиницу во второй строке (заглавные буквы)
        has_latin_second = (
            second_line.isupper() and
            any('A' <= char <= 'Z' for char in second_line) and
            len(second_line.split()) <= 5  # Короткий заголовок
        )
        
        return has_cyrillic_first and has_latin_second

    @staticmethod
    def normalize_chunk_id(chunk_id: str) -> str:
        """
        Нормализует chunk_id: удаляет NBSP, нормализует пробелы, удаляет префиксы.
        
        Args:
            chunk_id: Исходный chunk_id
            
        Returns:
            Нормализованный chunk_id
        """
        if not chunk_id:
            return chunk_id
        
        # Удаляем префиксы типа "ID:", "chunk_id:", "ID: " и т.д.
        normalized = chunk_id.strip()
        prefixes = ["ID:", "id:", "ID: ", "id: ", "chunk_id:", "chunk_id: "]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        # Заменяем неразрывные пробелы (NBSP) на обычные
        normalized = normalized.replace('\u00A0', ' ')  # NBSP
        normalized = normalized.replace('\u2009', ' ')  # Thin space
        normalized = normalized.replace('\u2000', ' ')  # En quad
        normalized = normalized.replace('\u2001', ' ')  # Em quad
        
        # Нормализуем множественные пробелы
        normalized = ' '.join(normalized.split())
        
        return normalized
