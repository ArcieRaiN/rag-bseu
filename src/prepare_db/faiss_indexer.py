from __future__ import annotations

"""
Модуль для построения FAISS индекса из чанков.

Отвечает за:
- Генерацию embeddings для поля context
- Построение FAISS IndexFlatIP индекса
- Сохранение индекса на диск
"""

from pathlib import Path
from typing import List

import faiss

from src.main.models import Chunk
from src.main.vectorizer import SentenceVectorizer


class FAISSIndexer:
    """
    Класс для построения FAISS индекса из чанков.
    
    Использует SentenceVectorizer для генерации embeddings поля context.
    IndexFlatIP используется для cosine similarity (SentenceVectorizer нормализует вектора).
    """

    def __init__(self, vectorizer: SentenceVectorizer):
        """
        Инициализация индексера.
        
        Args:
            vectorizer: Векторизатор для генерации embeddings
        """
        self._vectorizer = vectorizer

    def build_index(
        self,
        chunks: List[Chunk],
        index_path: Path,
    ) -> None:
        """
        Строит FAISS IndexFlatIP по embeddings поля context.
        
        Args:
            chunks: Список чанков для индексирования
            index_path: Путь для сохранения индекса
        """
        if not chunks:
            # Создаём пустой индекс на случай пустой базы (отладка)
            index = faiss.IndexFlatIP(self._vectorizer.dimension)
            faiss.write_index(index, str(index_path))
            return

        # Извлекаем тексты context для векторизации
        texts = [ch.context for ch in chunks]
        
        # Генерируем embeddings
        embeddings = self._vectorizer.embed_many(texts).astype("float32")

        # Создаём и заполняем индекс
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        # Сохраняем индекс
        faiss.write_index(index, str(index_path))
