from __future__ import annotations

"""
FAISS Store: построение и сохранение FAISS индекса.

Responsibilities:
- Генерация embeddings через SentenceVectorizer
- Построение IndexFlatIP для cosine similarity
- Сохранение индекса на диск
"""

from pathlib import Path
from typing import List

import faiss
import numpy as np

from src.core.models import Chunk
from src.vectorstore.vectorizer import SentenceVectorizer


class FAISSStore:
    """
    FAISS-based vector store.

    Использует SentenceVectorizer для генерации embeddings.
    Cosine similarity достигается через IndexFlatIP + нормализованные векторы.
    """

    def __init__(self, vectorizer: SentenceVectorizer):
        self._vectorizer = vectorizer

    def build_and_save(
        self,
        chunks: List[Chunk],
        index_path: Path,
    ) -> None:
        """
        Строит FAISS индекс из списка чанков и сохраняет на диск.

        Args:
            chunks: список объектов Chunk
            index_path: путь для сохранения FAISS индекса
        """
        dim = self._vectorizer.dimension

        if not chunks:
            # пустой индекс для отладки
            index = faiss.IndexFlatIP(dim)
            faiss.write_index(index, str(index_path))
            return

        # берем контексты чанков для векторизации
        texts = [ch.context or "" for ch in chunks]
        embeddings = self._vectorizer.embed_many(texts).astype(np.float32)

        if embeddings.shape[1] != dim:
            raise RuntimeError(
                f"Embeddings dimension {embeddings.shape[1]} != expected {dim}"
            )

        # создаём FAISS индекс
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # сохраняем
        faiss.write_index(index, str(index_path))

    def load_index(self, index_path: Path) -> faiss.Index:
        """
        Загружает FAISS индекс с диска.
        """
        return faiss.read_index(str(index_path))

