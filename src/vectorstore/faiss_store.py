from __future__ import annotations

"""
FAISS Store: построение и сохранение FAISS индекса.

Responsibilities:
- Генерация embeddings через SentenceVectorizer
- Построение IndexFlatIP для cosine similarity
- Сохранение индекса на диск
"""

import hashlib
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

    def load_index(self, index_path: Path) -> faiss.Index:
        """
        Загружает FAISS индекс с диска.
        """
        return faiss.read_index(str(index_path))

    def _hash_id(self, chunk_id: str) -> int:
        """Стабильный int64 ID для FAISS"""
        digest = hashlib.md5(chunk_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=True)

    def add_chunks(self, chunks: List[Chunk]):
        """Добавление новых чанков в индекс"""
        if not chunks:
            return
        texts = [ch.context or "" for ch in chunks]
        embeddings = self._vectorizer.embed_many(texts).astype(np.float32)
        ids = np.array([self._hash_id(ch.id) for ch in chunks], dtype=np.int64)

        if not hasattr(self, "index") or self.index is None:
            dim = self._vectorizer.dimension
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.index.add_with_ids(embeddings, ids)

    def delete_chunks_by_pdf(self, pdf_name: str, chunks: List[Chunk]):
        """Удаление всех chunk из указанного PDF"""
        if not hasattr(self, "index") or self.index is None:
            return
        ids_to_delete = [self._hash_id(ch.id) for ch in chunks if ch.source == pdf_name]
        if ids_to_delete:
            self.index.remove_ids(np.array(ids_to_delete, dtype=np.int64))

    def save(self, index_path: Path):
        if hasattr(self, "index") and self.index is not None:
            faiss.write_index(self.index, str(index_path))

    def load(self, index_path: Path):
        self.index = faiss.read_index(str(index_path))

