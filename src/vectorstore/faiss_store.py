from __future__ import annotations

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
    IndexIDMap2 используется для поддержки add_with_ids и удаления по ID.
    """

    def __init__(self, vectorizer: SentenceVectorizer):
        self._vectorizer = vectorizer
        self.index: faiss.Index = None

    def _hash_id(self, chunk_id: str) -> int:
        """Стабильный int64 ID для FAISS"""
        digest = hashlib.md5(chunk_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=True)

    def _ensure_index(self):
        """Создаёт новый индекс, если его ещё нет"""
        if self.index is None:
            dim = self._vectorizer.dimension
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add_chunks(self, chunks: List[Chunk]):
        """Добавление новых чанков в индекс"""
        if not chunks:
            return

        self._ensure_index()

        texts = [ch.context or "" for ch in chunks]
        embeddings = self._vectorizer.embed_many(texts)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        ids = np.array([self._hash_id(ch.id) for ch in chunks], dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)

    def delete_chunks_by_pdf(self, pdf_name: str, chunks: List[Chunk]):
        """Удаление всех chunk из указанного PDF"""
        if self.index is None:
            return

        ids_to_delete = [self._hash_id(ch.id) for ch in chunks if ch.source == pdf_name]
        if ids_to_delete:
            self.index.remove_ids(np.array(ids_to_delete, dtype=np.int64))

    def save(self, index_path: Path):
        """Сохраняет индекс на диск"""
        if self.index is not None:
            faiss.write_index(self.index, str(index_path))

    def load(self, index_path: Path):
        """Загружает индекс с диска"""
        self.index = faiss.read_index(str(index_path))
        # Если индекс уже создан как IDMap2 — оборачивать не нужно.
        # Новый IndexIDMap2 создаётся только через _ensure_index при добавлении новых данных.
