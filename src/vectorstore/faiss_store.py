"""
Векторное хранилище на базе FAISS.

Использует простой IndexFlatIP с позиционной адресацией (position == index в data.json).
Cosine similarity достигается через нормализованные векторы + inner product.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
import numpy as np

from src.core.models import Chunk
from src.vectorstore.vectorizer import SentenceVectorizer


class FAISSStore:
    """
    FAISS-based vector store with positional indexing.

    Vectors are L2-normalized before insertion so that inner product == cosine similarity.
    Index position i corresponds to chunks[i] in data.json.
    """

    def __init__(self, vectorizer: SentenceVectorizer):
        self._vectorizer = vectorizer
        self.index: faiss.Index | None = None

    def _ensure_index(self):
        if self.index is None:
            dim = self._vectorizer.dimension
            self.index = faiss.IndexFlatIP(dim)

    @staticmethod
    def _build_embed_text(chunk: Chunk) -> str:
        context = (chunk.context or "").strip()
        text_prefix = (chunk.text or "")[:500].strip()
        if context and text_prefix:
            return f"{context}\n{text_prefix}"
        return context or text_prefix or ""

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return

        self._ensure_index()

        texts = [self._build_embed_text(ch) for ch in chunks]
        embeddings = self._vectorizer.embed_many(texts, is_query=False)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

    def save(self, index_path: Path):
        if self.index is not None:
            faiss.write_index(self.index, str(index_path))

    def load(self, index_path: Path):
        self.index = faiss.read_index(str(index_path))
