"""
Генерация векторных представлений текста (эмбеддингов).

Использует sentence-transformers для кодирования текста в вектор.
Размерность определяется моделью (384d для paraphrase-multilingual-MiniLM-L12-v2).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, List

import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    SentenceTransformer = None  # type: ignore[assignment]
    _ST_IMPORT_ERROR = e
else:
    _ST_IMPORT_ERROR = None


class SentenceVectorizer:
    """
    Embedding generator based on sentence-transformers.

    Uses the native model dimension (no projection).
    Vectors are L2-normalized for cosine similarity via inner product.
    """

    def __init__(
        self,
        normalize: bool = True,
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dimension: Optional[int] = None,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required"
            ) from _ST_IMPORT_ERROR

        self.normalize = normalize
        self.model_name = model_name or os.getenv(
            "RAG_ST_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.device = device or os.getenv("RAG_ST_DEVICE")

        self._model: Optional[SentenceTransformer] = None
        self._model_dim: Optional[int] = None

        self._init_model()

        # dimension is now always the native model dimension
        # the parameter is accepted for backward compatibility but ignored
        self.dimension = self._model_dim

    def _init_model(self) -> None:
        if self._model is not None:
            return
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._model_dim = self._model.get_sentence_embedding_dimension()

    def _encode_many(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vecs = self._model.encode(
            texts,
            batch_size=int(os.getenv("RAG_ST_BATCH_SIZE", "32")),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        ).astype(np.float32)

        return vecs

    def embed(self, text: str) -> np.ndarray:
        if text is None:
            raise ValueError("text must not be None")
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")
        return self._encode_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        items = [str(t).strip() for t in texts if t and str(t).strip()]
        return self._encode_many(items)
