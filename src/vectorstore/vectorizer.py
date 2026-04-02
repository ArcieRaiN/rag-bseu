"""
Генерация эмбеддингов через sentence-transformers.

Для e5-моделей автоматически применяет префиксы "query: " / "passage: ".
Модель по умолчанию: intfloat/multilingual-e5-large (1024d).
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


def _is_e5_model(name: str) -> bool:
    return "e5" in name.lower()


class SentenceVectorizer:
    """
    Embedding generator based on sentence-transformers.

    Uses the native model dimension (no projection).
    Vectors are L2-normalized for cosine similarity via inner product.
    For e5 models, text prefixes ("query: " / "passage: ") are applied automatically.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(
        self,
        normalize: bool = True,
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required"
            ) from _ST_IMPORT_ERROR

        self.normalize = normalize
        self.model_name = model_name or os.getenv(
            "RAG_ST_MODEL", "intfloat/multilingual-e5-large"
        )
        self.device = device or os.getenv("RAG_ST_DEVICE")
        self._is_e5 = _is_e5_model(self.model_name)

        self._model: Optional[SentenceTransformer] = None
        self._model_dim: Optional[int] = None

        self._init_model()
        self.dimension = self._model_dim

    def _init_model(self) -> None:
        if self._model is not None:
            return
        kwargs = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = SentenceTransformer(self.model_name, **kwargs)
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

    def embed(self, text: str, *, is_query: bool = True) -> np.ndarray:
        """Embed a single text. For e5 models, is_query controls the prefix."""
        if text is None:
            raise ValueError("text must not be None")
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")
        if self._is_e5:
            prefix = self.QUERY_PREFIX if is_query else self.PASSAGE_PREFIX
            text = prefix + text
        return self._encode_many([text])[0]

    def embed_many(self, texts: Iterable[str], *, is_query: bool = True) -> np.ndarray:
        """Embed multiple texts. For e5 models, is_query controls the prefix."""
        items = [str(t).strip() for t in texts if t and str(t).strip()]
        if self._is_e5:
            prefix = self.QUERY_PREFIX if is_query else self.PASSAGE_PREFIX
            items = [prefix + t for t in items]
        return self._encode_many(items)
