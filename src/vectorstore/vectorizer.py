from __future__ import annotations

import hashlib
import os
from typing import Iterable, Optional, List

import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]
    _ST_IMPORT_ERROR = e
else:
    _ST_IMPORT_ERROR = None


class SentenceVectorizer:
    """
    Sentence-transformers based vectorizer.

    Features:
    - Lazy initialization
    - Optional deterministic random projection to target dimension
    - Normalization to unit vectors
    """

    def __init__(
        self,
        dimension: int = 256,
        normalize: bool = True,
        *,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if dimension <= 0:
            raise ValueError("`dimension` must be positive")
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required"
            ) from _ST_IMPORT_ERROR

        self.dimension = dimension
        self.normalize = normalize
        self.model_name = model_name or os.getenv(
            "RAG_ST_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.device = device or os.getenv("RAG_ST_DEVICE")

        self._model: Optional[SentenceTransformer] = None
        self._model_dim: Optional[int] = None
        self._proj: Optional[np.ndarray] = None

        self._init_model()

    def _init_model(self) -> None:
        """Lazy-ish model initialization."""
        if self._model is not None:
            return
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._model_dim = self._model.get_sentence_embedding_dimension()
        if self.dimension != self._model_dim:
            self._proj = self._make_projection(self._model_dim, self.dimension, seed=self.model_name)

    @staticmethod
    def _make_projection(input_dim: int, output_dim: int, *, seed: str) -> np.ndarray:
        """Deterministic Gaussian random projection."""
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        rng_seed = int(digest[:16], 16)
        rng = np.random.default_rng(rng_seed)
        proj = rng.standard_normal((output_dim, input_dim), dtype=np.float32) / np.sqrt(output_dim)
        return proj

    def _encode_many(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        vecs = self._model.encode(
            texts,
            batch_size=int(os.getenv("RAG_ST_BATCH_SIZE", "32")),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)

        if self._proj is not None:
            vecs = (self._proj @ vecs.T).T.astype(np.float32)

        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / (norms + 1e-9)

        return vecs

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if text is None:
            raise ValueError("text must not be None")
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")
        return self._encode_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        items = [str(t).strip() for t in texts if t and str(t).strip()]
        return self._encode_many(items)
