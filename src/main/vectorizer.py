from __future__ import annotations

import hashlib
import os
from typing import Iterable, Optional

import numpy as np

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = e
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None


class SentenceVectorizer:
    """
    Sentence-transformers based vectorizer (drop-in replacement).

    Why the name is kept:
    - the rest of the codebase imports `SentenceVectorizer`
    - we keep interface compatibility: `embed`, `embed_many`, `dimension`

    Notes:
    - SentenceTransformer outputs a fixed embedding size depending on the model.
    - If `dimension` differs from the model embedding size, we apply a deterministic
      random projection to `dimension` so existing configs (e.g. 256) keep working.
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
            raise ValueError("dimension must be positive")

        if SentenceTransformer is None:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required to use this vectorizer"
            ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR

        self.normalize = normalize

        self.model_name = model_name or os.getenv(
            "RAG_ST_MODEL",
            # good default for RU queries
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.device = device or os.getenv("RAG_ST_DEVICE")  # e.g. "cpu", "cuda"

        # Lazy-ish init is possible, but we keep it simple and explicit.
        self._model = SentenceTransformer(self.model_name, device=self.device)

        # Determine model embedding dimension
        model_dim = int(getattr(self._model, "get_sentence_embedding_dimension")())
        self._model_dim = model_dim

        # Output dimension (what FAISS expects / configured in pipeline)
        self.dimension = int(dimension)

        # Projection matrix (built only if needed)
        self._proj: Optional[np.ndarray] = None
        if self.dimension != self._model_dim:
            self._proj = self._make_projection(self._model_dim, self.dimension, seed=self.model_name)

    @staticmethod
    def _make_projection(input_dim: int, output_dim: int, *, seed: str) -> np.ndarray:
        """
        Deterministic Gaussian random projection matrix of shape (output_dim, input_dim).
        """
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        rng_seed = int(digest[:16], 16)
        rng = np.random.default_rng(rng_seed)
        # scale to preserve norm in expectation
        proj = rng.standard_normal((output_dim, input_dim), dtype=np.float32) / np.sqrt(float(output_dim))
        return proj

    def _encode_many(self, texts: list[str]) -> np.ndarray:
        # sentence-transformers can normalize internally; we normalize ourselves after projection
        vecs = self._model.encode(
            texts,
            batch_size=int(os.getenv("RAG_ST_BATCH_SIZE", "32")),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        if vecs.ndim != 2:
            raise RuntimeError(f"Unexpected embeddings shape: {vecs.shape}")

        if self._proj is not None:
            vecs = (self._proj @ vecs.T).T.astype(np.float32)

        if self.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / (norms + 1e-9)

        return vecs

    def embed(self, text: str) -> np.ndarray:
        if text is None:
            raise ValueError("text must not be None")
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")
        return self._encode_many([text])[0]

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        items = []
        for t in texts:
            if t is None:
                raise ValueError("text must not be None")
            s = str(t).strip()
            if not s:
                raise ValueError("text must be non-empty")
            items.append(s)
        if not items:
            # keep shape consistent: (0, dimension)
            return np.zeros((0, self.dimension), dtype=np.float32)
        return self._encode_many(items)
