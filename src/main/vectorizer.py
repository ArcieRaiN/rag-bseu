import hashlib
from typing import Iterable, List

import numpy as np


class HashVectorizer:
    """
    Deterministic text vectorizer based on hashing.
    Produces reproducible embeddings without external services.
    """

    def __init__(self, dimension: int = 128, normalize: bool = True):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.normalize = normalize

    def _seed_from_text(self, text: str) -> int:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Use lower bits for a stable RNG seed
        return int(digest[:16], 16)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single piece of text."""
        if text is None:
            raise ValueError("text must not be None")
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")
        rng = np.random.default_rng(self._seed_from_text(text))
        vec = rng.standard_normal(self.dimension, dtype=np.float32)
        if self.normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        """Embed multiple texts into a 2D numpy array."""
        vectors: List[np.ndarray] = [self.embed(t) for t in texts]
        return np.vstack(vectors)

