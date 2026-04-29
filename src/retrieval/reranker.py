from __future__ import annotations

"""
Cross-encoder reranker for post-retrieval precision improvement.

Uses a lightweight cross-encoder model to re-score the top-K candidates
from hybrid search, selecting the best top-N for final output.
"""

from typing import List, Optional

from src.core.models import ScoredChunk


class CrossEncoderReranker:
    """
    Re-ranks ScoredChunk candidates by query-document cross-encoder score.

    Lazy-loads the model on first use to avoid startup cost when disabled.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self._model_name)

    def rerank(
        self, query: str, candidates: List[ScoredChunk], top_k: int = 5
    ) -> List[ScoredChunk]:
        if not candidates:
            return []

        self._load_model()

        pairs = [
            (query, self._chunk_text(sc))
            for sc in candidates
        ]

        scores = self._model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: float(x[1]), reverse=True)

        result: List[ScoredChunk] = []
        for sc, cross_score in scored[:top_k]:
            sc.hybrid_score = float(cross_score)
            result.append(sc)

        return result

    @staticmethod
    def _chunk_text(sc: ScoredChunk) -> str:
        search_context = (sc.chunk.search_context or "").strip()
        text = (sc.chunk.text or "")[:500].strip()
        if search_context and text:
            return f"{search_context}\n{text}"
        return search_context or text or ""
