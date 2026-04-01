from __future__ import annotations

"""
Конфигурация гибридного retrieval pipeline.

Параметры RRF: K=60 (сглаживание рангов).
По умолчанию cross-encoder reranking выключен (use_reranker=False).
"""

from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    """Parameters for the hybrid retrieval pipeline."""

    semantic_top_k: int = 40
    lexical_top_k: int = 40
    metadata_top_k: int = 30
    final_top_k: int = 10

    rrf_k: int = 60

    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = 5
