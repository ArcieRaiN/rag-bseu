from __future__ import annotations

"""
Hybrid Search (PIPELINE 3.1–3.6).

Комбинирует:
- Semantic Search (FAISS)
- Lexical Search (BM25‑подобный)
- Metadata Scoring
и возвращает Top‑K кандидатов по hybrid_score.
"""

from dataclasses import dataclass
from typing import List, Dict

import numpy as np

from src.core.models import EnrichedQuery, ScoredChunk
from src.core.config import RetrievalConfig
from src.retrieval.semantic_search import FaissSemanticSearcher
from src.retrieval.lexical_search import InMemoryBM25
from src.retrieval.metadata_scorer import MetadataScorer


@dataclass
class HybridSearchResult:
    """
    Контейнер для результатов поиска с отладочной информацией.
    """
    candidates: List[ScoredChunk]
    debug_info: Dict[str, object]


class HybridSearcher:
    """
    Высокоуровневый фасад гибридного поиска.
    Не работает с LLM / reranking, только retrieval.
    """

    def __init__(self, semantic_searcher: FaissSemanticSearcher, config: RetrievalConfig):
        self._semantic = semantic_searcher
        self._config = config
        self._metadata_scorer = MetadataScorer()
        self._lexical = InMemoryBM25(self._semantic.get_all_chunks())  # через геттер

    # -------------------- Публичный интерфейс -------------------- #

    def search(self, enriched_query: EnrichedQuery) -> HybridSearchResult:
        sem_results = self._semantic_search(enriched_query)
        lex_results = self._lexical_search(enriched_query)
        merged = self._merge_results(sem_results, lex_results, enriched_query)
        self._apply_metadata_scoring(merged, enriched_query)
        self._normalize_scores(merged)
        self._compute_hybrid_score(merged)
        final = self._top_k(merged)
        debug = self._collect_debug_info(sem_results, lex_results, merged, enriched_query)
        return HybridSearchResult(candidates=final, debug_info=debug)

    # -------------------- Приватные этапы pipeline -------------------- #

    def _semantic_search(self, enriched_query: EnrichedQuery) -> List[ScoredChunk]:
        return self._semantic.search(
            enriched_query.embedded_query,
            top_k=self._config.semantic_top_k
        )

    def _lexical_search(self, enriched_query: EnrichedQuery) -> List[ScoredChunk]:
        return self._lexical.search(
            enriched_query,
            top_k=self._config.lexical_top_k
        )

    def _merge_results(
        self,
        sem_results: List[ScoredChunk],
        lex_results: List[ScoredChunk],
        enriched_query: EnrichedQuery
    ) -> List[ScoredChunk]:
        merged: Dict[str, ScoredChunk] = {}

        def _key(sc: ScoredChunk) -> str:
            return sc.chunk.id

        for sc in sem_results:
            merged[_key(sc)] = sc
        for sc in lex_results:
            key = _key(sc)
            if key in merged:
                merged[key].lexical_score = sc.lexical_score
            else:
                merged[key] = sc

        # Metadata recall для случаев, когда нужные чанки не попали ни в FAISS, ни в BM25
        if enriched_query.metrics:
            meta_seed = self._metadata_recall(enriched_query, top_k=max(50, self._config.final_top_k * 10))
            for sc in meta_seed:
                key = _key(sc)
                if key not in merged:
                    merged[key] = sc

        return list(merged.values())

    def _metadata_recall(self, enriched_query: EnrichedQuery, top_k: int) -> List[ScoredChunk]:
        chunks = self._semantic.get_all_chunks()
        if not chunks:
            return []

        scored: List[ScoredChunk] = []
        for ch in chunks:
            ms = self._metadata_scorer.score(ch, enriched_query)
            if ms <= 0.0:
                continue
            scored.append(ScoredChunk(chunk=ch, metadata_score=float(ms)))

        scored.sort(key=lambda sc: sc.metadata_score, reverse=True)
        return scored[:top_k]

    def _apply_metadata_scoring(self, candidates: List[ScoredChunk], enriched_query: EnrichedQuery) -> None:
        for sc in candidates:
            sc.metadata_score = self._metadata_scorer.score(sc.chunk, enriched_query)

    def _compute_hybrid_score(self, candidates: List[ScoredChunk]) -> None:
        for sc in candidates:
            sc.hybrid_score = (
                self._config.w_semantic * sc.semantic_score
                + self._config.w_lexical * sc.lexical_score
                + self._config.w_metadata * sc.metadata_score
            )

    def _top_k(self, candidates: List[ScoredChunk]) -> List[ScoredChunk]:
        candidates.sort(key=lambda x: x.hybrid_score, reverse=True)
        return candidates[: self._config.final_top_k]

    def _collect_debug_info(
        self,
        sem_results: List[ScoredChunk],
        lex_results: List[ScoredChunk],
        merged: List[ScoredChunk],
        enriched_query: EnrichedQuery
    ) -> Dict[str, object]:
        return {
            "semantic_count": len(sem_results),
            "lexical_count": len(lex_results),
            "merged_count": len(merged),
            "metadata_seed_count": len(merged) if enriched_query.metrics else 0,
        }

    # -------------------- Нормализация -------------------- #

    @staticmethod
    def _normalize_scores(candidates: List[ScoredChunk]) -> None:
        if not candidates:
            return

        sem = np.array([c.semantic_score for c in candidates], dtype=float)
        lex = np.array([c.lexical_score for c in candidates], dtype=float)

        sem_range = sem.max() - sem.min()
        if sem_range > 1e-3:
            sem_n = (sem - sem.min()) / (sem_range + 1e-9)
        else:
            sem_n = np.full_like(sem, fill_value=float(sem.mean() if sem.size else 0.0))

        lex_log = np.log1p(np.maximum(lex, 0.0))
        lex_range = lex_log.max() - lex_log.min()
        if lex_range > 1e-3:
            lex_n = (lex_log - lex_log.min()) / (lex_range + 1e-9)
        else:
            lex_n = lex_log

        for i, c in enumerate(candidates):
            c.semantic_score = float(sem_n[i])
            c.lexical_score = float(lex_n[i])
            # metadata_score оставляем как есть (уже [0,1])
