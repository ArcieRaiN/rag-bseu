from __future__ import annotations

"""
Гибридный поиск с Reciprocal Rank Fusion (RRF).

Объединяет три канала:
- Semantic Search (FAISS, cosine similarity, top-40)
- Lexical Search (BM25Okapi по search_context+text, top-40)
- Metadata Scoring (geo/years/metrics overlap, top-30)

RRF: score(d) = Σ 1/(K + rank_i), K=60.
Score-agnostic fusion — не требует нормализации между каналами.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

from src.core.models import EnrichedQuery, ScoredChunk
from src.core.config import RetrievalConfig
from src.retrieval.semantic_search import FaissSemanticSearcher
from src.retrieval.lexical_search import BM25Search
from src.retrieval.metadata_scorer import MetadataScorer


@dataclass
class HybridSearchResult:
    candidates: List[ScoredChunk]
    debug_info: Dict[str, object]


class HybridSearcher:
    """
    Hybrid retrieval using RRF (Reciprocal Rank Fusion).

    RRF formula: score(d) = sum(1 / (K + rank_i)) for each retriever i.
    No score normalization needed -- robust across different score scales.
    """

    def __init__(self, semantic_searcher: FaissSemanticSearcher, config: RetrievalConfig):
        self._semantic = semantic_searcher
        self._config = config
        self._metadata_scorer = MetadataScorer()
        self._lexical = BM25Search(
            self._semantic.get_all_chunks(),
            k1=self._config.bm25_k1,
            b=self._config.bm25_b,
        )

    def search(self, enriched_query: EnrichedQuery) -> HybridSearchResult:
        sem_results = self._semantic_search(enriched_query)
        lex_results = self._lexical_search(enriched_query)
        meta_results = self._metadata_search(enriched_query)

        final = self._rrf_fusion(sem_results, lex_results, meta_results)
        final = final[:self._config.final_top_k]

        debug = {
            "semantic_count": len(sem_results),
            "lexical_count": len(lex_results),
            "metadata_count": len(meta_results),
            "final_count": len(final),
        }
        return HybridSearchResult(candidates=final, debug_info=debug)

    def _semantic_search(self, enriched_query: EnrichedQuery) -> List[ScoredChunk]:
        return self._semantic.search(
            enriched_query.embedded_query,
            top_k=self._config.semantic_top_k
        )

    def _lexical_search(self, enriched_query: EnrichedQuery) -> List[ScoredChunk]:
        return self._lexical.search(
            enriched_query.query,
            top_k=self._config.lexical_top_k
        )

    def _metadata_search(self, enriched_query: EnrichedQuery) -> List[ScoredChunk]:
        chunks = self._semantic.get_all_chunks()
        scored: List[ScoredChunk] = []
        for ch in chunks:
            ms = self._metadata_scorer.score(ch, enriched_query)
            if ms > 0.0:
                scored.append(ScoredChunk(chunk=ch, metadata_score=float(ms)))
        scored.sort(key=lambda sc: sc.metadata_score, reverse=True)
        return scored[:self._config.metadata_top_k]

    def _rrf_fusion(
        self,
        sem_results: List[ScoredChunk],
        lex_results: List[ScoredChunk],
        meta_results: List[ScoredChunk],
    ) -> List[ScoredChunk]:
        K = self._config.rrf_k
        chunk_map: Dict[str, ScoredChunk] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)

        for rank, sc in enumerate(sem_results, 1):
            key = sc.chunk.id
            rrf_scores[key] += 1.0 / (K + rank)
            if key not in chunk_map:
                chunk_map[key] = ScoredChunk(chunk=sc.chunk, semantic_score=sc.semantic_score)
            else:
                chunk_map[key].semantic_score = sc.semantic_score

        for rank, sc in enumerate(lex_results, 1):
            key = sc.chunk.id
            rrf_scores[key] += 1.0 / (K + rank)
            if key not in chunk_map:
                chunk_map[key] = ScoredChunk(chunk=sc.chunk, lexical_score=sc.lexical_score)
            else:
                chunk_map[key].lexical_score = sc.lexical_score

        for rank, sc in enumerate(meta_results, 1):
            key = sc.chunk.id
            rrf_scores[key] += 1.0 / (K + rank)
            if key not in chunk_map:
                chunk_map[key] = ScoredChunk(chunk=sc.chunk, metadata_score=sc.metadata_score)
            else:
                chunk_map[key].metadata_score = sc.metadata_score

        for key, sc in chunk_map.items():
            sc.hybrid_score = rrf_scores[key]

        ranked = sorted(chunk_map.values(), key=lambda x: x.hybrid_score, reverse=True)
        return ranked
