from __future__ import annotations

"""
Lexical Search через rank_bm25 с поддержкой весов полей (text/context/hints).
Интерфейс search(query, top_k)
"""

from typing import List
from rank_bm25 import BM25Okapi

from src.core.models import Chunk, ScoredChunk, EnrichedQuery
from src.core.input_normalizer import normalize_text_lemmatized
from src.core.config import LexicalSearchConfig


class RankBM25Search:
    """
    Lexical search с использованием rank_bm25 + Cython для ускорения.
    Поддерживает веса полей:
        w_text, w_context, w_hints
    """

    def __init__(self, chunks: List[Chunk], config: LexicalSearchConfig | None = None):
        self._chunks = chunks
        self._config = config or LexicalSearchConfig()

        # токенизация по документам и полям
        self._text_tokens: List[List[str]] = []
        self._context_tokens: List[List[str]] = []
        self._hints_tokens: List[List[str]] = []

        self._build()

    # -------------------- Индексация -------------------- #
    def _build(self) -> None:
        for ch in self._chunks:
            # text
            text_tokens = [t for t in normalize_text_lemmatized(ch.text).split() if t]
            self._text_tokens.append(text_tokens)

            # context
            context_tokens = [t for t in normalize_text_lemmatized(ch.context).split() if t]
            self._context_tokens.append(context_tokens)

            # hints: metrics / geo / years
            hints_parts: list[str] = []
            if ch.metrics:
                hints_parts.append(" ".join(str(m) for m in ch.metrics[:3]) if isinstance(ch.metrics, list) else str(ch.metrics))
            if ch.geo:
                hints_parts.append(" ".join(str(g) for g in ch.geo) if isinstance(ch.geo, list) else str(ch.geo))
            if ch.years:
                if isinstance(ch.years, list):
                    y_sorted = sorted(ch.years)
                    years_repr = f"{y_sorted[0]}-{y_sorted[-1]}" if len(y_sorted) > 1 else str(y_sorted[0])
                    hints_parts.append(years_repr)
                else:
                    hints_parts.append(str(ch.years))
            hints_tokens = [t for t in normalize_text_lemmatized(" ".join(hints_parts)).split() if t]
            self._hints_tokens.append(hints_tokens)

        # Создаём BM25 объекты для каждого поля
        self._bm25_text = BM25Okapi(self._text_tokens)
        self._bm25_context = BM25Okapi(self._context_tokens)
        self._bm25_hints = BM25Okapi(self._hints_tokens)

    # -------------------- Подготовка запроса -------------------- #
    def _make_query_terms(self, enriched_query: EnrichedQuery) -> List[str]:
        terms = [t for t in normalize_text_lemmatized(enriched_query.query).split() if t]
        if enriched_query.geo:
            terms.extend([t for t in normalize_text_lemmatized(enriched_query.geo).split() if t])
        if enriched_query.metrics:
            for m in enriched_query.metrics:
                terms.extend([t for t in normalize_text_lemmatized(str(m)).split() if t])
        return terms

    # -------------------- Поиск -------------------- #
    def search(self, enriched_query: EnrichedQuery, top_k: int) -> List[ScoredChunk]:
        query_terms = self._make_query_terms(enriched_query)
        if not query_terms:
            return []

        # ранжирование по каждому полю
        scores_text = self._bm25_text.get_scores(query_terms)
        scores_context = self._bm25_context.get_scores(query_terms)
        scores_hints = self._bm25_hints.get_scores(query_terms)

        # суммируем с весами
        combined_scores = []
        for i, _ in enumerate(self._chunks):
            score = (
                self._config.w_text * scores_text[i] +
                self._config.w_context * scores_context[i] +
                self._config.w_hints * scores_hints[i]
            )
            combined_scores.append(score)

        # берём top_k
        top_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append(
                ScoredChunk(
                    chunk=self._chunks[idx],
                    lexical_score=float(combined_scores[idx])
                )
            )
        return results
