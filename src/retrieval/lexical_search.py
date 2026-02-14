from __future__ import annotations

"""
Lexical Search (BM25‑подобный) по полям text + context (PIPELINE 3.2).

Задачи:
- проиндексировать чанки в памяти
- по запросу вернуть Top‑K кандидатов с заполненным lexical_score

Реализация BM25 упрощена и ориентирована на отладку. При необходимости
её можно заменить на rank_bm25 или другой готовый backend, не меняя интерфейс.
"""

from dataclasses import dataclass
from typing import Iterable, List
import math

from src.core.models import Chunk, ScoredChunk, EnrichedQuery
from src.core.input_normalizer import normalize_text_lemmatized
from src.core.config import LexicalSearchConfig


@dataclass
class _Posting:
    doc_id: int
    tf: int


class InMemoryBM25:
    """
    Простейший BM25‑подобный индекс для text+context с поддержкой весов полей.
    """

    def __init__(self, chunks: List[Chunk], config: LexicalSearchConfig | None = None):
        self._chunks = chunks
        self._config = config or LexicalSearchConfig()
        self._index = {}  # term -> List[_Posting]
        self._doc_len = []
        self._avg_doc_len = 0.0
        self._build()

    # -------------------- Публичный интерфейс -------------------- #

    def search(self, enriched_query: EnrichedQuery, top_k: int) -> List[ScoredChunk]:
        terms = self._make_query_terms(enriched_query)
        if not terms:
            return []

        scores = self._score_documents(terms)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: List[ScoredChunk] = []
        for doc_id, score in ranked:
            if doc_id < 0 or doc_id >= len(self._chunks):
                continue
            results.append(
                ScoredChunk(
                    chunk=self._chunks[doc_id],
                    lexical_score=float(score),
                )
            )
        return results

    # -------------------- Индексация -------------------- #

    def _build(self) -> None:
        total_len = 0
        for doc_id, ch in enumerate(self._chunks):
            # Разделяем токены по полям
            text_tokens = [t for t in normalize_text_lemmatized(ch.text).split() if t]
            context_tokens = [t for t in normalize_text_lemmatized(ch.context).split() if t]

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

            # длина документа для BM25
            dl = len(text_tokens) + len(context_tokens) + len(hints_tokens)
            self._doc_len.append(dl)
            total_len += dl

            # строим tf_map для каждого токена
            for term, tf in self._make_tf_map(text_tokens + context_tokens + hints_tokens).items():
                self._index.setdefault(term, []).append(_Posting(doc_id=doc_id, tf=tf))

        self._avg_doc_len = total_len / max(len(self._chunks), 1)

    def _make_tf_map(self, tokens: list[str]) -> dict[str, int]:
        tf_map: dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        return tf_map

    # -------------------- Подготовка запроса -------------------- #

    def _make_query_terms(self, enriched_query: EnrichedQuery) -> List[str]:
        terms = [t for t in normalize_text_lemmatized(enriched_query.query).split() if t]
        if enriched_query.geo:
            terms.extend([t for t in normalize_text_lemmatized(enriched_query.geo).split() if t])
        if enriched_query.metrics:
            for m in enriched_query.metrics:
                terms.extend([t for t in normalize_text_lemmatized(str(m)).split() if t])
        return terms

    # -------------------- BM25‑подобный скоринг с весами -------------------- #

    def _score_documents(self, query_terms: Iterable[str]) -> dict[int, float]:
        scores: dict[int, float] = {}
        N = len(self._chunks)

        for term in query_terms:
            postings = self._index.get(term)
            if not postings:
                continue

            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            for p in postings:
                dl = self._doc_len[p.doc_id]
                denom = p.tf + self._config.k1 * (1.0 - self._config.b + self._config.b * dl / max(self._avg_doc_len, 1e-6))
                score = idf * (p.tf * (self._config.k1 + 1.0)) / denom
                # применяем веса полей
                scores[p.doc_id] = scores.get(p.doc_id, 0.0) + (
                    score * (self._config.w_text + self._config.w_context + self._config.w_hints)
                )

        return scores
