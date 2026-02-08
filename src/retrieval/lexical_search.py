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


@dataclass
class _Posting:
    doc_id: int
    tf: int


class InMemoryBM25:
    """
    Простейший BM25‑подобный индекс для text+context.

    Не оптимизирован под миллион документов, но прекрасно подходит:
    - для локальных экспериментов
    - для чтения и отладки алгоритма
    """

    def __init__(self, chunks: List[Chunk]):
        self._chunks = chunks
        self._index = {}  # term -> List[_Posting]
        self._doc_len = []  # количество токенов в документе
        self._avg_doc_len = 0.0

        self._build()

    # -------------------- Публичный интерфейс -------------------- #

    def search(self, enriched_query: EnrichedQuery, top_k: int) -> List[ScoredChunk]:
        """
        Поиск по ключевым словам:
        - из исходного текста запроса
        - плюс geo / metrics как дополнительные ключевые слова
        """
        terms = self._make_query_terms(enriched_query)
        if not terms:
            return []

        scores = self._score_documents(terms)
        # Берём top_k по score
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
        """
        Строим обратный индекс по объединённому тексту:
        `context + " " + text + query_hints(geo/metrics/years)`.
        """
        total_len = 0
        for doc_id, ch in enumerate(self._chunks):
            # Базовый текст: обогащённый context + сырой text
            combined = f"{ch.context}\n{ch.text}"

            # Добавляем псевдо‑запросы (query hints), построенные из метаданных.
            hints_parts: list[str] = []

            # Метрики
            if ch.metrics:
                if isinstance(ch.metrics, list):
                    hints_parts.append(" ".join(str(m) for m in ch.metrics[:3]))
                else:
                    hints_parts.append(str(ch.metrics))

            # Гео
            if ch.geo:
                if isinstance(ch.geo, list):
                    hints_parts.append(" ".join(str(g) for g in ch.geo))
                else:
                    hints_parts.append(str(ch.geo))

            # Годы
            if ch.years:
                if isinstance(ch.years, list):
                    years_sorted = sorted(ch.years)
                    if len(years_sorted) > 1:
                        years_repr = f"{years_sorted[0]}-{years_sorted[-1]}"
                    else:
                        years_repr = str(years_sorted[0])
                    hints_parts.append(years_repr)
                else:
                    hints_parts.append(str(ch.years))

            if hints_parts:
                combined += "\n" + " ".join(hints_parts)

            # Нормализация текста и токенизация
            norm = normalize_text_lemmatized(combined)
            tokens = [t for t in norm.split() if t]
            self._doc_len.append(len(tokens))
            total_len += len(tokens)

            # Построение tf_map
            tf_map = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term, tf in tf_map.items():
                self._index.setdefault(term, []).append(_Posting(doc_id=doc_id, tf=tf))

        self._avg_doc_len = total_len / max(len(self._chunks), 1)

    # -------------------- Подготовка запроса -------------------- #

    def _make_query_terms(self, enriched_query: EnrichedQuery) -> List[str]:
        """
        Формируем набор терминов для lexical‑поиска:
        - лемматизированный текст запроса
        - geo и metrics (если заданы) как дополнительные токены
        """
        base = normalize_text_lemmatized(enriched_query.query)
        terms = [t for t in base.split() if t]

        if enriched_query.geo:
            terms.extend(
                t for t in normalize_text_lemmatized(enriched_query.geo).split() if t
            )
        if enriched_query.metrics:
            for m in enriched_query.metrics:
                terms.extend(
                    t
                    for t in normalize_text_lemmatized(str(m)).split()
                    if t
                )
        # Можно добавить эвристику по годам / окэду при необходимости
        return terms

    # -------------------- BM25‑подобный скоринг -------------------- #

    def _score_documents(self, query_terms: Iterable[str]) -> dict[int, float]:
        """
        Упрощённая реализация BM25:
        - k1, b заданы константами, достаточными для отладки.
        """
        k1 = 1.5
        b = 0.75

        scores: dict[int, float] = {}
        N = len(self._chunks)

        for term in query_terms:
            postings = self._index.get(term)
            if not postings:
                continue

            df = len(postings)
            # классический idf BM25
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

            for p in postings:
                dl = self._doc_len[p.doc_id]
                denom = p.tf + k1 * (1.0 - b + b * dl / max(self._avg_doc_len, 1e-6))
                score = idf * (p.tf * (k1 + 1.0)) / denom
                scores[p.doc_id] = scores.get(p.doc_id, 0.0) + score

        return scores

