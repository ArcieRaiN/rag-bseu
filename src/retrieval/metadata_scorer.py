from __future__ import annotations

"""
Metadata Scoring (PIPELINE 3.3).

Оценка совпадения:
- geo (exact/partial)
- metrics (fuzzy)
- years (пересечение диапазонов)
- time_granularity (exact)
- oked (optional)
"""

from typing import List, Optional, Set

from src.core.models import Chunk, EnrichedQuery
from src.core.input_normalizer import normalize_text_lemmatized


class MetadataScorer:
    """
    Отдельный компонент, считающий metadata_score для чанка относительно EnrichedQuery.

    ВАЖНО:
    - возвращаемое значение уже нормализовано в [0, 1]
    - веса внутри метода можно тонко настраивать без изменения остальных модулей
    """

    def score(self, chunk: Chunk, query: EnrichedQuery) -> float:
        geo_score = self._geo_score(chunk.geo, query.geo)
        metrics_score = self._metrics_score(chunk.metrics, query.metrics or [])
        years_score = self._years_score(chunk.years or [], query.years or [])
        tg_score = self._time_granularity_score(chunk.time_granularity, query.time_granularity)
        oked_score = self._oked_score(chunk.oked, query.oked)

        # Внутренние веса
        w_geo = 0.35
        w_metrics = 0.30
        w_years = 0.20
        w_tg = 0.05
        w_oked = 0.10

        total = (
            w_geo * geo_score
            + w_metrics * metrics_score
            + w_years * years_score
            + w_tg * tg_score
            + w_oked * oked_score
        )
        return float(max(0.0, min(1.0, total)))

    # -------------------- Частные метрики -------------------- #

    @staticmethod
    def _geo_score(chunk_geo: Optional[str], query_geo: Optional[str]) -> float:
        if not chunk_geo or not query_geo:
            return 0.0
        c = normalize_text_lemmatized(chunk_geo)
        q = normalize_text_lemmatized(query_geo)
        if c == q:
            return 1.0
        c_tokens, q_tokens = set(c.split()), set(q.split())
        if not c_tokens or not q_tokens:
            return 0.0
        inter = c_tokens & q_tokens
        return len(inter) / max(len(q_tokens), 1)

    @staticmethod
    def _metrics_score(chunk_metrics: Optional[List[str]], query_metrics: List[str]) -> float:
        if not chunk_metrics or not query_metrics:
            return 0.0
        c_norm: Set[str] = set()
        for m in chunk_metrics:
            c_norm.update(normalize_text_lemmatized(str(m)).split())
        q_norm: Set[str] = set()
        for m in query_metrics:
            q_norm.update(normalize_text_lemmatized(str(m)).split())
        if not c_norm or not q_norm:
            return 0.0
        return len(c_norm & q_norm) / len(c_norm | q_norm)

    @staticmethod
    def _years_score(chunk_years: List[int], query_years: List[int]) -> float:
        if not chunk_years or not query_years:
            return 0.0
        c_set, q_set = set(chunk_years), set(query_years)
        inter = c_set & q_set
        if inter:
            return len(inter) / len(q_set)
        # мягкий штраф, если интервалы не пересекаются
        if max(chunk_years) < min(query_years) or max(query_years) < min(chunk_years):
            return 0.0
        return 0.3

    @staticmethod
    def _time_granularity_score(chunk_tg: Optional[str], query_tg: Optional[str]) -> float:
        if not chunk_tg or not query_tg:
            return 0.0
        return 1.0 if chunk_tg == query_tg else 0.0

    @staticmethod
    def _oked_score(chunk_oked: Optional[str], query_oked: Optional[str]) -> float:
        if not chunk_oked or not query_oked:
            return 0.0
        c, q = chunk_oked.strip().lower(), query_oked.strip().lower()
        if c == q:
            return 1.0
        if c.startswith(q) or q.startswith(c):
            return 0.7
        return 0.0
