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

from src.main.models import Chunk, EnrichedQuery
from src.main.input_normalizer import normalize_text_lemmatized


class MetadataScorer:
    """
    Отдельный компонент, считающий metadata_score для чанка относительно EnrichedQuery.

    ВАЖНО:
    - возвращаемое значение уже нормализовано в [0, 1]
    - веса внутри метода можно тонко настраивать без изменения остальных модулей
    """

    def score(self, chunk: Chunk, query: EnrichedQuery) -> float:
        # geo: exact / partial
        geo_score = self._geo_score(chunk.geo, query.geo)
        metrics_score = self._metrics_score(chunk.metrics, query.metrics or [])
        years_score = self._years_score(chunk.years or [], query.years or [])
        tg_score = self._time_granularity_score(
            chunk.time_granularity, query.time_granularity
        )
        oked_score = self._oked_score(chunk.oked, query.oked)

        # Внутренние веса внутри metadata_score (отдельно от hybrid_score)
        w_geo = 0.35
        w_metrics = 0.30
        w_years = 0.20
        w_tg = 0.10
        w_oked = 0.05

        total = (
            w_geo * geo_score
            + w_metrics * metrics_score
            + w_years * years_score
            + w_tg * tg_score
            + w_oked * oked_score
        )
        # total по определению в [0,1]
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
        # частичное совпадение по подстроке/токенам
        c_tokens = set(c.split())
        q_tokens = set(q.split())
        if not c_tokens or not q_tokens:
            return 0.0
        inter = c_tokens & q_tokens
        if not inter:
            return 0.0
        # доля пересечения
        return len(inter) / max(len(q_tokens), 1)

    @staticmethod
    def _metrics_score(
        chunk_metrics: Optional[List[str]], query_metrics: List[str]
    ) -> float:
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
        inter = c_norm & q_norm
        if not inter:
            return 0.0
        # Jaccard‑подобная мера
        return len(inter) / len(c_norm | q_norm)

    @staticmethod
    def _years_score(chunk_years: List[int], query_years: List[int]) -> float:
        if not chunk_years or not query_years:
            return 0.0
        c_set = set(chunk_years)
        q_set = set(query_years)
        inter = c_set & q_set
        if not inter:
            # мягкий штраф за сдвиг по годам: если интервалы не пересекаются вообще
            if max(chunk_years) < min(query_years) or max(query_years) < min(chunk_years):
                return 0.0
            return 0.3
        # доля пересечения относительно запроса
        return len(inter) / len(q_set)

    @staticmethod
    def _time_granularity_score(
        chunk_tg: Optional[str], query_tg: Optional[str]
    ) -> float:
        if not query_tg or not chunk_tg:
            return 0.0
        return 1.0 if chunk_tg == query_tg else 0.0

    @staticmethod
    def _oked_score(chunk_oked: Optional[str], query_oked: Optional[str]) -> float:
        if not chunk_oked or not query_oked:
            return 0.0
        c = chunk_oked.strip().lower()
        q = query_oked.strip().lower()
        if c == q:
            return 1.0
        # частичное совпадение по префиксу кода
        if c.startswith(q) or q.startswith(c):
            return 0.7
        return 0.0

