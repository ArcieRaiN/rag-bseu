from __future__ import annotations

"""
Metadata Scoring для гибридного поиска.

Взвешенный скоринг совпадений метаданных чанка и запроса:
- geo (40%): exact/partial token overlap
- metrics (30%): Jaccard по нормализованным токенам
- years (30%): intersection по множествам лет

Возвращает значение в [0, 1].
"""

import re
from typing import List, Optional, Set

from src.core.models import Chunk, EnrichedQuery

_TOKEN_RE = re.compile(r'[а-яёa-z0-9]+', re.IGNORECASE)


def _normalize_simple(text: str) -> str:
    if not text:
        return ""
    return " ".join(_TOKEN_RE.findall(text.lower()))


class MetadataScorer:
    """
    Scores metadata overlap between a chunk and an enriched query.
    Returns a value in [0, 1].
    """

    def score(self, chunk: Chunk, query: EnrichedQuery) -> float:
        geo_score = self._geo_score(chunk.geo, query.geo)
        metrics_score = self._metrics_score(chunk.metrics, query.metrics or [])
        years_score = self._years_score(chunk.years or [], query.years or [])

        total = (
            0.40 * geo_score
            + 0.30 * metrics_score
            + 0.30 * years_score
        )
        return float(max(0.0, min(1.0, total)))

    @staticmethod
    def _geo_score(
        chunk_geo: Optional[str] | List[str],
        query_geo: Optional[str] | List[str],
    ) -> float:
        if not chunk_geo or not query_geo:
            return 0.0

        def _to_str(geo) -> str:
            if isinstance(geo, list):
                return " ".join(str(g) for g in geo)
            return str(geo)

        c = _normalize_simple(_to_str(chunk_geo))
        q = _normalize_simple(_to_str(query_geo))
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
            c_norm.update(_normalize_simple(str(m)).split())
        q_norm: Set[str] = set()
        for m in query_metrics:
            q_norm.update(_normalize_simple(str(m)).split())
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
        if max(chunk_years) < min(query_years) or max(query_years) < min(chunk_years):
            return 0.0
        return 0.3
