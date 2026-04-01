from __future__ import annotations
"""
Context enrichment for user queries.

Uses regex-based extraction for structured fields (years, geo)
instead of LLM calls, making retrieval ~100x faster.
"""

import re
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.core.models import EnrichedQuery
from src.vectorstore.vectorizer import SentenceVectorizer

KNOWN_GEO = {
    "беларусь", "белоруссия", "россия", "российская федерация",
    "минск", "брест", "гродно", "витебск", "могилев", "гомель",
    "минская", "брестская", "гродненская", "витебская", "могилевская", "гомельская",
    "снг", "евразийский",
}

YEAR_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')


def _extract_years(query: str) -> List[int]:
    matches = YEAR_RE.findall(query)
    years = sorted(set(int(y) for y in matches))
    return years


def _extract_geo(query: str) -> List[str]:
    query_lower = query.lower()
    found: List[str] = []
    for geo in KNOWN_GEO:
        if geo in query_lower:
            found.append(geo.title())
    return found


class QueryContextEnricher:
    """
    Fast query enrichment using embedding + regex extraction.
    No LLM calls -- all processing is deterministic and sub-second.
    """

    def __init__(self, vectorizer: SentenceVectorizer, llm_client=None):
        self._vectorizer = vectorizer
        # llm_client accepted for backward compatibility but not used

    def enrich(self, query: str) -> EnrichedQuery:
        embedded = self._vectorizer.embed(query)

        years = _extract_years(query)
        geo = _extract_geo(query)

        return EnrichedQuery(
            query=query,
            embedded_query=embedded.astype("float32"),
            geo=geo if geo else None,
            years=years if years else None,
            metrics=None,
            time_granularity=None,
            oked=None,
            raw_llm_response=None,
        )
