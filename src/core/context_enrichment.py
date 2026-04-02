from __future__ import annotations
"""
Обогащение пользовательских запросов для retrieval.

Regex-извлечение структурированных полей (years, geo) без LLM-вызовов.
Время: <10ms vs ~60s при LLM-обогащении — ускорение ~6000x.
"""

import re
from typing import List, Optional

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

    def enrich(self, query: str) -> EnrichedQuery:
        embedded = self._vectorizer.embed(query, is_query=True)

        years = _extract_years(query)
        geo = _extract_geo(query)

        return EnrichedQuery(
            query=query,
            embedded_query=embedded.astype("float32"),
            geo=geo if geo else None,
            years=years if years else None,
            metrics=None,
        )
