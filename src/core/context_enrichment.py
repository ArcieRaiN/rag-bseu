from __future__ import annotations
"""
Обогащение пользовательских запросов для retrieval (без LLM).

Regex-извлечение years и geo + embedding через SentenceVectorizer.
Время: <10 ms (vs ~60 s при LLM-обогащении).
"""

import re
from typing import List, Optional

from src.core.models import EnrichedQuery
from src.core.models import Chunk
from src.enrichers.rule_metadata_extractor import RuleMetadataExtractor, normalize_year_text
from src.vectorstore.vectorizer import SentenceVectorizer

KNOWN_GEO = {
    "беларусь", "белоруссия", "россия", "российская федерация",
    "минск", "брест", "гродно", "витебск", "могилев", "гомель",
    "минская", "брестская", "гродненская", "витебская", "могилевская", "гомельская",
    "снг", "евразийский",
}

YEAR_RE = re.compile(r'\b(19\d{2}|20\d{2})\b')


def _extract_years(query: str) -> List[int]:
    matches = YEAR_RE.findall(normalize_year_text(query))
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
        self._rules = RuleMetadataExtractor()

    def enrich(self, query: str) -> EnrichedQuery:
        embedded = self._vectorizer.embed(query, is_query=True)

        years = _extract_years(query)
        rule_meta = self._rules.extract(Chunk(text=query, search_context=query))
        geo = rule_meta.geo or _extract_geo(query)
        metrics = rule_meta.metric_candidates or self._extract_query_metrics(query)

        return EnrichedQuery(
            query=query,
            embedded_query=embedded.astype("float32"),
            geo=geo if geo else None,
            years=years if years else None,
            metrics=metrics if metrics else None,
        )

    @staticmethod
    def _extract_query_metrics(query: str) -> List[str]:
        cleaned = re.sub(YEAR_RE, " ", normalize_year_text(query))
        for geo in KNOWN_GEO:
            cleaned = re.sub(re.escape(geo), " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(за|по|на|в|и|или|с|до|от|годы|год|динамика|сравнение)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?.,;:")
        return [cleaned] if len(cleaned) >= 4 else []
