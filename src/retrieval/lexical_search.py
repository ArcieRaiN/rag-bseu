from __future__ import annotations

"""
Lexical (BM25) search over chunks.

Simplified: single BM25 index over concatenated text+context.
Tokenization: lowercase + split (no Natasha lemmatization for speed).
"""

import re
from typing import List

from rank_bm25 import BM25Okapi

from src.core.models import Chunk, ScoredChunk


_TOKEN_RE = re.compile(r'[а-яёa-z0-9]+', re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase + word-level split."""
    return _TOKEN_RE.findall(text.lower())


class BM25Search:
    """
    BM25-based lexical search using rank_bm25.

    Builds one index over `text + context` for each chunk.
    """

    def __init__(self, chunks: List[Chunk], k1: float = 1.5, b: float = 0.75):
        self._chunks = chunks
        self._corpus_tokens: List[List[str]] = []

        for ch in chunks:
            combined = f"{ch.context or ''} {ch.text or ''}"
            self._corpus_tokens.append(_tokenize(combined))

        self._bm25 = BM25Okapi(self._corpus_tokens, k1=k1, b=b)

    def search(self, query: str, top_k: int = 20) -> List[ScoredChunk]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        top_indices = scores.argsort()[::-1][:top_k]

        results: List[ScoredChunk] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                break
            results.append(ScoredChunk(
                chunk=self._chunks[idx],
                lexical_score=score,
            ))

        return results
