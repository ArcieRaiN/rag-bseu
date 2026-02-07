from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Sequence


@dataclass
class Chunk:
    id: str
    context: str
    text: str
    source: str
    page: int

    doc_id: Optional[str] = None

    geo: Optional[str] = None
    metrics: Optional[List[str]] = None
    years: Optional[List[int]] = None
    time_granularity: Optional[str] = None
    oked: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedQuery:
    query: str
    embedded_query: Sequence[float]

    geo: Optional[str] = None
    years: Optional[List[int]] = None
    metrics: Optional[List[str]] = None
    time_granularity: Optional[str] = None
    oked: Optional[str] = None

    raw_llm_response: Optional[str] = None


@dataclass
class ScoredChunk:
    chunk: Chunk

    semantic_score: float = 0.0
    lexical_score: float = 0.0
    metadata_score: float = 0.0

    hybrid_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class PipelineResult:
    query: str
    enriched_query: EnrichedQuery
    candidates: List[ScoredChunk]
    top_chunks: List[ScoredChunk]

    timings: Dict[str, float] = field(default_factory=dict)
