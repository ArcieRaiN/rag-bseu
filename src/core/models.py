from __future__ import annotations

"""
Data models for the RAG pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Chunk:
    id: str = ""
    context: str = ""
    text: str = ""
    source: str = ""
    page: int = 0
    section: Optional[str] = None
    geo: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    years: Optional[List[int]] = field(default_factory=list)
    time_granularity: Optional[str] = None
    oked: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


@dataclass
class ScoredChunk:
    chunk: Chunk
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    metadata_score: float = 0.0
    hybrid_score: float = 0.0


@dataclass
class EnrichedQuery:
    query: str
    embedded_query: Optional[np.ndarray] = None
    geo: Optional[List[str]] = None
    years: Optional[List[int]] = None
    metrics: Optional[List[str]] = None
    time_granularity: Optional[str] = None
    oked: Optional[str] = None
    raw_llm_response: Optional[str] = None


@dataclass
class PipelineResult:
    query: str
    enriched_query: Optional[EnrichedQuery] = None
    candidates: List[ScoredChunk] = field(default_factory=list)
    top_chunks: List[ScoredChunk] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)
