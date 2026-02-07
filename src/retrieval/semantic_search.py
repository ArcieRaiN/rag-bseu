from __future__ import annotations

"""
Semantic Search поверх FAISS (PIPELINE 3.1).

Ответственность:
- загрузка FAISS‑индекса и массива чанков (data.json)
- выдача Top‑K по cosine similarity для embedded_query

Не знает ни о BM25, ни о metadata‑score, ни о reranking.
"""

from pathlib import Path
from typing import List
import json

import faiss  # type: ignore
import numpy as np

from src.core.models import Chunk, ScoredChunk


class FaissSemanticSearcher:
    """
    Обёртка над FAISS для поиска по embeddings поля `context`.
    """

    def __init__(self, index_path: Path, data_path: Path):
        self._index_path = Path(index_path)
        self._data_path = Path(data_path)

        self._index = self._load_index(self._index_path)
        self._chunks = self._load_chunks(self._data_path)

        if self._index.ntotal != len(self._chunks):
            raise ValueError(
                f"FAISS index size ({self._index.ntotal}) != number of chunks ({len(self._chunks)})"
            )

    # -------------------- Публичный интерфейс -------------------- #

    def search(self, embedded_query: np.ndarray, top_k: int) -> List[ScoredChunk]:
        """
        Вернуть Top‑K чанков с заполненным semantic_score.

        Метрика: cosine similarity.
        FAISS индекс должен быть построен для косинусного пространства (IndexFlatIP + нормализация).
        """
        query_vec = self._prepare_query(embedded_query)
        scores, indices = self._index.search(query_vec, top_k)
        return self._format_results(scores[0], indices[0])

    def get_all_chunks(self) -> List[Chunk]:
        """
        Безопасный геттер для доступа к всем чанкам (используется HybridSearcher и тестами).
        """
        return self._chunks

    # -------------------- Приватные методы -------------------- #

    @staticmethod
    def _load_index(path: Path) -> faiss.Index:
        return faiss.read_index(str(path))

    @staticmethod
    def _load_chunks(path: Path) -> List[Chunk]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        chunks: List[Chunk] = []
        for item in raw:
            chunks.append(
                Chunk(
                    id=str(item["id"]),
                    context=item["context"],
                    text=item["text"],
                    source=item["source"],
                    page=int(item["page"]),
                    geo=item.get("geo"),
                    metrics=item.get("metrics"),
                    years=item.get("years") or [],
                    time_granularity=item.get("time_granularity"),
                    oked=item.get("oked"),
                    extra={k: v for k, v in item.items() if k not in {
                        "id", "context", "text", "source", "page",
                        "geo", "metrics", "years", "time_granularity", "oked"
                    }},
                )
            )
        return chunks

    @staticmethod
    def _prepare_query(embedded_query: np.ndarray) -> np.ndarray:
        if embedded_query.ndim == 1:
            return embedded_query.reshape(1, -1).astype("float32")
        return embedded_query.astype("float32")

    def _format_results(self, scores: np.ndarray, indices: np.ndarray) -> List[ScoredChunk]:
        results: List[ScoredChunk] = []
        for score, idx in zip(scores, indices):
            if 0 <= idx < len(self._chunks):
                results.append(
                    ScoredChunk(
                        chunk=self._chunks[idx],
                        semantic_score=float(score),
                    )
                )
        return results
