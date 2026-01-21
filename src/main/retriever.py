import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None

from src.main.vectorizer import HashVectorizer


class _NumpyIndex:
    """
    Лёгкий индекс для поиска через numpy, fallback при отсутствии faiss.
    Косинусная близость.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.embeddings = np.zeros((0, dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.embeddings = (
            vectors if self.embeddings.size == 0 else np.vstack([self.embeddings, vectors])
        )

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings.size == 0:
            return np.array([]), np.array([])
        # Косинусная близость
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        emb_norms = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9)
        sims = emb_norms @ query_norm
        top_k_idx = np.argsort(sims)[::-1][:k]
        return sims[top_k_idx], top_k_idx

    def save(self, path: Path) -> None:
        np.save(path, self.embeddings)

    @classmethod
    def load(cls, path: Path) -> "_NumpyIndex":
        arr = np.load(path)
        inst = cls(arr.shape[1])
        inst.embeddings = arr
        return inst


class TableRetriever:
    """
    Поиск по таблицам (chunks) через векторное представление заголовков.
    """

    def __init__(
        self,
        vectorizer: HashVectorizer,
        index_path: Path,
        metadata_path: Path,
        data_path: Path,
    ):
        self.vectorizer = vectorizer
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.data_path = Path(data_path)

        self.metadata = self._load_metadata()
        self.data = self._load_data()
        self.index, self.using_faiss = self._load_index()

        # Создаём массив embeddings заголовков для поиска
        self.title_embeddings = self._compute_title_embeddings()

    def _load_metadata(self) -> Dict[str, Dict]:
        if not self.metadata_path.exists():
            return {}
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_data(self) -> Dict[str, Dict]:
        if not self.data_path.exists():
            return {}
        with open(self.data_path, "r", encoding="utf-8") as f:
            items = json.load(f)
        return {str(item["id"]): item for item in items}

    def _load_index(self):
        if faiss is not None and self.index_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                return index, True
            except Exception:
                pass
        if self.index_path.exists():
            return _NumpyIndex.load(self.index_path), False
        return _NumpyIndex(self.vectorizer.dimension), False

    def _compute_title_embeddings(self) -> np.ndarray:
        """
        Создаёт массив embedding'ов заголовков таблиц для поиска по заголовку.
        """
        titles = []
        for idx in sorted(self.data.keys(), key=int):
            entry = self.data[idx]
            title_text = entry.get("title", "")
            emb = self.vectorizer.embed(title_text)
            titles.append(emb)
        return np.stack(titles).astype(np.float32) if titles else np.zeros((0, self.vectorizer.dimension), dtype=np.float32)

    def _cosine_similarity(self, vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Косинусная близость."""
        vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
        matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        return matrix_norm @ vec_norm

    def search_by_table_title(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Ищет топ-K таблиц по заголовку (title) с косинусной близостью.
        """
        if not query or top_k <= 0:
            return []

        query_vec = self.vectorizer.embed(query).astype(np.float32)
        sims = self._cosine_similarity(query_vec, self.title_embeddings)
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            str_idx = str(idx)
            if str_idx not in self.data:
                continue
            entry = self.data[str_idx]
            meta = self.metadata.get(str_idx, {})
            results.append(
                {
                    "id": idx,
                    "score": float(sims[idx]),
                    "title": entry.get("title"),
                    "source": meta.get("source") or entry.get("source"),
                    "page": meta.get("page") or entry.get("page"),
                    "data": entry.get("data"),
                }
            )
        return results
