import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

from src.main.vectorizer import HashVectorizer


class _NumpyIndex:
    """Fallback in-memory index with cosine similarity."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.embeddings = np.zeros((0, dimension), dtype=np.float32)

    def add(self, vectors: np.ndarray) -> None:
        self.embeddings = vectors if self.embeddings.size == 0 else np.vstack([self.embeddings, vectors])

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.embeddings.size == 0:
            return np.array([]), np.array([])
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
    """Retrieve table chunks by comparing query embeddings against a vector store."""

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

    def _search_faiss(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        q = np.array([query_vec]).astype(np.float32)
        faiss.normalize_L2(q)
        sims, ids = self.index.search(q, top_k)
        return sims[0], ids[0]

    def _search_numpy(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.index.search(query_vec, top_k)

    def search_by_table_title(self, query: str, top_k: int = 1) -> List[Dict]:
        """
        Search tables by title embedding.
        Returns top_k results with score, title, data, source, page.
        """
        query_vec = self.vectorizer.embed(query).astype(np.float32)
        query_vec = query_vec.reshape(1, -1)

        if self.using_faiss:
            faiss.normalize_L2(query_vec)
            scores, ids = self.index.search(query_vec, top_k)
            scores = scores[0]
            ids = ids[0]
        else:
            scores, ids = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores, ids):
            if idx == -1 or str(idx) not in self.data:
                continue
            entry = self.data[str(idx)]
            results.append({
                "id": idx,
                "title": entry.get("title"),
                "data": entry.get("data"),
                "source": entry.get("source"),
                "page": entry.get("page"),
                "score": float(score)
            })
        return results
