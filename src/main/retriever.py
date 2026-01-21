import json
from pathlib import Path
from typing import List, Dict
import numpy as np

from src.main.vectorizer import HashVectorizer
from src.main.input_normalizer import normalize_text_lemmatized


class SemanticRetriever:
    """
    Семантический поиск по нормализованным текстовым чанкам
    """

    def __init__(
        self,
        vectorizer: HashVectorizer,
        data_path: Path,
    ):
        self.vectorizer = vectorizer
        self.data_path = Path(data_path)

        self.data = self._load_data()

        # берём ГОТОВУЮ нормализацию из data.json
        self.texts = [item["normalized"] for item in self.data]

        self.embeddings = np.stack(
            [self.vectorizer.embed(t) for t in self.texts]
        ).astype(np.float32)

    def _load_data(self) -> List[Dict]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        query = query / (np.linalg.norm(query) + 1e-9)
        matrix = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        )
        return matrix @ query

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not query.strip():
            return []

        normalized_query = normalize_text_lemmatized(query)
        if not normalized_query:
            return []

        query_vec = self.vectorizer.embed(normalized_query).astype(np.float32)
        sims = self._cosine_similarity(query_vec)

        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_idx:
            item = self.data[idx]
            results.append(
                {
                    "score": float(sims[idx]),
                    "text": item["text"],
                    "source": item["source"],
                    "page": item["page"],
                }
            )

        return results
