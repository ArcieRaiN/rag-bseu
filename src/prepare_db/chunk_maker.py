import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

from src.main.file_utils import list_pdfs
from src.main.vectorizer import HashVectorizer


@dataclass
class VectorStoreArtifacts:
    index_path: Path
    metadata_path: Path
    data_path: Path


def _flatten_table(table: List[List]) -> str:
    """Convert a 2D table into a flat string for embedding."""
    return " ".join(str(cell) for row in table for cell in row)


class ChunkMaker:
    """
    Prepares table chunks and builds a vector store (faiss or numpy fallback).
    """

    def __init__(
        self,
        vectorizer: HashVectorizer,
        documents_dir: Optional[Path] = None,
        vector_store_dir: Optional[Path] = None,
    ):
        self.vectorizer = vectorizer
        self.documents_dir = Path(documents_dir or Path(__file__).parent / "documents")
        self.vector_store_dir = Path(vector_store_dir or Path(__file__).parent / "vector_store")
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

    def build_from_tables(
        self, tables: List[Dict], output_dir: Optional[Path] = None
    ) -> VectorStoreArtifacts:
        """
        Build vector store files from already extracted tables.
        Each table dict must include: title, data (list of rows), source, and optional page.
        """
        if not tables:
            raise ValueError("tables must not be empty")

        out_dir = Path(output_dir or self.vector_store_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        embeddings = []
        metadata: Dict[str, Dict] = {}
        data_entries: List[Dict] = []

        for idx, table in enumerate(tables):
            title = table.get("title") or f"table_{idx}"
            table_data = table.get("data") or []
            source = table.get("source") or ""
            page = table.get("page")
            text_for_embedding = f"{title} {_flatten_table(table_data)}"
            embedding = self.vectorizer.embed(text_for_embedding)

            embeddings.append(embedding)
            metadata[str(idx)] = {"title": title, "source": source, "page": page}
            data_entries.append(
                {"id": idx, "title": title, "data": table_data, "source": source, "page": page}
            )

        emb_array = np.stack(embeddings).astype(np.float32)
        artifacts = VectorStoreArtifacts(
            index_path=out_dir / "index.faiss",
            metadata_path=out_dir / "metadata.json",
            data_path=out_dir / "data.json",
        )

        self._save_index(emb_array, artifacts.index_path)
        with open(artifacts.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        with open(artifacts.data_path, "w", encoding="utf-8") as f:
            json.dump(data_entries, f, ensure_ascii=False, indent=2)

        return artifacts

    def build_from_pdfs(self, output_dir: Optional[Path] = None) -> VectorStoreArtifacts:
        """
        Minimal PDF handler: uses document names as titles to keep the prototype running
        without heavy PDF parsing dependencies.
        """
        pdfs = list_pdfs(self.documents_dir)
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {self.documents_dir}")

        tables = []
        for pdf in pdfs:
            tables.append(
                {
                    "title": pdf.stem,
                    "data": [["document", pdf.name]],
                    "source": str(pdf),
                    "page": 1,
                }
            )
        return self.build_from_tables(tables, output_dir=output_dir)

    def _save_index(self, embeddings: np.ndarray, index_path: Path) -> None:
        if faiss is not None:
            index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            faiss.write_index(index, str(index_path))
        else:
            np.save(index_path, embeddings)

