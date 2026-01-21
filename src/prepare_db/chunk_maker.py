import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

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


def _extract_text_chunks(pdf_path: Path, max_chars: int = 800, overlap: int = 80) -> List[Dict]:
    """
    Extract text from a PDF and split it into overlapping character chunks.
    Falls back to a single stub chunk if parsing fails.
    """
    chunks: List[Dict] = []
    try:
        pages_text: List[str] = []
        for page_layout in extract_pages(str(pdf_path)):
            parts: List[str] = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    text = element.get_text().strip()
                    if text:
                        parts.append(text)
            if parts:
                pages_text.append(" ".join(parts))

        for page_idx, text in enumerate(pages_text, start=1):
            normalized = " ".join(text.split())
            if not normalized:
                continue
            start = 0
            chunk_idx = 1
            while start < len(normalized):
                end = start + max_chars
                chunk_text = normalized[start:end]
                chunks.append(
                    {
                        "title": f"{pdf_path.stem} p{page_idx} chunk{chunk_idx}",
                        "data": [["text"], [chunk_text]],
                        "source": str(pdf_path),
                        "page": page_idx,
                        "raw_text": chunk_text,
                    }
                )
                start = max(end - overlap, end) if overlap >= max_chars else end - overlap
                chunk_idx += 1
    except Exception:
        pass

    if not chunks:
        chunks.append(
            {
                "title": pdf_path.stem,
                "data": [["text"], [pdf_path.name]],
                "source": str(pdf_path),
                "page": 1,
                "raw_text": pdf_path.stem,
            }
        )
    return chunks


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
        Optional key `text` overrides embedding text (useful for non-tabular chunks).
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
            text_for_embedding = table.get("text") or f"{title} {_flatten_table(table_data)}"
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
        Parse PDFs from the documents directory, split text into chunks, and build the vector store.
        Falls back to filename-based chunks if parsing fails, ensuring at least one chunk per PDF.
        """
        pdfs = list_pdfs(self.documents_dir)
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {self.documents_dir}")

        tables: List[Dict] = []
        for pdf in pdfs:
            chunk_entries = _extract_text_chunks(pdf)
            for chunk in chunk_entries:
                tables.append(
                    {
                        "title": chunk["title"],
                        "data": chunk["data"],
                        "source": chunk["source"],
                        "page": chunk["page"],
                        "text": chunk["raw_text"],
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
            with open(index_path, "wb") as f:
                np.save(f, embeddings)

