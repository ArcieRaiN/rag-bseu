import json
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from src.main.input_normalizer import normalize_text_lemmatized
from src.main.vectorizer import HashVectorizer


@dataclass
class BuildArtifacts:
    index_path: Path
    metadata_path: Path
    data_path: Path


class ChunkMaker:
    def __init__(
        self,
        vectorizer: HashVectorizer,
        documents_dir: Path,
        min_words: int = 20,
    ):
        self.vectorizer = vectorizer
        self.documents_dir = documents_dir
        self.min_words = min_words

    # -----------------------------

    def build_from_pdfs(self, output_dir: Path) -> BuildArtifacts:
        chunks: List[Dict] = []
        embeddings = []

        chunk_id = 0

        for pdf_path in self.documents_dir.glob("*.pdf"):
            print(f"📘 Обработка: {pdf_path.name}")

            for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
                page_blocks = []

                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text = self._clean_text(element.get_text())
                        if text:
                            page_blocks.append(text)

                semantic_chunks = self._build_semantic_chunks(page_blocks)

                for chunk_text in semantic_chunks:
                    normalized = normalize_text_lemmatized(chunk_text)

                    if not normalized:
                        continue

                    emb = self.vectorizer.embed(normalized)

                    chunks.append(
                        {
                            "id": chunk_id,
                            "text": chunk_text,
                            "normalized": normalized,
                            "source": pdf_path.name,
                            "page": page_num,
                        }
                    )

                    embeddings.append(emb)
                    chunk_id += 1

        embeddings_np = np.array(embeddings, dtype=np.float32)

        index_path = output_dir / "index.npy"
        data_path = output_dir / "data.json"
        meta_path = output_dir / "metadata.json"

        np.save(index_path, embeddings_np)

        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vectorizer": "HashVectorizer",
                    "dimension": self.vectorizer.dimension,
                    "chunks": len(chunks),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return BuildArtifacts(
            index_path=index_path,
            metadata_path=meta_path,
            data_path=data_path,
        )

    # -----------------------------

    def _build_semantic_chunks(self, blocks: List[str]) -> List[str]:
        """
        Склеиваем текст по смыслу, а не по layout
        """
        chunks = []
        buffer = ""

        for block in blocks:
            if self._is_header(block):
                if self._is_good_chunk(buffer):
                    chunks.append(buffer.strip())
                buffer = block
            else:
                buffer += " " + block

        if self._is_good_chunk(buffer):
            chunks.append(buffer.strip())

        return chunks

    # -----------------------------

    def _is_header(self, text: str) -> bool:
        return (
            len(text.split()) <= 6
            and text.isupper()
        )

    def _is_good_chunk(self, text: str) -> bool:
        words = text.split()
        return (
            len(words) >= self.min_words
            and not text.strip().isdigit()
        )

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        if len(text) < 5:
            return ""

        return text
