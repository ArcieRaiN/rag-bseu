import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLineHorizontal

try:
    import faiss
except ImportError:
    faiss = None

from src.main.vectorizer import HashVectorizer
from src.main.file_utils import list_pdfs


@dataclass
class VectorStoreArtifacts:
    index_path: Path
    metadata_path: Path
    data_path: Path


def extract_tables_from_pdf(pdf_path: Path) -> List[Dict]:
    """
    Ищет таблицы в PDF. Для каждой таблицы:
    - title = первая строка
    - data = все строки таблицы
    """
    tables: List[Dict] = []

    for page_idx, page_layout in enumerate(extract_pages(str(pdf_path)), start=1):
        # получаем все строки текста на странице
        lines: List[str] = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for line in element:
                    if isinstance(line, LTTextLineHorizontal):
                        text = line.get_text().strip()
                        if text:
                            lines.append(text)
        if not lines:
            continue

        # Простейшее разбиение на таблицы по пустой строке
        current_table: List[List[str]] = []
        for line in lines + [""]:
            row = line.split()  # простое разбиение на ячейки по пробелам
            if not line.strip():
                if current_table:
                    title = current_table[0][0] if current_table[0] else "Без названия"
                    tables.append({
                        "title": title,
                        "data": current_table,
                        "source": str(pdf_path),
                        "page": page_idx
                    })
                    current_table = []
            else:
                current_table.append(row)
    if not tables:
        # fallback, если таблиц не найдено
        tables.append({
            "title": pdf_path.stem,
            "data": [["text"], [pdf_path.name]],
            "source": str(pdf_path),
            "page": 1
        })
    return tables


class ChunkMaker:
    """
    Строим векторное хранилище из таблиц PDF.
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

    def build_from_pdfs(self, output_dir: Optional[Path] = None) -> VectorStoreArtifacts:
        pdfs = list_pdfs(self.documents_dir)
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found in {self.documents_dir}")

        all_tables: List[Dict] = []
        for pdf in pdfs:
            tables = extract_tables_from_pdf(pdf)
            all_tables.extend(tables)

        return self.build_from_tables(all_tables, output_dir=output_dir)

    def build_from_tables(
        self, tables: List[Dict], output_dir: Optional[Path] = None
    ) -> VectorStoreArtifacts:
        out_dir = Path(output_dir or self.vector_store_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        embeddings = []
        metadata: Dict[str, Dict] = {}
        data_entries: List[Dict] = []

        for idx, table in enumerate(tables):
            title = table.get("title") or f"table_{idx}"
            text_for_embedding = title  # embedding только по заголовку
            embedding = self.vectorizer.embed(text_for_embedding)
            embeddings.append(embedding)

            metadata[str(idx)] = {
                "title": title,
                "source": table.get("source"),
                "page": table.get("page")
            }
            data_entries.append({
                "id": idx,
                "title": title,
                "data": table.get("data"),
                "source": table.get("source"),
                "page": table.get("page")
            })

        emb_array = np.stack(embeddings).astype(np.float32)

        artifacts = VectorStoreArtifacts(
            index_path=out_dir / "index.faiss",
            metadata_path=out_dir / "metadata.json",
            data_path=out_dir / "data.json",
        )

        # сохраняем индекс
        if faiss is not None:
            index = faiss.IndexFlatIP(emb_array.shape[1])
            faiss.normalize_L2(emb_array)
            index.add(emb_array)
            faiss.write_index(index, str(artifacts.index_path))
        else:
            with open(artifacts.index_path, "wb") as f:
                np.save(f, emb_array)

        with open(artifacts.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        with open(artifacts.data_path, "w", encoding="utf-8") as f:
            json.dump(data_entries, f, ensure_ascii=False, indent=2)

        return artifacts
