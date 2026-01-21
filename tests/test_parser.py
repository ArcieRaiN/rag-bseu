import json
from pathlib import Path

import numpy as np
import pytest

from src.main.vectorizer import HashVectorizer
from src.prepare_db.chunk_maker import ChunkMaker

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def sample_tables(tmp_path: Path):
    return [
        {
            "title": "Численность населения по городам",
            "data": [["Город", "Численность"], ["Брест", 340000], ["Пинск", 130000]],
            "source": str(tmp_path / "doc.pdf"),
            "page": 1,
        },
        {
            "title": "ВРП по областям",
            "data": [["Область", "ВРП"], ["Брестская", 123.4], ["Минская", 210.5]],
            "source": str(tmp_path / "doc.pdf"),
            "page": 2,
        },
    ]


def test_chunk_maker_builds_vector_store(tmp_path: Path):
    vectorizer = HashVectorizer(dimension=16)
    maker = ChunkMaker(vectorizer, documents_dir=tmp_path / "docs")

    tables = sample_tables(tmp_path)
    artifacts = maker.build_from_tables(tables, output_dir=tmp_path)

    assert artifacts.index_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.data_path.exists()

    with open(artifacts.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert len(metadata) == len(tables)

    with open(artifacts.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == len(tables)
    assert data[0]["title"] == tables[0]["title"]

    if faiss is None:
        embeddings = np.load(artifacts.index_path)
        assert embeddings.shape == (len(tables), vectorizer.dimension)
    else:
        assert artifacts.index_path.stat().st_size > 0


def test_build_from_pdfs_without_files_raises(tmp_path: Path):
    vectorizer = HashVectorizer(dimension=8)
    maker = ChunkMaker(vectorizer, documents_dir=tmp_path / "empty")
    with pytest.raises(FileNotFoundError):
        maker.build_from_pdfs(output_dir=tmp_path)

