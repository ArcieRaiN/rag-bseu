from pathlib import Path

from src.main.rag_pipeline import RagPipeline
from src.main.retriever import TableRetriever
from src.main.vectorizer import HashVectorizer
from src.prepare_db.chunk_maker import ChunkMaker


def prepare_store(tmp_path: Path):
    vectorizer = HashVectorizer(dimension=24)
    maker = ChunkMaker(vectorizer, documents_dir=tmp_path / "docs")
    tables = [
        {
            "title": "Численность населения по городам",
            "data": [["Город", "Численность"], ["Брест", 340000], ["Барановичи", 170000]],
            "source": "local",
            "page": 1,
        },
        {
            "title": "Экспорт по областям",
            "data": [["Область", "Экспорт"], ["Гомельская", 200], ["Брестская", 180]],
            "source": "local",
            "page": 2,
        },
    ]
    artifacts = maker.build_from_tables(tables, output_dir=tmp_path)
    return vectorizer, artifacts


def test_retriever_returns_relevant_rows(tmp_path: Path):
    vectorizer, artifacts = prepare_store(tmp_path)
    retriever = TableRetriever(
        vectorizer=vectorizer,
        index_path=artifacts.index_path,
        metadata_path=artifacts.metadata_path,
        data_path=artifacts.data_path,
    )

    results = retriever.search("население Брест", top_k=1)
    assert results
    top = results[0]
    assert "население" in top["title"].lower()
    assert any("Брест" in str(cell) for row in top["table"] for cell in row)


def test_rag_pipeline_formats_answer(tmp_path: Path):
    vectorizer, artifacts = prepare_store(tmp_path)
    retriever = TableRetriever(
        vectorizer=vectorizer,
        index_path=artifacts.index_path,
        metadata_path=artifacts.metadata_path,
        data_path=artifacts.data_path,
    )
    pipeline = RagPipeline(retriever, default_top_k=1)

    result = pipeline.run("экспорт по областям")
    assert result["hits"]
    assert "экспорт" in result["hits"][0]["title"].lower()
    assert "экспорт" in result["answer"].lower()

