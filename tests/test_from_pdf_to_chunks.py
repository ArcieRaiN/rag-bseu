from pathlib import Path

from src.main.retriever import TableRetriever
from src.main.vectorizer import HashVectorizer
from src.prepare_db.chunk_maker import ChunkMaker


def test_from_pdf_to_chunks_returns_top3(tmp_path: Path):
    docs_dir = Path(__file__).resolve().parents[1] / "src" / "prepare_db" / "documents"
    vectorizer = HashVectorizer(dimension=32)
    chunk_maker = ChunkMaker(vectorizer, documents_dir=docs_dir)

    artifacts = chunk_maker.build_from_pdfs(output_dir=tmp_path)
    retriever = TableRetriever(
        vectorizer=vectorizer,
        index_path=artifacts.index_path,
        metadata_path=artifacts.metadata_path,
        data_path=artifacts.data_path,
    )

    results = retriever.search("население беларусь", top_k=3)
    assert len(results) == 3

    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
