from typing import Dict, List

import pandas as pd

from src.main.retriever import TableRetriever


class RagPipeline:
    """
    Simplified RAG pipeline:
    - embed user query
    - retrieve top tables
    - format an answer from retrieved tables
    """

    def __init__(self, retriever: TableRetriever, default_top_k: int = 3):
        self.retriever = retriever
        self.default_top_k = default_top_k

    def run(self, query: str, top_k: int | None = None) -> Dict:
        k = top_k or self.default_top_k
        hits = self.retriever.search(query, top_k=k)
        answer = self._compose_answer(query, hits)
        return {"query": query, "hits": hits, "answer": answer}

    def _compose_answer(self, query: str, hits: List[Dict]) -> str:
        if not hits:
            return f"По запросу «{query}» таблицы не найдены."
        formatted_tables = []
        for hit in hits:
            table = hit.get("table") or []
            if not table:
                formatted_tables.append(f"{hit['title']} (нет данных таблицы)")
                continue
            df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
            formatted_tables.append(
                f"{hit['title']} (источник: {hit.get('source')}, стр.: {hit.get('page')})\n"
                f"{df.to_string(index=False)}"
            )
        return "\n\n".join(formatted_tables)

