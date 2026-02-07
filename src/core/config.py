from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    """
    Конфигурация гибридного поиска.
    """
    semantic_top_k: int = 20
    lexical_top_k: int = 20
    final_top_k: int = 10

    w_semantic: float = 0.55   # embedding similarity
    w_lexical: float = 0.25   # BM25 / TF-IDF
    w_metadata: float = 0.20  # geo / years / metrics

    def __post_init__(self):
        s = self.w_semantic + self.w_lexical + self.w_metadata
        if not 0.99 <= s <= 1.01:
            raise ValueError(
                f"Retrieval weights must sum to 1.0, got {s}"
            )


@dataclass
class RerankConfig:
    """
    Конфигурация reranking-этапа.
    """
    top_k: int = 3
    model_name: str = "DiTy/cross-encoder-russian-msmarco"  # HF model id
    temperature: float = 0.0
    max_retries: int = 2
    max_workers: int = 4
    min_relevance_score: float = 0.2
