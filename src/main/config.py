from dataclasses import dataclass

@dataclass
class RetrievalConfig:
    """
    Конфигурация весов и параметров гибридного поиска.
    """
    semantic_top_k: int = 20
    lexical_top_k: int = 20
    final_top_k: int = 10
    w_semantic: float = 0.55
    w_lexical: float = 0.25
    w_metadata: float = 0.20

@dataclass
class RerankConfig:
    """
    Конфигурация reranking‑этапа.
    """
    top_k: int = 3
    model_name: str = "DiTy/cross-encoder-russian-msmarco"
    temperature: float = 0.0
    max_retries: int = 2
    max_workers: int = 4
    min_relevance_score: float = 0.2
