from dataclasses import dataclass


@dataclass
class RetrievalConfig:
    """
    Конфигурация гибридного поиска.
    """
    semantic_top_k: int = 20
    lexical_top_k: int = 20
    final_top_k: int = 5

    w_semantic: float = 0.55   # embedding similarity
    w_lexical: float = 0.25    # BM25 / TF-IDF
    w_metadata: float = 0.20   # geo / years / metrics

    def __post_init__(self):
        s = self.w_semantic + self.w_lexical + self.w_metadata
        if not 0.99 <= s <= 1.01:
            raise ValueError(f"Retrieval weights must sum to 1.0, got {s}")


@dataclass
class LexicalSearchConfig:
    """
    Конфигурация для BM25 / lexical search:
    отдельные веса для разных полей документа.
    """
    w_text: float = 1.0      # вес основного текста chunk.text
    w_context: float = 1.0   # вес context / описания
    w_hints: float = 1.0     # вес метаданных: geo / metrics / years

    k1: float = 1.5          # BM25-параметр k1
    b: float = 0.75          # BM25-параметр b
