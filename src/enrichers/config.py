"""
Конфигурация LLM-обогащения чанков.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class EnricherConfig:
    """
    Параметры работы LLMEnricher.

    Attributes:
        max_parallel_requests: Макс. параллельных запросов (зарезервировано).
        reset_interval: Через сколько чанков сбрасывать контекст LLM.
        max_retries: Количество повторных попыток при ошибке парсинга.
        keep_alive: Время жизни модели в памяти Ollama (например, ``"5m"``).
        request_options: Низкоуровневые параметры запроса к Ollama.
    """
    # concurrency (currently unused if processing one chunk at a time)
    max_parallel_requests: int = 4

    # after how many calls to call client.reset_context()
    reset_interval: int = 50

    # per-chunk attempts to ask LLM (parsing/retry)
    max_retries: int = 2

    # keep_alive value for Ollama (string, e.g. "5m") or None
    keep_alive: str | None = "5m"

    # low-level request options passed to Ollama (wrapper)
    request_options: Dict[str, Any] = None

    def __post_init__(self):
        if self.request_options is None:
            self.request_options = {"temperature": 0, "top_p": 1, "num_predict": 512}
