from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EnricherConfig:
    """
    Configuration for LLMEnricher.
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
