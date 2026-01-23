from __future__ import annotations

"""
Небольшой HTTP‑клиент для Ollama.

Задачи:
- инкапсулировать работу с HTTP API
- дать простой, подменяемый интерфейс для юнит‑тестов
- не заниматься генерацией финального ответа RAG (только служебные вызовы)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama2"
    timeout: float = 60.0


class OllamaClient:
    """
    Обёртка над Ollama HTTP API.

    ВАЖНО:
    - этот клиент используется только для:
      * enrichment чанков при подготовке БД
      * enrichment пользовательских запросов
      * LLM‑based reranking
    - генерация финального ответа на вопрос пользователя здесь НЕ реализуется.
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()

    def generate(self, prompt: str, *, system_prompt: str | None = None, **params: Any) -> str:
        """
        Получить текстовую completion от Ollama.

        Интерфейс специально упрощён: вызывающий код сам отвечает за формат prompt
        и JSON‑схему ответа.
        """
        url = f"{self.config.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            # отключаем стриминг, чтобы вернуть цельный текст
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        payload.update(params)

        # ВНИМАНИЕ: это минимальный, отладочно‑ориентированный клиент.
        # В боевом коде сюда можно добавить:
        # - retries
        # - логирование
        # - более подробную обработку ошибок
        resp = requests.post(url, json=payload, timeout=self.config.timeout)
        resp.raise_for_status()
        data = resp.json()
        # Ollama по умолчанию возвращает поле "response" с текстом
        return data.get("response", "")

