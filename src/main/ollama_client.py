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
    model: str = "llama3.2:3b"
    timeout: float = 300.0  # Увеличено до 5 минут для больших батчей


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

    def generate(self, prompt: str, *, system_prompt: str | None = None, max_retries: int = 3, **params: Any) -> str:
        """
        Получить текстовую completion от Ollama.

        Интерфейс специально упрощён: вызывающий код сам отвечает за формат prompt
        и JSON‑схему ответа.
        
        Args:
            prompt: Текст запроса
            system_prompt: Системный промпт
            max_retries: Максимальное количество повторных попыток при таймауте
            **params: Дополнительные параметры для Ollama API
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

        # Retry логика для обработки таймаутов
        import time
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=self.config.timeout)
                resp.raise_for_status()
                data = resp.json()
                # Ollama по умолчанию возвращает поле "response" с текстом
                return data.get("response", "")
            except requests.exceptions.ReadTimeout as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5  # Экспоненциальная задержка: 5, 10, 20 сек
                    print(f"⚠️  Таймаут при запросе к LLM (попытка {attempt + 1}/{max_retries}). Повтор через {wait_time} сек...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Ошибка: Превышено максимальное количество попыток. Таймаут: {self.config.timeout} сек")
                    raise
            except requests.exceptions.RequestException as e:
                print(f"❌ Ошибка при запросе к Ollama: {e}")
                raise

