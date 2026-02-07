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
    """
    Конфигурация клиента Ollama.

    По умолчанию используем qwen2.5:7b как основную модель.
    """

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    timeout: float = 120.0

    # Дефолтные параметры генерации — можно переопределить в вызове generate(...)
    temperature: float = 0.0
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    num_predict: int = 800

    # Формат ответа; для JSON‑задач можно указать "json"
    format: Optional[str] = None


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

    def reset_context(self) -> None:
        """
        Сбрасывает контекст модели через очистку кеша.

        В Ollama это делается через параметр keep_alive: "0" в запросе,
        который очищает кеш модели в памяти.
        """
        url = f"{self.config.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": "",
            "stream": False,
            "keep_alive": "0",
            "num_predict": 1,
        }
        try:
            resp = requests.post(url, json=payload, timeout=5.0)
            resp.raise_for_status()
        except Exception:
            # Игнорируем ошибки при сбросе - это не критично
            pass

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_retries: int = 3,
        **params: Any,
    ) -> str:
        """
        Получить текстовую completion от Ollama.

        Вызов выглядит так:

            client = OllamaClient(
                OllamaConfig(
                    model="qwen2.5:7b",
                    format="json",
                )
            )

            result = client.generate(prompt, system_prompt=system_prompt)
        """
        url = f"{self.config.base_url}/api/generate"

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            # Дефолтные параметры генерации
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repeat_penalty": self.config.repeat_penalty,
            "num_predict": self.config.num_predict,
        }

        # Если формат задан в конфиге — используем его как дефолт
        if self.config.format:
            payload["format"] = self.config.format

        if system_prompt:
            payload["system"] = system_prompt

        # Параметры вызова имеют приоритет над конфигом
        payload.update(params)

        import time

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, json=payload, timeout=self.config.timeout)
                resp.raise_for_status()
                data = resp.json()
                # Ollama по умолчанию возвращает поле "response" с текстом
                return data.get("response", "")
            except requests.exceptions.ReadTimeout:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5
                    print(
                        f"⚠️  Таймаут при запросе к Ollama (попытка {attempt + 1}/{max_retries}). "
                        f"Повтор через {wait_time} сек..."
                    )
                    time.sleep(wait_time)
                else:
                    print(
                        f"❌ Ollama недоступен: превышено кол-во попыток. "
                        f"Таймаут: {self.config.timeout} сек\n"
                        "→ Проверьте, что Ollama запущен (ollama serve)\n"
                        f"→ Если нужно, скачайте модель: ollama pull {self.config.model}"
                    )
                    raise
            except requests.exceptions.RequestException as e:
                print(f"❌ Ollama недоступен: {e}\n"
                      "→ Проверьте, что Ollama запущен (ollama serve)\n"
                      f"→ Если нужно, скачайте модель: ollama pull {self.config.model}")
                raise

