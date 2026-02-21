"""
Парсинг JSON-ответов от LLM.

Обрабатывает типичные проблемы: markdown code fences, мусор вокруг JSON,
несбалансированные скобки. Используется LLMEnricher для извлечения
структурированных метаданных из сырых LLM-ответов.
"""

from __future__ import annotations
import json
from typing import Optional, Dict, Any


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Извлекает первый сбалансированный JSON-объект из текста.

    Сканирует скобки ``{`` / ``}`` для определения границ объекта,
    вместо regex с рекурсией (которую stdlib ``re`` не поддерживает).

    Args:
        text: Сырой текст, содержащий JSON.

    Returns:
        Подстрока ``{...}`` или None, если объект не найден.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_single_enrichment(raw: str) -> Optional[Dict[str, Any]]:
    """
    Парсит ответ LLM с метаданными обогащения.

    Стратегия:
    1. Убрать markdown code fences (````` ```json ... ``` `````)
    2. Попробовать ``json.loads`` на весь очищенный текст
    3. Fallback: извлечь первый сбалансированный ``{...}`` и распарсить

    Args:
        raw: Сырая строка ответа LLM.

    Returns:
        Словарь с метаданными или None при ошибке парсинга.
    """
    if not raw:
        return None
    txt = raw.strip()

    # Drop markdown code fences if present
    if txt.startswith("```"):
        # drop triple fence header
        first_nl = txt.find("\n")
        if first_nl != -1:
            txt = txt[first_nl + 1 :]
        if txt.endswith("```"):
            txt = txt[:-3]
        txt = txt.strip()

    # Try direct parse
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find first balanced {...}
    snippet = _extract_first_json_object(txt)
    if snippet:
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None
