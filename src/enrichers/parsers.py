from __future__ import annotations
import json
import regex as re
from typing import Optional, Dict, Any

_json_obj_re = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)  # PCRE-like recursive (but Python re doesn't support (?R))
# Python re doesn't support recursion; use a safer approach: find first {...} pair by scanning braces.

def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Find the first top-level JSON object by scanning braces balance.
    Returns the substring containing {...} or None.
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


def parse_single_enrichment(raw: str, expected_chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Parse an enrichment LLM response. Returns dict or None.

    Strategy:
    1) strip common code-block markers
    2) try json.loads(full cleaned text)
    3) fallback: extract first balanced {...} and parse it
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
            txt = txt[: -3]
        txt = txt.strip()

    # Try direct parse
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            if "chunk_id" not in obj:
                obj["chunk_id"] = expected_chunk_id
            return obj
    except Exception:
        pass

    # Find first balanced {...}
    snippet = _extract_first_json_object(txt)
    if snippet:
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                if "chunk_id" not in obj:
                    obj["chunk_id"] = expected_chunk_id
                return obj
        except Exception:
            pass

    return None
