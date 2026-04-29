from __future__ import annotations

"""
Deterministic metadata extraction for Belstat page chunks.

The extractor fills simple structural fields before/after LLM enrichment:
territories, years, units, table title and indicator candidates.  It is
intentionally conservative: rules should improve recall without inventing
facts that are absent from the page text.
"""

from dataclasses import dataclass, field
import re
from typing import Iterable, List, Optional

from src.core.models import Chunk


YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
SPLIT_YEAR_RE = re.compile(r"(?<!\d)(20)\s+(\d{2})(?!\d)")
TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

UNIT_PATTERNS = [
    r"тыс\.\s*(?:человек|руб(?:лей|\.?)?|тонн|единиц|ед\.|га|км2?)",
    r"млн\.?\s*(?:руб(?:лей|\.?)?|долл(?:аров)?|человек|тонн|единиц|ед\.?)",
    r"млрд\.?\s*(?:руб(?:лей|\.?)?|долл(?:аров)?|тонн)",
    r"(?:в\s+)?процентах(?:\s+к\s+[^;\n,.]{3,60})?",
    r"процент(?:ов|а)?",
    r"%",
    r"руб(?:лей|\.?)",
    r"человек(?:\s+на\s+1\s+км2)?",
    r"единиц(?:ы)?",
    r"ед\.",
    r"км2|км\s*2|кв\.\s*км",
    r"на\s+10\s*000\s+человек(?:\s+населения)?",
    r"на\s+1000\s+человек(?:\s+населения)?",
    r"на\s+душу\s+населения",
]
UNIT_RE = re.compile("|".join(f"(?:{p})" for p in UNIT_PATTERNS), re.IGNORECASE)

GEO_PATTERNS = {
    "Республика Беларусь": [
        r"\bреспублик[аи]\s+беларусь\b",
        r"\bбеларус[ьи]\b",
        r"\bbelarus\b",
    ],
    "Российская Федерация": [
        r"\bроссийск(?:ая|ой)\s+федерац(?:ия|ии)\b",
        r"\bросси[яи]\b",
        r"\brussia\b",
    ],
    "Брестская область": [r"\bбрестск(?:ая|ой)\s+област[ьи]\b"],
    "Витебская область": [r"\bвитебск(?:ая|ой)\s+област[ьи]\b"],
    "Гомельская область": [r"\bгомельск(?:ая|ой)\s+област[ьи]\b"],
    "Гродненская область": [r"\bгродненск(?:ая|ой)\s+област[ьи]\b"],
    "Минская область": [r"\bминск(?:ая|ой)\s+област[ьи]\b"],
    "Могилевская область": [r"\bмогил[её]вск(?:ая|ой)\s+област[ьи]\b"],
    "г. Минск": [r"\bг\.\s*минск\b", r"\bгород\s+минск\b"],
    "Брест": [r"\bг\.\s*брест\b", r"\bгород\s+брест\b"],
    "Витебск": [r"\bг\.\s*витебск\b", r"\bгород\s+витебск\b"],
    "Гомель": [r"\bг\.\s*гомель\b", r"\bгород\s+гомель\b"],
    "Гродно": [r"\bг\.\s*гродно\b", r"\bгород\s+гродно\b"],
    "Могилев": [r"\bг\.\s*могил[её]в\b", r"\bгород\s+могил[её]в\b"],
    "СНГ": [r"\bснг\b"],
    "ЕАЭС": [r"\bеаэс\b"],
    "Китай": [r"\bкита[йяе]\b"],
    "Казахстан": [r"\bказахстан[ае]?\b"],
    "Узбекистан": [r"\bузбекистан[ае]?\b"],
    "Украина": [r"\bукраин[аеы]\b"],
    "Польша": [r"\bпольш[аеуы]\b"],
    "Литва": [r"\bлитв[аеы]\b"],
    "Латвия": [r"\bлатви[яие]\b"],
}

SERVICE_LINE_RE = re.compile(
    r"^(?:содержание|contents|продолжение|continued|стр\.|pg\.|www\.belstat|"
    r"национальный статистический комитет|беларусь в цифрах|социальное положение|"
    r"статистический ежегодник|регионы республики беларусь)\b",
    re.IGNORECASE,
)


@dataclass
class RuleMetadata:
    geo: List[str] = field(default_factory=list)
    years: List[int] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    table_title: Optional[str] = None
    metric_candidates: List[str] = field(default_factory=list)
    search_context: str = ""
    quality: dict = field(default_factory=dict)


def _unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        item = _clean_space(str(item))
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def normalize_year_text(text: str) -> str:
    return SPLIT_YEAR_RE.sub(r"\1\2", text or "")


class RuleMetadataExtractor:
    """Rule-based extractor for page-level statistical chunks."""

    def extract(self, chunk: Chunk) -> RuleMetadata:
        text = normalize_year_text(chunk.text or "")
        lines = self._clean_lines(text)
        title = self.extract_table_title(lines, chunk.section)
        metrics = self.extract_metric_candidates(lines, title)
        geo = self.extract_geo(text)
        years = self.extract_years(text)
        units = self.extract_units(text)
        search_context = self.build_search_context(
            section=chunk.section,
            title=title,
            metrics=metrics,
            geo=geo,
            years=years,
            units=units,
        )
        quality = {
            "rule_geo": bool(geo),
            "rule_years": bool(years),
            "rule_units": bool(units),
            "rule_metrics": bool(metrics),
            "rule_title": bool(title),
        }
        return RuleMetadata(
            geo=geo,
            years=years,
            units=units,
            table_title=title,
            metric_candidates=metrics,
            search_context=search_context,
            quality=quality,
        )

    def apply_to_chunk(self, chunk: Chunk, *, overwrite: bool = False) -> Chunk:
        meta = self.extract(chunk)
        existing_geo = chunk.geo if isinstance(chunk.geo, list) else ([chunk.geo] if chunk.geo else [])
        existing_metrics = chunk.metrics if isinstance(chunk.metrics, list) else ([chunk.metrics] if chunk.metrics else [])
        existing_units = chunk.units if isinstance(chunk.units, list) else ([chunk.units] if chunk.units else [])

        if overwrite or not chunk.geo:
            chunk.geo = meta.geo or chunk.geo
        else:
            chunk.geo = _unique_keep_order([*existing_geo, *meta.geo])

        if overwrite or not chunk.years:
            chunk.years = meta.years or chunk.years
        else:
            chunk.years = sorted(set([*chunk.years, *meta.years]))

        if overwrite or not chunk.units:
            chunk.units = meta.units or chunk.units
        else:
            chunk.units = _unique_keep_order([*existing_units, *meta.units])

        if overwrite or not chunk.metrics:
            chunk.metrics = meta.metric_candidates or chunk.metrics
        else:
            chunk.metrics = _unique_keep_order([*existing_metrics, *meta.metric_candidates])

        chunk.search_context = meta.search_context or chunk.search_context
        extra = dict(chunk.extra or {})
        if meta.table_title:
            extra["table_title"] = meta.table_title
        chunk.extra = extra or chunk.extra

        quality = dict(chunk.metadata_quality or {})
        quality.update(meta.quality)
        quality["filled_fields"] = sum(
            bool(value)
            for value in (chunk.geo, chunk.years, chunk.units, chunk.metrics, chunk.section)
        )
        chunk.metadata_quality = quality
        return chunk

    @staticmethod
    def extract_years(text: str) -> List[int]:
        normalized = normalize_year_text(text)
        years = sorted({int(y) for y in YEAR_RE.findall(normalized) if 1900 <= int(y) <= 2035})
        return years

    @staticmethod
    def extract_geo(text: str) -> List[str]:
        normalized = _clean_space(text).lower()
        found: List[str] = []
        for canonical, patterns in GEO_PATTERNS.items():
            if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns):
                found.append(canonical)
        return _unique_keep_order(found)

    @staticmethod
    def extract_units(text: str) -> List[str]:
        normalized = normalize_year_text(text).lower()
        units = [m.group(0) for m in UNIT_RE.finditer(normalized)]
        return _unique_keep_order(units)[:8]

    @staticmethod
    def extract_table_title(lines: List[str], section: Optional[str] = None) -> Optional[str]:
        candidates: List[str] = []
        for line in lines[:60]:
            clean = _clean_space(line)
            if not clean or SERVICE_LINE_RE.search(clean.lower()):
                continue
            if re.fullmatch(r"[\d\s.,:;–—-]+", clean):
                continue
            numbered = re.match(r"^\d+(?:\.\d+)*\.\s+(.{4,120})$", clean)
            if numbered:
                candidates.append(numbered.group(1))
                continue
            has_cyrillic = any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in clean)
            digit_share = sum(ch.isdigit() for ch in clean) / max(len(clean), 1)
            if has_cyrillic and digit_share < 0.25 and 8 <= len(clean) <= 140:
                candidates.append(clean)
        if candidates:
            return candidates[0].strip(" .;:")
        return section

    @staticmethod
    def extract_metric_candidates(lines: List[str], title: Optional[str]) -> List[str]:
        candidates: List[str] = []
        if title:
            candidates.append(title)
        for line in lines[:90]:
            clean = _clean_space(re.sub(r"^\d+(?:\.\d+)*\.\s+", "", line))
            if not clean or SERVICE_LINE_RE.search(clean.lower()):
                continue
            if len(clean) < 4 or len(clean) > 110:
                continue
            if not any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in clean):
                continue
            digit_share = sum(ch.isdigit() for ch in clean) / max(len(clean), 1)
            if digit_share > 0.35:
                continue
            if UNIT_RE.search(clean.lower()):
                continue
            if clean.lower() in {"всего", "итого", "годы", "год"}:
                continue
            candidates.append(clean.strip(" .;:"))
        return _unique_keep_order(candidates)[:8]

    @staticmethod
    def build_search_context(
        *,
        section: Optional[str],
        title: Optional[str],
        metrics: List[str],
        geo: List[str],
        years: List[int],
        units: List[str],
    ) -> str:
        parts: List[str] = []
        if section:
            parts.append(f"раздел: {section}")
        if title:
            parts.append(f"таблица: {title}")
        if metrics:
            parts.append("показатели: " + ", ".join(metrics[:5]))
        if units:
            parts.append("единицы измерения: " + ", ".join(units[:4]))
        if geo:
            parts.append("география: " + ", ".join(geo[:8]))
        if years:
            years_sorted = sorted(set(years))
            if len(years_sorted) > 6:
                years_repr = f"{years_sorted[0]}-{years_sorted[-1]}"
            else:
                years_repr = ", ".join(str(y) for y in years_sorted)
            parts.append("годы: " + years_repr)
        return " | ".join(parts)[:700]

    @staticmethod
    def _clean_lines(text: str) -> List[str]:
        lines = []
        for raw in text.splitlines():
            clean = _clean_space(raw)
            if clean:
                lines.append(clean)
        if len(lines) <= 1:
            lines = [p.strip() for p in re.split(r"\s{2,}|\s\|\s", _clean_space(text)) if p.strip()]
        return lines
