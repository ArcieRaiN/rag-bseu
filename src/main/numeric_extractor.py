from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from natasha import Doc, MorphVocab, NewsEmbedding, NewsMorphTagger, NewsNERTagger, Segmenter

# -----------------------------
# Lazy-ish Natasha init (offline)
# -----------------------------

_segmenter = Segmenter()
_emb = NewsEmbedding()
_morph = NewsMorphTagger(_emb)
_ner = NewsNERTagger(_emb)
_vocab = MorphVocab()


Metric = str
Unit = str


@dataclass(frozen=True)
class FragmentMeta:
    source_file: str
    page: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    title: str = ""


# -----------------------------
# Normalization helpers
# -----------------------------

_SPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    return _SPACE_RE.sub(" ", (text or "").strip())


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-zА-Яа-яЁёІіЎў0-9]+", text.lower()) if t]


def _context_window(text: str, start: int, end: int, left_words: int = 5, right_words: int = 5) -> str:
    # take ~N words left/right around match (works well enough for RU)
    left = text[:start]
    right = text[end:]
    left_toks = _tokens(left)
    right_toks = _tokens(right)
    mid = text[start:end]
    return _clean_text(" ".join(left_toks[-left_words:] + [mid] + right_toks[:right_words]))


# -----------------------------
# Entity extraction (regex first, Natasha fallback)
# -----------------------------

_REGION_CANONICAL = {
    "брестская область": "Брестская область",
    "витебская область": "Витебская область",
    "гомельская область": "Гомельская область",
    "гродненская область": "Гродненская область",
    "минская область": "Минская область",
    "могилевская область": "Могилевская область",
}

_CITY_CANONICAL = {
    "г. минск": "г.Минск",
    "город минск": "г.Минск",
    "минск": "г.Минск",
    "брест": "г.Брест",
    "витебск": "г.Витебск",
    "гомель": "г.Гомель",
    "гродно": "г.Гродно",
    "могилев": "г.Могилев",
}

_ENTITY_REGEXES: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(брестская|витебская|гомельская|гродненская|минская|могилевская)\s+область\b", re.I), "region"),
    (re.compile(r"\bг\.\s*(минск|брест|витебск|гомель|гродно|могилев)\b", re.I), "city"),
    (re.compile(r"\bгород\s+(минск|брест|витебск|гомель|гродно|могилев)\b", re.I), "city"),
    (re.compile(r"\bреспублика\s+беларусь\b", re.I), "country"),
    (re.compile(r"\bбеларусь\b", re.I), "country"),
]


def _canonicalize_entity(raw: str, kind: str) -> Optional[str]:
    s = _clean_text(raw).lower()
    if kind == "region":
        # ensure has "область"
        if not s.endswith("область"):
            s = f"{s} область"
        return _REGION_CANONICAL.get(s, raw.strip())
    if kind == "city":
        # normalize to "г.X"
        if s.startswith("г."):
            s = _clean_text(s.replace("г.", "г.")).lower()
        return _CITY_CANONICAL.get(s, raw.strip())
    if kind == "country":
        return "Беларусь"
    return raw.strip() or None


def extract_entity(text: str, title: str = "") -> Tuple[Optional[str], float]:
    hay = f"{title}\n{text}"
    # 1) regex
    for rx, kind in _ENTITY_REGEXES:
        m = rx.search(hay)
        if m:
            entity = _canonicalize_entity(m.group(0), kind)
            return entity, 0.95

    # 2) Natasha NER (LOC)
    doc = Doc(hay)
    doc.segment(_segmenter)
    doc.tag_ner(_ner)
    best = None
    for span in doc.spans:
        if span.type != "LOC":
            continue
        span.normalize(_vocab)
        val = span.normal or span.text
        if not val:
            continue
        best = val
        break
    if best:
        # Natasha may return just "Минск" etc.
        entity = _CITY_CANONICAL.get(best.lower(), best)
        return entity, 0.7

    return None, 0.3


# -----------------------------
# Number parsing + unit detection
# -----------------------------

_NUM_RE = re.compile(
    r"(?P<num>\d{1,3}(?:[ \u00A0]?\d{3})*(?:[.,]\d+)?)\s*(?P<suf>тыс\.?|млн\.?|млрд\.?|%)?",
    re.I,
)

_UNIT_HINTS = {
    "percent": [r"%", r"\bпроцент", r"\bпроцентов\b", r"\bиндекс\b"],
    "persons": [r"\bчеловек\b", r"\bчел\.?\b", r"\bжител"],
    "km2": [r"\bкм2\b", r"\bкм²\b", r"\bкв\.?\s*км\b"],
    "hectares": [r"\bга\b", r"\bгектар"],
    "tonnes": [r"\bтонн\b", r"\bтонна\b", r"\bт\b"],
}


def parse_localized_number(number_text: str) -> Optional[float]:
    s = number_text.replace("\u00A0", " ").strip()
    s = s.replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _suffix_multiplier(suf: str) -> Tuple[float, str]:
    if not suf:
        return 1.0, ""
    suf = suf.lower().strip()
    if suf.startswith("тыс"):
        return 1000.0, "thousand"
    if suf.startswith("млн"):
        return 1_000_000.0, "million"
    if suf.startswith("млрд"):
        return 1_000_000_000.0, "billion"
    if suf == "%":
        return 1.0, "percent"
    return 1.0, ""


def detect_unit(context: str, suffix: str, title: str = "") -> Unit:
    ctx = f"{title} {context}".lower()

    # explicit density: "... человек на 1 км2"
    if re.search(r"\bчел", ctx) and re.search(r"\bна\s+1\s*(км2|км²|кв\.?\s*км)\b", ctx):
        return "persons_per_km2"

    mult, mult_kind = _suffix_multiplier(suffix)
    if mult_kind == "percent" or any(re.search(p, ctx) for p in _UNIT_HINTS["percent"]):
        return "percent"

    if any(re.search(p, ctx) for p in _UNIT_HINTS["km2"]):
        return "km2"
    if any(re.search(p, ctx) for p in _UNIT_HINTS["hectares"]):
        return "hectares"
    if any(re.search(p, ctx) for p in _UNIT_HINTS["tonnes"]):
        return "tonnes"

    # persons with масштаб (also allow title-driven население)
    is_people_context = any(re.search(p, ctx) for p in _UNIT_HINTS["persons"]) or ("населен" in ctx)
    if is_people_context:
        if mult_kind == "thousand":
            return "thousand_persons"
        if mult_kind == "million":
            return "million_persons"
        return "persons"

    # count fallback
    return "count"


# -----------------------------
# Metric classification (rule-based)
# -----------------------------

_METRIC_RULES: List[Tuple[Metric, List[str], float]] = [
    ("population_density", [r"\bплотност", r"persons_per_km2", r"\bна\s+1\s*км"], 0.95),
    ("population", [r"\bнаселен", r"\bпрожив", r"\bчеловек\b", r"\bжител"], 0.9),
    ("area", [r"\bплощад", r"\bкм2\b", r"\bкм²\b", r"\bга\b", r"\bгектар"], 0.9),
    ("districts_count", [r"\bрайон", r"\bрайонов\b"], 0.9),
    ("cities_count", [r"\bгород", r"\bгородов\b"], 0.9),
    ("institutions_count", [r"\bучрежден", r"\bорганизац"], 0.75),
    ("gdp", [r"\bввп\b", r"\bврп\b", r"\bвалов", r"\bпродукт"], 0.8),
    ("production_volume", [r"\bпроизводств", r"\bдобыч", r"\bурожай", r"\bсбор"], 0.8),
    ("index_percent", [r"\bиндекс\b", r"%", r"\bпроцент"], 0.8),
]


def classify_metric(context: str, unit: Unit, title: str = "") -> Tuple[Optional[Metric], float]:
    ctx = f"{title} {context}".lower()
    # unit-driven
    if unit == "persons_per_km2":
        return "population_density", 0.95
    if unit in ("percent",) and "индекс" in ctx:
        return "index_percent", 0.9

    best: Tuple[Optional[str], float] = (None, 0.0)
    for metric, pats, score in _METRIC_RULES:
        hit = False
        for p in pats:
            if p == unit:
                hit = True
                continue
            if re.search(p, ctx):
                hit = True
        if hit and score > best[1]:
            best = (metric, score)
    return best


# -----------------------------
# Extraction core
# -----------------------------

def _context_similarity(query: Optional[str], text: str, title: str) -> float:
    if not query:
        return 1.0
    q = set(_tokens(query))
    if not q:
        return 0.7
    t = set(_tokens(f"{title} {text}"))
    inter = len(q & t)
    return max(0.1, min(1.0, inter / max(1, len(q))))


def extract_numeric_indicators(
    fragment_text: str,
    meta: FragmentMeta,
    *,
    query: Optional[str] = None,
    strict: bool = True,
    min_confidence: float = 0.6,
) -> List[Dict[str, Any]]:
    """
    Input:
      - text fragment + metadata {file, page, bbox(optional), title}
    Output:
      - list of dicts with entity/metric/unit/normalized_value + provenance + confidence
    """
    text = fragment_text or ""
    title = meta.title or ""
    raw = _clean_text(text)
    if not raw:
        return []

    entity, ner_score = extract_entity(raw, title=title)
    ctx_sim = _context_similarity(query, raw, title)

    out: List[Dict[str, Any]] = []
    for m in _NUM_RE.finditer(raw):
        num_text = m.group("num")
        suf = m.group("suf") or ""
        if not num_text:
            continue

        # local context (3-5 words) around number
        ctx = _context_window(raw, m.start(), m.end(), left_words=5, right_words=5)

        unit = detect_unit(ctx, suf, title=title)
        metric, rule_score = classify_metric(ctx, unit, title=title)
        if not metric:
            if strict:
                continue
            metric = "unknown"
            rule_score = 0.25

        base = parse_localized_number(num_text)
        if base is None:
            continue
        mult, mult_kind = _suffix_multiplier(suf)

        # normalized_value always in base unit (persons/km2/%/km2/tonnes/count)
        normalized_value = base * mult

        # unit names: keep scale in unit (thousand_persons) but normalized_value in persons
        if unit == "thousand_persons":
            # normalized already persons; keep unit as specified
            pass
        elif unit == "million_persons":
            pass
        elif unit == "persons":
            pass
        elif unit == "count":
            normalized_value = float(int(round(base)))  # counts
        elif unit == "percent":
            normalized_value = float(base)  # percent points
        elif unit == "persons_per_km2":
            normalized_value = float(base)
        elif unit in ("km2", "hectares", "tonnes"):
            normalized_value = float(base * mult)  # e.g. тыс. тонн

        confidence = float(max(0.0, min(1.0, rule_score * ner_score * ctx_sim)))

        # strict filters
        if strict:
            if confidence < min_confidence:
                continue
            # require entity for geo metrics (population/area/density/etc.)
            if metric in ("population", "population_density", "area") and not entity:
                continue

        out.append(
            {
                "entity": entity,
                "metric": metric,
                "raw_text": raw,
                "number_text": f"{num_text}{(' ' + suf) if suf else ''}".strip(),
                "unit": unit,
                "normalized_value": float(normalized_value),
                "source_file": meta.source_file,
                "page": meta.page,
                "bbox": meta.bbox,
                "title": title,
                "context": ctx,
                "confidence": confidence,
            }
        )

    return out


def aggregate_homogeneous(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate ONLY homogeneous values: same entity + metric + unit.
    Keeps main source (highest confidence).
    """
    buckets: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for it in items:
        key = (it.get("entity"), it.get("metric"), it.get("unit"))
        buckets.setdefault(key, []).append(it)

    out: List[Dict[str, Any]] = []
    for key, vals in buckets.items():
        if len(vals) == 1:
            out.append(vals[0])
            continue
        # sum only for count-like and additive metrics; keep conservative: sum for *_count and population
        metric = key[1]
        additive = metric in ("districts_count", "cities_count", "institutions_count", "population", "production_volume")
        if not additive:
            # keep the best one
            best = max(vals, key=lambda x: x.get("confidence", 0.0))
            out.append(best)
            continue
        summed = sum(float(v.get("normalized_value", 0.0)) for v in vals)
        best = max(vals, key=lambda x: x.get("confidence", 0.0))
        merged = dict(best)
        merged["normalized_value"] = float(summed)
        merged["confidence"] = float(max(v.get("confidence", 0.0) for v in vals))
        merged["aggregated_from"] = [
            {"source_file": v.get("source_file"), "page": v.get("page"), "confidence": v.get("confidence")}
            for v in vals
        ]
        out.append(merged)
    return out

