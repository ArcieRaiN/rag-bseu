from __future__ import annotations

"""
Section mapper: extracts Table of Contents from PDF and maps pages to sections.

For statistical publications, the TOC is typically in the first few pages.
This module parses section headers and page ranges to assign each page a section name.
"""

import re
from typing import Dict, List, Optional, Tuple

from src.core.models import Chunk


# Known section patterns for Belstat publications
SECTION_PATTERNS = [
    (re.compile(r'(?:^|\n)\s*(\d+)\.\s+([А-ЯЁ][А-ЯЁа-яё\s,]+)', re.MULTILINE), True),
    (re.compile(r'(?:^|\n)\s*([А-ЯЁ][А-ЯЁ\s,]{5,})', re.MULTILINE), False),
]

# Hardcoded TOC for known publications (fallback)
KNOWN_TOCS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "Беларусь в цифрах, 2025.pdf": {
        "ТЕРРИТОРИЯ": (7, 12),
        "НАСЕЛЕНИЕ": (13, 16),
        "ТРУД": (16, 17),
        "ОБРАЗОВАНИЕ": (17, 19),
        "ЗДРАВООХРАНЕНИЕ": (19, 21),
        "УРОВЕНЬ ЖИЗНИ НАСЕЛЕНИЯ": (21, 26),
        "ПРАВОНАРУШЕНИЯ": (26, 27),
        "КУЛЬТУРА": (27, 30),
        "ФИЗИЧЕСКАЯ КУЛЬТУРА И СПОРТ": (30, 31),
        "НАЦИОНАЛЬНЫЕ СЧЕТА": (31, 34),
        "МАЛОЕ И СРЕДНЕЕ ПРЕДПРИНИМАТЕЛЬСТВО": (34, 36),
        "СЕЛЬСКОЕ ХОЗЯЙСТВО": (36, 40),
        "ПРОМЫШЛЕННОСТЬ": (40, 42),
        "СТРОИТЕЛЬСТВО": (42, 44),
        "ИНВЕСТИЦИИ": (44, 47),
        "ТРАНСПОРТ": (47, 49),
        "ТОРГОВЛЯ": (49, 51),
        "ТУРИЗМ": (51, 55),
        "ВНЕШНЯЯ ТОРГОВЛЯ": (55, 59),
        "ИНОСТРАННЫЕ ИНВЕСТИЦИИ": (59, 61),
        "ЦЕНЫ": (61, 63),
        "СВЯЗЬ И ИНФОРМАЦИОННЫЕ ТЕХНОЛОГИИ": (63, 65),
    },
    "Беларусь и Россия, 2024.pdf": {
        "ТЕРРИТОРИЯ И ПРИРОДНЫЕ РЕСУРСЫ": (9, 36),
        "НАСЕЛЕНИЕ": (37, 56),
        "УРОВЕНЬ ЖИЗНИ НАСЕЛЕНИЯ": (57, 66),
        "ОБРАЗОВАНИЕ": (67, 76),
        "ЗДРАВООХРАНЕНИЕ": (77, 80),
        "КУЛЬТУРА": (80, 84),
        "НАЦИОНАЛЬНЫЕ СЧЕТА": (85, 97),
        "ПРОМЫШЛЕННОЕ ПРОИЗВОДСТВО": (98, 107),
        "СЕЛЬСКОЕ ХОЗЯЙСТВО": (107, 120),
        "СТРОИТЕЛЬСТВО": (121, 124),
        "ТРАНСПОРТ": (125, 130),
        "ИНФОРМАЦИОННЫЕ И КОММУНИКАЦИОННЫЕ ТЕХНОЛОГИИ": (130, 137),
        "ТОРГОВЛЯ": (137, 144),
        "ВНЕШНЯЯ ТОРГОВЛЯ": (144, 152),
        "ФИНАНСЫ": (152, 156),
        "ЦЕНЫ И ТАРИФЫ": (156, 178),
        "ПОКАЗАТЕЛИ ДОСТИЖЕНИЯ ЦЕЛЕЙ УСТОЙЧИВОГО РАЗВИТИЯ": (178, 186),
    },
}


class SectionMapper:
    """
    Maps page numbers to section names based on TOC data.

    First tries known hardcoded TOC, then falls back to text-based extraction.
    """

    def __init__(self, pdf_name: str, chunks: Optional[List[Chunk]] = None):
        self._pdf_name = pdf_name
        self._page_to_section: Dict[int, str] = {}

        if pdf_name in KNOWN_TOCS:
            self._build_from_known_toc(KNOWN_TOCS[pdf_name])
        elif chunks:
            self._build_from_chunks(chunks)

    def _build_from_known_toc(self, toc: Dict[str, Tuple[int, int]]) -> None:
        for section_name, (start_page, end_page) in toc.items():
            for page in range(start_page, end_page + 1):
                self._page_to_section[page] = section_name

    def _build_from_chunks(self, chunks: List[Chunk]) -> None:
        """Try to extract section headers from early pages (TOC)."""
        toc_chunks = [ch for ch in chunks if ch.page <= 12]
        entries: List[Tuple[int, str]] = []
        for ch in toc_chunks:
            entries.extend(self._extract_toc_entries(ch.text or ""))

        if not entries:
            self._build_from_page_headers(chunks)
            return

        # Keep the first title seen for each page and map until the next TOC page.
        dedup: Dict[int, str] = {}
        for page, title in sorted(entries, key=lambda item: item[0]):
            if page not in dedup and 1 <= page <= max(ch.page for ch in chunks):
                dedup[page] = title

        pages = sorted(dedup)
        max_page = max(ch.page for ch in chunks)
        for idx, start_page in enumerate(pages):
            end_page = (pages[idx + 1] - 1) if idx + 1 < len(pages) else max_page
            title = dedup[start_page]
            for page in range(start_page, end_page + 1):
                self._page_to_section[page] = title

    @staticmethod
    def _extract_toc_entries(text: str) -> List[Tuple[int, str]]:
        entries: List[Tuple[int, str]] = []
        normalized = re.sub(r"\s+", " ", text.replace("\xa0", " "))
        # Split before numbered TOC items while keeping unnumbered headings usable.
        parts = re.split(r"(?=(?:\d+\.){1,3}\s+[А-ЯЁA-Z])", normalized)
        line_like = []
        for part in parts:
            line_like.extend(re.split(r"\s{2,}|Стр\.|Pg\.", part))

        for raw in line_like:
            clean = raw.strip(" .;\t")
            if len(clean) < 8:
                continue
            match = re.search(r"(?P<title>[А-ЯЁа-яёA-Za-z0-9№.,«»() /-]{6,140}?)\s+(?P<page>\d{1,3})(?:\s|$)", clean)
            if not match:
                continue
            title = re.sub(r"^\d+(?:\.\d+)*\.\s*", "", match.group("title")).strip(" .;")
            title = re.sub(r"^\d+\s+", "", title).strip(" .;")
            page = int(match.group("page"))
            if page <= 0 or len(title) < 4 or len(title) > 90 or re.match(r"^\d", title):
                continue
            if title.lower() in {"содержание", "contents", "продолжение", "continued"}:
                continue
            entries.append((page, title.upper() if title.isupper() else title))
        return entries

    def _build_from_page_headers(self, chunks: List[Chunk]) -> None:
        current: Optional[str] = None
        for ch in sorted(chunks, key=lambda item: item.page):
            header = self._first_header(ch.text or "")
            if header:
                current = header
            if current:
                self._page_to_section[ch.page] = current

    @staticmethod
    def _first_header(text: str) -> Optional[str]:
        for raw in text.splitlines()[:8]:
            clean = re.sub(r"\s+", " ", raw).strip(" .;")
            if not clean or len(clean) < 6 or len(clean) > 120:
                continue
            if re.search(r"\d", clean):
                continue
            if clean.lower().startswith(("www.", "стр", "page", "содержание", "contents")):
                continue
            letters = [ch for ch in clean if ch.isalpha()]
            if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.6:
                return clean
        return None

    def get_section(self, page: int) -> Optional[str]:
        return self._page_to_section.get(page)

    def apply_to_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Assign section field to each chunk based on its page number."""
        for ch in chunks:
            section = self.get_section(ch.page)
            if section:
                ch.section = section
                if ch.search_context and section.upper() not in (ch.search_context or "").upper():
                    ch.search_context = f"{section}. {ch.search_context}"
        return chunks
