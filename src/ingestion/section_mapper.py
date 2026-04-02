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
        toc_chunks = [ch for ch in chunks if ch.page <= 8]
        for ch in toc_chunks:
            for pattern, has_number in SECTION_PATTERNS:
                for match in pattern.finditer(ch.text or ""):
                    if has_number:
                        section_name = match.group(2).strip()
                    else:
                        section_name = match.group(1).strip()
                    if len(section_name) > 3:
                        pass

    def get_section(self, page: int) -> Optional[str]:
        return self._page_to_section.get(page)

    def apply_to_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Assign section field to each chunk based on its page number."""
        for ch in chunks:
            section = self.get_section(ch.page)
            if section:
                ch.section = section
                if ch.context and section.upper() not in (ch.context or "").upper():
                    ch.context = f"{section}. {ch.context}"
        return chunks
