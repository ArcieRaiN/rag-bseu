# src/prepare_db/chunk_maker.py
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from src.main.input_normalizer import normalize_text_lemmatized
from src.main.vectorizer import HashVectorizer


@dataclass
class BuildArtifacts:
    index_path: Path
    metadata_path: Path
    data_path: Path


class ChunkMaker:
    """
    ChunkMaker для ТАБЛИЦ: ищет блоки с числовыми строками, определяет заголовки таблиц
    и сохраняет семантические эмбеддинги по нормализованному заголовку (или тексту).
    """

    AUTHOR_RE = re.compile(r"(^[А-ЯЁ][\w\-]+(?:\s+[А-Я]\.){1,3}|^[A-Z]\.[A-Z]\.|редакци|©|издатель|издание|университет|институт)", re.IGNORECASE)
    NUMBER_RE = re.compile(r"[-+]?\d[\d\s\.,]*\d|^\d+$")
    MULTI_SPACES = re.compile(r"\s{2,}|\t")
    WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)

    def __init__(
        self,
        vectorizer: HashVectorizer,
        documents_dir: Path,
        # таблица считается, если подряд найдено min_rows строк с числами/табличным видом
        min_rows: int = 2,
        # минимальное число колонок (оценка по разделителю) чтобы не брать один столбик
        min_cols: int = 2,
        # lookback при поиске заголовка (строк перед таблицей)
        lookback_title_lines: int = 6,
        # минимальное количество слов в заголовке (если заголовок короткий — может быть мусором)
        min_title_words: int = 2,
        # максимальное количество слов в итоговом заголовке (урезаем длинные)
        max_title_words: int = 20,
    ):
        self.vectorizer = vectorizer
        self.documents_dir = Path(documents_dir)
        self.min_rows = min_rows
        self.min_cols = min_cols
        self.lookback_title_lines = lookback_title_lines
        self.min_title_words = min_title_words
        self.max_title_words = max_title_words

    # -----------------------------

    def build_tables_from_pdfs(self, output_dir: Path) -> BuildArtifacts:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables: List[Dict] = []
        embeddings: List[np.ndarray] = []
        table_id = 0

        for pdf_path in sorted(self.documents_dir.glob("*.pdf")):
            print(f"📘 Обработка: {pdf_path.name}")

            for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
                # соберём все текстовые блоки, сохраняя порядок
                page_blocks: List[str] = []
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        text = element.get_text()
                        if text and text.strip():
                            page_blocks.append(text)

                if not page_blocks:
                    continue

                lines = self._blocks_to_lines(page_blocks)
                # detect table regions (start_idx, end_idx)
                table_regions = self._detect_table_regions(lines)

                for start_idx, end_idx in table_regions:
                    table_lines = lines[start_idx:end_idx]
                    # estimate columns
                    est_cols = self._estimate_columns(table_lines)
                    if est_cols < self.min_cols:
                        # слишком мало колонок — возможно не таблица
                        continue

                    title = self._find_title(lines, start_idx, page_top_lines=page_blocks[:3])
                    # fallback: take first non-empty line in table if no title
                    if not title:
                        # try first non-empty row that looks like a header (letters)
                        for ln in table_lines[:3]:
                            if self.WORD_RE.search(ln) and len(ln.split()) >= self.min_title_words:
                                title = ln.strip()
                                break

                    raw_table_text = "\n".join(table_lines).strip()

                    # normalized title or fallback normalized text
                    normalized_title = normalize_text_lemmatized(title) if title else ""
                    if not normalized_title:
                        normalized_title = normalize_text_lemmatized(raw_table_text)

                    if not normalized_title:
                        # nothing to embed
                        continue

                    emb = self.vectorizer.embed(normalized_title)

                    tables.append(
                        {
                            "id": table_id,
                            "title": self._clean_title(title) if title else "",
                            "normalized": normalized_title,
                            "text": raw_table_text,
                            "source": pdf_path.name,
                            "page": page_num,
                            "rows": [self._split_row_to_cells(r) for r in table_lines],
                            "est_columns": est_cols,
                        }
                    )

                    embeddings.append(emb)
                    table_id += 1

        if not embeddings:
            embeddings_np = np.zeros((0, self.vectorizer.dimension), dtype=np.float32)
        else:
            embeddings_np = np.vstack([np.asarray(e, dtype=np.float32).reshape(1, -1) for e in embeddings]).astype(np.float32)

        index_path = output_dir / "index.npy"
        data_path = output_dir / "data.json"
        meta_path = output_dir / "metadata.json"

        np.save(index_path, embeddings_np)

        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(tables, f, ensure_ascii=False, indent=2)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vectorizer": type(self.vectorizer).__name__,
                    "dimension": self.vectorizer.dimension,
                    "tables": len(tables),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"✅ Найдено и сохранено таблиц: {len(tables)}")
        return BuildArtifacts(index_path=index_path, metadata_path=meta_path, data_path=data_path)

    # -----------------------------

    def _blocks_to_lines(self, blocks: List[str]) -> List[str]:
        """
        Разбиваем блоки на строки, чистим лишние пробелы.
        Сохраняем порядок.
        """
        lines: List[str] = []
        for block in blocks:
            for ln in block.splitlines():
                ln2 = ln.strip()
                if not ln2:
                    continue
                # нормализуем пробелы (табличные разделители часто — мн. пробелы)
                ln2 = self.MULTI_SPACES.sub("  ", ln2)  # двойной пробел для дальнейшего split
                lines.append(ln2)
        return lines

    def _detect_table_regions(self, lines: List[str]) -> List[tuple]:
        """
        Возвращает список (start_idx, end_idx) блоков, которые являются таблицами.
        Критерий: подряд min_rows линий, где есть числовые токены / таблицо-подобная структура.
        """
        regions = []
        n = len(lines)
        i = 0
        while i < n:
            # ищем начало потенц. таблицы
            if self._is_table_like_line(lines[i]):
                # растём вниз, пока идут table-like линии
                j = i + 1
                while j < n and self._is_table_like_line(lines[j]):
                    j += 1
                # длина блока
                block_len = j - i
                if block_len >= self.min_rows:
                    regions.append((i, j))
                    i = j
                    continue
            i += 1
        return regions

    def _is_table_like_line(self, line: str) -> bool:
        """
        Линия 'таблична', если:
         - содержит числа в нескольких ячейках, или
         - содержит повторяющиеся разделители (двойные пробелы), или
         - содержит много цифр относительно слов
        """
        if not line or len(line) < 3:
            return False

        # count numeric tokens
        numeric_tokens = re.findall(r"[-+]?\d[\d\.,]*", line)
        words = re.findall(r"\w+", line)
        numeric_count = len(numeric_tokens)
        total_tokens = max(len(words), 1)
        numeric_ratio = numeric_count / total_tokens

        # check separators (двойные пробелы or tabs) — признак колонок
        col_splits = re.split(r"\s{2,}|\t", line)
        est_cols = len([s for s in col_splits if s.strip() != ""])

        # heuristics
        if numeric_count >= 2:
            return True
        if numeric_ratio >= 0.35 and total_tokens >= 3:
            return True
        if est_cols >= self.min_cols:
            return True

        return False

    def _estimate_columns(self, lines: List[str]) -> int:
        max_cols = 0
        for ln in lines:
            parts = re.split(r"\s{2,}|\t", ln)
            non_empty = [p for p in parts if p.strip()]
            max_cols = max(max_cols, len(non_empty))
        return max_cols

    def _find_title(self, lines: List[str], table_start_idx: int, page_top_lines: List[str] = None) -> Optional[str]:
        """
        Ищем заголовок таблицы:
         1) Верхняя строка страницы (если UPPERCASE и не авторская/издательская)
         2) Несколько строк перед началом таблицы (lookback_title_lines),
            берём те, которые выглядят как заголовочные (не автор, не короткий мусор)
         3) В качестве последней попытки — любая ближайшая предыдущая строка с буквами
        """
        # 1) верх страницы
        if page_top_lines:
            for block in page_top_lines:
                first_line = block.splitlines()[0].strip() if block else ""
                if first_line and first_line.isupper() and self.WORD_RE.search(first_line):
                    if not self._is_author_or_publisher_line(first_line):
                        return first_line

        # 2) строки перед таблицей
        start_search = max(0, table_start_idx - self.lookback_title_lines)
        candidate_lines = []
        for i in range(start_search, table_start_idx):
            ln = lines[i].strip()
            if not ln:
                continue
            if self._is_author_or_publisher_line(ln):
                continue
            # убираем линии, состоящие только из нумерации или пунктов (например "1.", "2.")
            if re.fullmatch(r"^[\d\.\)\-]+$", ln):
                continue
            # clean numeric-only short lines
            words = ln.split()
            # ignore if mostly digits
            digits = sum(1 for w in words if self.NUMBER_RE.search(w))
            if digits >= len(words):
                continue
            # accept if has letters and not too short
            if self.WORD_RE.search(ln) and len(words) >= 1:
                candidate_lines.append(ln)

        if candidate_lines:
            # keep only meaningful ones, drop too short
            filtered = [c for c in candidate_lines if len(c.split()) >= self.min_title_words]
            if not filtered:
                filtered = candidate_lines[-1:]  # last resort: last candidate
            # join but limit words
            joined = " | ".join(filtered)
            joined = self._truncate_title(joined)
            return joined

        # 3) fallback: nearest previous non-empty non-author line
        for i in range(table_start_idx - 1, max(-1, table_start_idx - self.lookback_title_lines - 1), -1):
            ln = lines[i].strip()
            if ln and not self._is_author_or_publisher_line(ln) and self.WORD_RE.search(ln):
                return self._truncate_title(ln)

        return None

    def _is_author_or_publisher_line(self, line: str) -> bool:
        # авторские подписи: много запятых, инициалы или ключевые слова "редакция", "издание"
        if not line:
            return True
        # Если есть инициалы вида "И.О.Фамилия" или "Фамилия И.О."
        if re.search(r"\b[А-Я]\.[А-Я]\.", line):
            return True
        if re.search(r"\b[А-Я]\.[А-Я]\.[А-Я]\.", line):
            return True
        # слишком короткие и с заглавными буквами, часто фамилии/инициалы
        if len(line) < 4 and re.fullmatch(r"[А-ЯЁA-Z\.\-]+", line):
            return True
        # ключевые слова издательства/редакции
        if re.search(r"(редакци|издание|©|при участии|авторы|под ред|отв\.|издатель)", line, flags=re.IGNORECASE):
            return True
        # If many commas and capitals (author list)
        if line.count(",") >= 2 and re.search(r"[А-Я]\w+", line):
            return True
        # heuristics by regex
        if self.AUTHOR_RE.search(line):
            return True
        return False

    def _truncate_title(self, title: str) -> str:
        words = title.split()
        if len(words) <= self.max_title_words:
            return title.strip()
        return " ".join(words[: self.max_title_words]) + " …"

    def _clean_title(self, title: str) -> str:
        if not title:
            return ""
        t = title.strip()
        # remove leading numbering like "1.8." or "1)"
        t = re.sub(r"^[\d\.\)\-]+\s*", "", t)
        # replace multiple separators
        t = re.sub(r"\s*\|\s*", " | ", t)
        t = re.sub(r"\s+", " ", t)
        # remove trailing page footers like "НАЦИОНАЛЬНЫЕ СЧЕТА РЕСПУБЛИКИ БЕЛАРУСЬ, 2018 – 2023 27"
        t = re.sub(r"\s+\d{1,4}$", "", t)
        return t.strip()

    def _split_row_to_cells(self, row: str) -> List[str]:
        # split by 2+ spaces or tab, fallback to single space
        parts = re.split(r"\s{2,}|\t", row)
        parts = [p.strip() for p in parts if p.strip() != ""]
        if len(parts) <= 1:
            # fallback try splitting by single space but keep multi-digit groups together
            parts = [p.strip() for p in re.split(r"\s+", row) if p.strip() != ""]
        return parts
