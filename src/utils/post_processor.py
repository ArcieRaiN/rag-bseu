from __future__ import annotations

"""
Модуль для post-processing обогащенных данных.

Исправляет типичные ошибки LLM:
- Галлюцинации годов (удаляет годы из методологии/определений)
- Неправильные metrics (фильтрует определения, заголовки)
- Нормализация данных
"""

from typing import List, Optional
import re

from src.core.models import Chunk


class EnrichmentPostProcessor:
    """
    Post-processor для исправления типичных ошибок LLM-обогащения.
    """

    # Паттерны для определения методологии/определений (не должны иметь years)
    METHODOLOGY_PATTERNS = [
        re.compile(r"методолог", re.IGNORECASE),
        re.compile(r"определение", re.IGNORECASE),
        re.compile(r"понятие", re.IGNORECASE),
        re.compile(r"термин", re.IGNORECASE),
        re.compile(r"содержание", re.IGNORECASE),
        re.compile(r"оглавление", re.IGNORECASE),
        re.compile(r"предисловие", re.IGNORECASE),
    ]

    # Паттерны для определения "не метрик"
    NON_METRIC_PATTERNS = [
        re.compile(r"^коммерческие организации$", re.IGNORECASE),
        re.compile(r"^цифровые технологии$", re.IGNORECASE),
        re.compile(r"^организации$", re.IGNORECASE),
        re.compile(r"^технологии$", re.IGNORECASE),
        re.compile(r"^определение", re.IGNORECASE),
        re.compile(r"^понятие", re.IGNORECASE),
        re.compile(r"../prepare_db", re.IGNORECASE),
        re.compile(r"^[А-ЯЁ\s]{20,}", re.IGNORECASE),  # Длинные заголовки
    ]

    def process_chunk(self, chunk: Chunk) -> Chunk:
        """
        Обрабатывает один чанк, исправляя типичные ошибки.
        
        Args:
            chunk: Чанк для обработки
            
        Returns:
            Обработанный чанк
        """
        # Проверка на методологию/определения - удаляем годы
        if self._is_methodology_or_definition(chunk):
            chunk.years = None
        
        # Фильтрация неправильных metrics
        if chunk.metrics:
            chunk.metrics = self._filter_valid_metrics(chunk.metrics, chunk.text)

        # Довыделение metrics из сырого текста (таблицы/OCR часто содержат нужную формулировку,
        # даже если LLM вернул слишком общий ответ вроде "обучение").
        inferred_metrics = self._infer_metrics_from_text(chunk.text or "")
        if inferred_metrics:
            merged: List[str] = []
            # inferred ставим первыми (они обычно ближе к запросу пользователя)
            for m in inferred_metrics:
                if m not in merged:
                    merged.append(m)
            for m in (chunk.metrics or []):
                if m not in merged:
                    merged.append(m)
            chunk.metrics = self._filter_valid_metrics(merged, chunk.text)
        
        # Проверка на галлюцинации годов
        if chunk.years:
            chunk.years = self._filter_valid_years(chunk.years, chunk.text)
        
        # Автоматическое определение time_granularity на основе years
        if chunk.years and not chunk.time_granularity:
            if len(chunk.years) > 1:
                # Если несколько годов - это годовая гранулярность
                chunk.time_granularity = "year"
            elif len(chunk.years) == 1:
                # Если один год - тоже годовая гранулярность
                chunk.time_granularity = "year"

        # Усиление семантики: добавляем краткое описание показателей в context.
        # Это помогает привести "мир чанков" ближе к "миру запросов".
        if chunk.metrics or chunk.geo or chunk.years:
            parts = []
            if chunk.metrics:
                metrics_str = ", ".join(chunk.metrics[:3])
                parts.append(f"показатели: {metrics_str}")
            if chunk.geo:
                parts.append(f"география: {chunk.geo}")
            if chunk.years:
                years_sorted = sorted(chunk.years)
                if len(years_sorted) > 1:
                    years_repr = f"{years_sorted[0]}-{years_sorted[-1]}"
                else:
                    years_repr = str(years_sorted[0])
                parts.append(f"годы: {years_repr}")

            if parts:
                semantic_summary = "; ".join(parts)
                if chunk.context:
                    chunk.context = f"{chunk.context} | {semantic_summary}"
                else:
                    chunk.context = semantic_summary
        
        # Очистка context от мусора (ALL CAPS заголовки, дубли RU/EN)
        if chunk.context:
            original_context = chunk.context
            cleaned_context = self._clean_context(chunk.context)
            # Гарантируем, что context не станет пустым:
            # если после очистки ничего не осталось, откатываемся к исходному значению.
            chunk.context = cleaned_context or original_context
        
        return chunk

    @staticmethod
    def _infer_metrics_from_text(text: str) -> List[str]:
        """
        Эвристически извлекает названия показателей из сырого текста чанка.

        Цель: поднять recall retrieval (BM25/metadata) для табличных чанков, где LLM
        часто возвращает слишком общий metrics.
        """
        if not text:
            return []

        t = text.lower()
        out: List[str] = []

        def _add(metric: str) -> None:
            m = metric.strip().lower()
            if not m:
                return
            # ограничиваем длину, чтобы не протаскивать целые определения
            if len(m) > 60:
                m = m[:60].rstrip()
            if m and m not in out:
                out.append(m)

        # Частые целевые метрики (в т.ч. кейс "численность студентов")
        if "численность студентов" in t or "число студентов" in t:
            _add("численность студентов")

        # Обучение/образование: охват, квалификация, ПК и т.п.
        if "охват детей дошкольным образованием" in t or "охват дошкольным образованием" in t:
            _add("охват детей дошкольным образованием")
        if "персональные компьютеры" in t and "учебн" in t:
            _add("персональные компьютеры в учебных целях")
        if "доля учителей" in t and "квалификац" in t:
            _add("доля учителей с минимальной требуемой квалификацией")

        # Обобщённый паттерн для табличных заголовков показателей:
        # "Численность ...", "Доля ...", "Уровень ...", "Индекс ..."
        # Берём короткую фразу до скобки/перевода строки.
        header_re = re.compile(
            r"(?:(?:^)|(?:\n))\s*(численность|число|доля|уровень|индекс)\s+([^\n(]{3,80})",
            re.IGNORECASE,
        )
        for m in header_re.finditer(text):
            head = (m.group(1) or "").strip().lower()
            tail = (m.group(2) or "").strip().lower()
            # убираем мусорные хвосты
            tail = re.sub(r"\s{2,}", " ", tail)
            candidate = f"{head} {tail}".strip()
            # слишком длинные/похожее на методологию не берём
            if len(candidate) <= 60 and "показател" not in candidate:
                _add(candidate)

        # Держим максимум 5 — дальше всё равно обрежется валидатором.
        return out[:5]
    
    def _clean_context(self, context: str) -> str:
        """
        Очищает context от мусора: ALL CAPS заголовки, дубли RU/EN.
        
        Args:
            context: Исходный context
            
        Returns:
            Очищенный context
        """
        if not context:
            return context
        
        lines = context.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Пропускаем ALL CAPS строки (вероятно, заголовки)
            if line.isupper() and len(line) > 10:
                # Но если это короткий заголовок или содержит важную информацию - оставляем
                if len(line.split()) <= 3:
                    continue
            
            # Пропускаем дубликаты RU/EN заголовков
            # (если строка содержит только заглавные буквы и короткая)
            if line.isupper() and len(line) < 50:
                # Проверяем, не является ли это дубликатом предыдущей строки
                if cleaned_lines and cleaned_lines[-1].upper() == line:
                    continue
            
            cleaned_lines.append(line)
        
        cleaned = ' '.join(cleaned_lines)
        
        # Ограничиваем длину
        if len(cleaned) > 256:
            cleaned = cleaned[:256]
        
        return cleaned.strip()

    def process_batch(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Обрабатывает батч чанков.
        
        Args:
            chunks: Список чанков для обработки
            
        Returns:
            Список обработанных чанков
        """
        return [self.process_chunk(chunk) for chunk in chunks]

    def _is_methodology_or_definition(self, chunk: Chunk) -> bool:
        """
        Определяет, является ли чанк методологией или определением.
        
        Args:
            chunk: Чанк для проверки
            
        Returns:
            True если это методология/определение
        """
        text = (chunk.text or "").lower()
        context = (chunk.context or "").lower()
        
        combined = f"{text} {context}"
        
        for pattern in self.METHODOLOGY_PATTERNS:
            if pattern.search(combined[:500]):  # Проверяем первые 500 символов
                return True
        
        return False

    def _filter_valid_metrics(
        self,
        metrics: List[str],
        chunk_text: Optional[str] = None,
    ) -> Optional[List[str]]:
        """
        Фильтрует валидные metrics, удаляя определения и заголовки.
        
        Args:
            metrics: Список метрик для фильтрации
            chunk_text: Текст чанка для контекста
            
        Returns:
            Отфильтрованный список метрик или None
        """
        if not metrics:
            return None
        
        valid_metrics = []
        text_lower = (chunk_text or "").lower()
        
        for metric in metrics:
            metric_lower = metric.lower().strip()
            
            # Пропускаем слишком длинные "метрики"
            if len(metric) > 50:
                continue
            
            # Пропускаем метрики, которые являются паттернами "не метрик"
            is_non_metric = False
            for pattern in self.NON_METRIC_PATTERNS:
                if pattern.search(metric_lower):
                    is_non_metric = True
                    break
            
            if is_non_metric:
                continue
            
            # Проверяем, что метрика содержит кириллицу
            has_cyrillic = any('\u0400' <= char <= '\u04FF' for char in metric_lower)
            if not has_cyrillic:
                continue
            
            # Проверяем, что метрика не является просто заголовком
            # (заголовки обычно в верхнем регистре и короткие)
            if metric.isupper() and len(metric.split()) <= 3:
                # Это может быть заголовок, но если он встречается в тексте как метрика - оставляем
                if metric_lower not in text_lower:
                    continue
            
            valid_metrics.append(metric_lower)
        
        # Ограничиваем максимум 5 метрик
        return valid_metrics[:5] if valid_metrics else None

    def _filter_valid_years(
        self,
        years: List[int],
        chunk_text: Optional[str] = None,
    ) -> Optional[List[int]]:
        """
        Фильтрует валидные годы, удаляя галлюцинации.
        
        Args:
            years: Список годов для фильтрации
            chunk_text: Текст чанка для проверки наличия годов
            
        Returns:
            Отфильтрованный список годов или None
        """
        if not years:
            return None
        
        if not chunk_text:
            # Если нет текста, оставляем годы как есть (но ограничиваем)
            return years[:5] if years else None
        
        # Ищем годы в тексте
        text = chunk_text
        valid_years = []
        
        for year in years:
            # Проверяем, упоминается ли год в тексте
            year_str = str(year)
            if year_str in text:
                valid_years.append(year)
            # Также проверяем диапазоны (например, "2020-2024")
            elif f"{year}-" in text or f"-{year}" in text:
                valid_years.append(year)
        
        # Если не нашли ни одного года в тексте, но есть годы в списке,
        # это может быть галлюцинация - возвращаем None
        if not valid_years and years:
            # Но если это может быть год публикации документа, оставляем первый год
            # (это эвристика - можно улучшить)
            return None
        
        # Ограничиваем максимум 9 годов (для поддержки временных рядов)
        return valid_years[:9] if valid_years else None
