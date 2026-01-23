import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from src.main.numeric_extractor import FragmentMeta, aggregate_homogeneous, extract_numeric_indicators
from src.main.retriever import SemanticRetriever
from src.main.vectorizer import HashVectorizer


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG CLI: extract numeric indicators (offline).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--strict", action="store_true", help="Strict filtering (default).")
    mode.add_argument("--relaxed", action="store_true", help="Relaxed mode (keep more extractions).")
    p.add_argument("--top-k", type=int, default=10, help="Retriever top_k.")
    p.add_argument("--hybrid-weight", type=float, default=0.3, help="Retriever hybrid weight.")
    p.add_argument("--aggregate", action="store_true", help="Aggregate only homogeneous (metric+unit+entity).")
    p.add_argument("--min-confidence", type=float, default=0.6, help="Min confidence in strict mode.")
    p.add_argument("--query", type=str, default="", help="Single query (if provided).")
    p.add_argument("--log-raw", type=str, default="", help="Write raw hits JSONL for human review.")
    return p


def _extract_from_hits(
    query: str,
    hits: List[Dict[str, Any]],
    *,
    strict: bool,
    aggregate: bool,
    min_confidence: float,
) -> List[Dict[str, Any]]:
    extracted: List[Dict[str, Any]] = []
    for h in hits:
        meta = FragmentMeta(
            source_file=str(h.get("source", "")),
            page=int(h.get("page", 0) or 0),
            bbox=h.get("bbox"),
            title=str(h.get("title", "")),
        )
        extracted.extend(
            extract_numeric_indicators(
                fragment_text=str(h.get("text", "")),
                meta=meta,
                query=query,
                strict=strict,
                min_confidence=min_confidence,
            )
        )
    if aggregate:
        extracted = aggregate_homogeneous(extracted)
    extracted.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)
    return extracted


def main() -> None:
    args = _build_argparser().parse_args()
    strict = True if not args.relaxed else False
    if args.strict:
        strict = True

    base_dir = Path(__file__).resolve().parents[1]
    vector_store_dir = base_dir / "prepare_db" / "vector_store"

    vectorizer = HashVectorizer(dimension=256)
    retriever = SemanticRetriever(vectorizer=vectorizer, data_path=vector_store_dir / "data.json")

    if args.query.strip():
        queries = [args.query.strip()]
    else:
        # small default for local smoke runs
        queries = [
            "Численность населения по областям Беларуси",
            "Производство молока",
            "Число учреждений здравоохранения",
        ]

    raw_fp = None
    if args.log_raw:
        os.makedirs(str(Path(args.log_raw).parent), exist_ok=True)
        raw_fp = open(args.log_raw, "w", encoding="utf-8")

    try:
        for q in queries:
            hits = retriever.search(q, top_k=args.top_k, hybrid_weight=args.hybrid_weight)
            if raw_fp:
                raw_fp.write(json.dumps({"query": q, "hits": hits}, ensure_ascii=False) + "\n")
            results = _extract_from_hits(
                q,
                hits,
                strict=strict,
                aggregate=bool(args.aggregate),
                min_confidence=float(args.min_confidence),
            )
            print(json.dumps({"query": q, "results": results}, ensure_ascii=False, indent=2))
    finally:
        if raw_fp:
            raw_fp.close()


if __name__ == "__main__":
    main()

import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from src.main.retriever import SemanticRetriever
from src.main.vectorizer import HashVectorizer

# Флаги запуска
predefined_queries_flag = False   # запустить в начале предопределённые запросы
user_queries_flag = True        # дать пользователю возможность вводить запросы

# Порог релевантности (минимальный score для включения в ответ)
MIN_SCORE_THRESHOLD = 0.1


def _clean_text(text: str, max_length: int = 800) -> str:
    """
    Очищает и обрезает текст для лучшей читаемости.
    
    Args:
        text: Исходный текст
        max_length: Максимальная длина текста
    
    Returns:
        Очищенный текст
    """
    if not text:
        return ""
    
    # Убираем лишние пробелы и переносы строк
    text = " ".join(text.split())
    
    # Если текст слишком длинный, обрезаем по предложениям
    if len(text) > max_length:
        # Пытаемся обрезать по последней точке или переносу строки
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        cut_point = max(last_period, last_newline)
        if cut_point > max_length * 0.7:  # Если нашли разумное место для обрезки
            text = truncated[:cut_point + 1] + "..."
        else:
            text = truncated + "..."
    
    return text


# Словари для нормализации единиц измерения
UNIT_MULTIPLIERS = {
    'тыс.': 1000, 'тысяч': 1000, 'тыс': 1000,
    'млн.': 1000000, 'миллион': 1000000, 'млн': 1000000,
    'млрд.': 1000000000, 'миллиард': 1000000000, 'млрд': 1000000000,
    'га': 1, 'гектар': 1, 'гектаров': 1,
    'км²': 1, 'кв.км': 1, 'км2': 1,
    'м²': 1, 'кв.м': 1, 'м2': 1,
    'тонн': 1, 'т': 1, 'тонна': 1,
    'кг': 1, 'килограмм': 1,
    'руб.': 1, 'рублей': 1, 'руб': 1,
    'долл.': 1, 'долларов': 1, 'usd': 1,
    'человек': 1, 'чел.': 1, 'чел': 1,
    'процент': 1, '%': 1, 'процентов': 1,
}

# Ключевые слова для определения типа данных
STATISTICAL_CONTEXT_KEYWORDS = {
    'население': ['населен', 'человек', 'жител', 'чел'],
    'производство': ['производств', 'выпуск', 'выработк', 'добыч'],
    'площадь': ['площад', 'га', 'гектар', 'км²', 'кв.км'],
    'стоимость': ['стоимость', 'цена', 'руб', 'долл', 'стоим'],
    'заработная': ['заработн', 'зарплат', 'доход', 'оплат'],
    'экспорт': ['экспорт', 'вывоз'],
    'импорт': ['импорт', 'ввоз'],
    'ввп': ['ввп', 'валовой', 'внутренний', 'продукт'],
    'инвестиция': ['инвестиц', 'вложен', 'капитал'],
    'бюджет': ['бюджет', 'доход', 'расход'],
    'урожайность': ['урожайност', 'сбор', 'валов'],
    'поголовье': ['поголовь', 'скот', 'животн'],
}

# Географические объекты
GEOGRAPHIC_ENTITIES = {
    'области': ['брестск', 'витебск', 'гомельск', 'гродненск', 'минск', 'минская', 'могилевск'],
    'города': ['минск', 'брест', 'витебск', 'гомель', 'гродно', 'могилев'],
    'регионы': ['беларусь', 'республика', 'рб', 'белорус'],
}


def _normalize_unit(value_str: str, unit: str) -> Tuple[float, str]:
    """
    Нормализует число с единицей измерения.
    Возвращает (нормализованное_число, нормализованная_единица).
    """
    # Извлекаем число
    number_match = re.search(r'[\d\s.,]+', value_str)
    if not number_match:
        return None, None
    
    number_str = number_match.group().replace(' ', '').replace(',', '.')
    try:
        number = float(number_str)
    except ValueError:
        return None, None
    
    # Определяем множитель для единицы
    unit_lower = unit.lower().strip('.,;:')
    multiplier = UNIT_MULTIPLIERS.get(unit_lower, 1)
    
    # Нормализуем число
    normalized_value = number * multiplier
    
    # Определяем нормализованную единицу
    if unit_lower in ['тыс.', 'тысяч', 'тыс']:
        normalized_unit = 'тыс.'
    elif unit_lower in ['млн.', 'миллион', 'млн']:
        normalized_unit = 'млн.'
    elif unit_lower in ['млрд.', 'миллиард', 'млрд']:
        normalized_unit = 'млрд.'
    else:
        normalized_unit = unit
    
    return normalized_value, normalized_unit


def _extract_entity_from_context(context: str, query: str) -> str:
    """
    Извлекает географический объект или другую сущность из контекста.
    """
    context_lower = context.lower()
    query_lower = query.lower()
    
    # Ищем области
    for region_type, keywords in GEOGRAPHIC_ENTITIES.items():
        for keyword in keywords:
            if keyword in context_lower or keyword in query_lower:
                return keyword.capitalize()
    
    # Ищем упоминания "Беларусь", "Республика Беларусь"
    if 'беларусь' in context_lower or 'беларусь' in query_lower:
        return 'Беларусь'
    
    return None


def _is_relevant_number(context: str, query: str, number_type: str = None) -> bool:
    """
    Проверяет, релевантно ли число запросу на основе контекста.
    """
    context_lower = context.lower()
    query_lower = query.lower()
    
    # Если указан тип числа, проверяем соответствие
    if number_type:
        keywords = STATISTICAL_CONTEXT_KEYWORDS.get(number_type, [])
        if keywords:
            if not any(kw in context_lower for kw in keywords):
                return False
    
    # Проверяем наличие ключевых слов из запроса в контексте
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    context_words = set(re.findall(r'\b\w+\b', context_lower))
    
    # Должно быть хотя бы одно совпадение значимых слов (длина > 3)
    significant_matches = query_words & context_words
    significant_matches = {w for w in significant_matches if len(w) > 3}
    
    if not significant_matches:
        return False
    
    # Исключаем нерелевантные контексты
    noise_patterns = [
        r'примечан', r'сноск', r'методик', r'расчет',
        r'источник данн', r'использован', r'приведен',
        r'редакционн', r'содержан', r'оглавлен'
    ]
    for pattern in noise_patterns:
        if re.search(pattern, context_lower):
            return False
    
    return True


def _extract_numbers_from_text(
    text: str, 
    query: str, 
    title: str = "",
    source: str = "",
    page: int = 0
) -> List[Dict]:
    """
    Улучшенное извлечение чисел из текста с контекстом, единицами и привязкой к объектам.
    
    Returns:
        Список словарей с ключами:
        - value: нормализованное число
        - original: оригинальное значение
        - unit: единица измерения
        - context: контекст вокруг числа
        - entity: привязанный объект (область, город и т.д.)
        - title: заголовок таблицы
        - source: источник
        - page: страница
    """
    # Определяем тип статистического запроса
    query_lower = query.lower()
    number_type = None
    for stat_type, keywords in STATISTICAL_CONTEXT_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            number_type = stat_type
            break
    
    # Паттерн для чисел с единицами измерения
    # Ищем: число + пробел + единица измерения
    number_unit_pattern = r'(\d{1,3}(?:\s?\d{3})*(?:[.,]\d+)?)\s*([а-яё]+\.?|[а-яё]+)'
    
    numbers = []
    lines = text.split('\n')
    
    for line_idx, line in enumerate(lines):
        # Ищем числа с единицами
        matches = list(re.finditer(number_unit_pattern, line, re.IGNORECASE))
        
        # Если не нашли единицы, ищем просто числа
        if not matches:
            number_pattern = r'\b(\d{1,3}(?:\s?\d{3})*(?:[.,]\d+)?)\b'
            matches = list(re.finditer(number_pattern, line))
        
        for match in matches:
            number_str = match.group(1) if match.lastindex >= 1 else match.group(0)
            unit = match.group(2) if match.lastindex >= 2 else ""
            
            # Берем расширенный контекст (до 50 символов с каждой стороны)
            start = max(0, match.start() - 50)
            end = min(len(line), match.end() + 50)
            context = line[start:end].strip()
            
            # Также берем предыдущую и следующую строки для лучшего контекста
            full_context = context
            if line_idx > 0:
                prev_line = lines[line_idx - 1][:50]
                full_context = prev_line + " | " + full_context
            if line_idx < len(lines) - 1:
                next_line = lines[line_idx + 1][:50]
                full_context = full_context + " | " + next_line
            
            # Проверяем релевантность
            if not _is_relevant_number(full_context, query, number_type):
                continue
            
            # Нормализуем число и единицу
            normalized_value, normalized_unit = _normalize_unit(number_str, unit)
            if normalized_value is None:
                continue
            
            # Извлекаем объект (область, город и т.д.)
            entity = _extract_entity_from_context(full_context, query)
            
            # Формируем результат
            numbers.append({
                'value': normalized_value,
                'original': number_str,
                'unit': normalized_unit or unit,
                'context': full_context,
                'entity': entity,
                'title': title,
                'source': source,
                'page': page,
                'line': line_idx
            })
    
    return numbers


def _is_statistical_query(query: str) -> bool:
    """Определяет, является ли запрос статистическим."""
    statistical_keywords = {
        'число', 'количество', 'численность', 'сколько',
        'население', 'человек', 'житель', 'ввп', 'производство',
        'экспорт', 'импорт', 'объем', 'добыча', 'тонн', 'тонна',
        'поголовье', 'урожайность', 'площадь', 'стоимость',
        'заработная', 'доход', 'расход', 'инвестиция', 'бюджет'
    }
    query_lower = query.lower()
    return any(kw in query_lower for kw in statistical_keywords)


def _format_table_like_data(text: str) -> str:
    """
    Пытается форматировать табличные данные для лучшей читаемости.
    """
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Если строка содержит много чисел, разделенных пробелами - форматируем как таблицу
        numbers = re.findall(r'\d+[\d\s.,]*', line)
        if len(numbers) >= 3:
            # Разделяем по множественным пробелам
            parts = re.split(r'\s{2,}', line)
            if len(parts) >= 2:
                formatted_lines.append(' | '.join(p.strip() for p in parts if p.strip()))
            else:
                formatted_lines.append(line)
        else:
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


def compose_answer(query: str, results: List[Dict], top_k: int = 3) -> str:
    """
    Формирует структурированный ответ на основе найденных чанков.
    
    Args:
        query: Исходный запрос пользователя
        results: Список найденных чанков с метаданными
        top_k: Количество лучших результатов для ответа
    
    Returns:
        Отформатированный ответ с контекстом и источниками
    """
    if not results:
        return f"По запросу «{query}» ничего не найдено."
    
    # Фильтруем результаты по порогу релевантности (снижен для гибридного поиска)
    filtered_results = [r for r in results if r.get('score', 0) >= MIN_SCORE_THRESHOLD * 0.5]
    
    if not filtered_results:
        return f"По запросу «{query}» найдены результаты с низкой релевантностью."
    
    # Сортируем по score (на всякий случай)
    filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    # Берем топ результатов
    top_results = filtered_results[:top_k]
    
    # Определяем тип запроса
    is_statistical = _is_statistical_query(query)
    
    # Сначала извлекаем все числа из всех результатов
    extracted_numbers = []  # Для статистических запросов - извлеченные числа
    results_data = []  # Сохраняем данные результатов для последующей обработки
    
    for i, result in enumerate(top_results, 1):
        text = result.get('text', '').strip()
        title = result.get('title', '').strip()
        source = result.get('source', 'Неизвестный источник')
        page = result.get('page', 0)
        
        if not text:
            continue
        
        # Для статистических запросов извлекаем числа с улучшенной обработкой
        if is_statistical:
            numbers = _extract_numbers_from_text(text, query, title, source, page)
            if numbers:
                extracted_numbers.extend(numbers)
        
        # Сохраняем данные результата
        results_data.append({
            'text': text,
            'title': title,
            'source': source,
            'page': page,
            'result': result
        })
    
    # Теперь формируем контекст из найденных чанков
    context_parts = []
    sources = []
    seen_texts = set()  # Для дедупликации похожих фрагментов
    
    for i, data in enumerate(results_data, 1):
        text = data['text']
        title = data['title']
        source = data['source']
        page = data['page']
        
        # Простая дедупликация: пропускаем очень похожие тексты
        text_hash = hash(text[:100])  # Хеш первых 100 символов
        if text_hash in seen_texts:
            continue
        seen_texts.add(text_hash)
        
        # Очищаем и форматируем текст
        cleaned_text = _clean_text(text, max_length=1000)
        
        # Пытаемся форматировать как таблицу, если есть много чисел
        if is_statistical and re.search(r'\d+', cleaned_text):
            cleaned_text = _format_table_like_data(cleaned_text)
        
        # Формируем структурированный фрагмент
        fragment_parts = []
        
        # Заголовок таблицы
        if title and len(title) > 10:
            clean_title = " ".join(title.split()[:25])  # Первые 25 слов
            fragment_parts.append(f"[Заголовок таблицы] {clean_title}")
            fragment_parts.append("")
        
        # Если есть извлеченные числа для этого фрагмента, показываем их отдельно
        fragment_numbers = [n for n in extracted_numbers if n.get('source') == source and n.get('page') == page]
        if fragment_numbers and is_statistical:
            fragment_parts.append("[Ключевые значения из таблицы]:")
            for num_data in fragment_numbers[:5]:  # Топ-5 чисел
                value = num_data['value']
                unit = num_data.get('unit', '')
                entity = num_data.get('entity', '')
                context_short = num_data.get('context', '')[:50]
                
                if entity:
                    fragment_parts.append(f"  {entity}: {value:,.2f} {unit}")
                elif context_short:
                    fragment_parts.append(f"  {value:,.2f} {unit} ({context_short}...)")
                else:
                    fragment_parts.append(f"  {value:,.2f} {unit}")
            fragment_parts.append("")
        
        # Полный текст таблицы
        fragment_parts.append("[Полный текст таблицы]:")
        fragment_parts.append(cleaned_text)
        
        # Добавляем контекст
        context_parts.append("\n".join(fragment_parts))
        
        # Собираем уникальные источники
        source_info = f"{source} (стр. {page})"
        if source_info not in sources:
            sources.append(source_info)
    
    # Формируем итоговый ответ
    answer_parts = [
        f"[ОТВЕТ] Запрос: «{query}»",
        ""
    ]
    
    # Для статистических запросов показываем извлеченные числа отдельно
    if is_statistical and extracted_numbers:
        answer_parts.append("[ИЗВЛЕЧЕННЫЕ ДАННЫЕ]:")
        answer_parts.append("")
        
        # Группируем числа по объектам (области, города и т.д.)
        numbers_by_entity = {}
        numbers_without_entity = []
        
        for num_data in extracted_numbers:
            entity = num_data.get('entity')
            if entity:
                if entity not in numbers_by_entity:
                    numbers_by_entity[entity] = []
                numbers_by_entity[entity].append(num_data)
            else:
                numbers_without_entity.append(num_data)
        
        # Агрегируем данные по объектам
        if numbers_by_entity:
            answer_parts.append("По объектам:")
            answer_parts.append("")
            for entity, nums_list in sorted(numbers_by_entity.items()):
                # Агрегируем: среднее, сумма, мин, макс
                values = [n['value'] for n in nums_list]
                if values:
                    avg_val = sum(values) / len(values)
                    sum_val = sum(values)
                    min_val = min(values)
                    max_val = max(values)
                    
                    # Берем основной источник (самый частый)
                    sources_count = {}
                    for n in nums_list:
                        key = f"{n['source']} (стр. {n['page']})"
                        sources_count[key] = sources_count.get(key, 0) + 1
                    main_source = max(sources_count.items(), key=lambda x: x[1])[0]
                    
                    # Формируем ответ
                    unit = nums_list[0].get('unit', '')
                    answer_parts.append(f"  {entity}:")
                    if len(values) == 1:
                        answer_parts.append(f"    Значение: {values[0]:,.2f} {unit}")
                    else:
                        answer_parts.append(f"    Среднее: {avg_val:,.2f} {unit}")
                        answer_parts.append(f"    Сумма: {sum_val:,.2f} {unit}")
                        answer_parts.append(f"    Диапазон: {min_val:,.2f} - {max_val:,.2f} {unit}")
                        answer_parts.append(f"    Количество значений: {len(values)}")
                    answer_parts.append(f"    Источник: {main_source}")
                    answer_parts.append("")
        
        # Показываем числа без привязки к объектам
        if numbers_without_entity:
            answer_parts.append("Общие данные:")
            answer_parts.append("")
            
            # Группируем по источникам
            numbers_by_source = {}
            for num_data in numbers_without_entity[:15]:  # Ограничиваем
                key = f"{num_data['source']} (стр. {num_data['page']})"
                if key not in numbers_by_source:
                    numbers_by_source[key] = []
                numbers_by_source[key].append(num_data)
            
            for source_key, nums_list in list(numbers_by_source.items())[:3]:
                answer_parts.append(f"  Источник: {source_key}")
                for num_data in nums_list[:5]:
                    value = num_data['value']
                    unit = num_data.get('unit', '')
                    title = num_data.get('title', '')
                    context = num_data.get('context', '')[:60]
                    
                    if title:
                        answer_parts.append(f"    • {value:,.2f} {unit} (таблица: {title[:40]}...)")
                    elif context:
                        answer_parts.append(f"    • {value:,.2f} {unit} (контекст: {context}...)")
                    else:
                        answer_parts.append(f"    • {value:,.2f} {unit}")
                answer_parts.append("")
        
        answer_parts.append("---")
        answer_parts.append("")
    
    answer_parts.append("На основе найденных данных:")
    answer_parts.append("")
    
    # Добавляем контекст
    if context_parts:
        # Разделяем фрагменты визуально
        for i, fragment in enumerate(context_parts, 1):
            answer_parts.append(f"{'='*10}")
            answer_parts.append(f"Фрагмент {i}:")
            answer_parts.append(f"{'='*10}")
            answer_parts.append(fragment)
            answer_parts.append("")
    else:
        answer_parts.append("Не удалось извлечь релевантный контекст из найденных результатов.")
        answer_parts.append("")
    
    # Добавляем источники
    if sources:
        answer_parts.append("[ИСТОЧНИКИ]:")
        for source in sources:
            answer_parts.append(f"  - {source}")
    
    # Добавляем информацию о релевантности
    if top_results:
        avg_score = sum(r.get('score', 0) for r in top_results) / len(top_results)
        max_score = max(r.get('score', 0) for r in top_results)
        answer_parts.append("")
        answer_parts.append(f"Релевантность: {avg_score:.3f} (макс: {max_score:.3f}, найдено: {len(filtered_results)})")
    
    return "\n".join(answer_parts)


def format_detailed_results(query: str, results: List[Dict]) -> str:
    """
    Форматирует детальные результаты поиска с метаданными.
    
    Args:
        query: Исходный запрос
        results: Список найденных чанков
    
    Returns:
        Отформатированная строка с детальной информацией
    """
    if not results:
        return "Ничего не найдено."
    
    lines = [f"🔍 Детальные результаты для запроса: «{query}»", ""]
    
    for i, r in enumerate(results, start=1):
        lines.append(f"{'='*10}")
        lines.append(f"Результат #{i} (релевантность: {r.get('score', 0):.3f})")
        lines.append(f"{'='*10}")
        lines.append(f"📄 Источник: {r.get('source', 'Неизвестно')}")
        lines.append(f"📑 Страница: {r.get('page', '?')}")
        lines.append("")
        lines.append("Текст:")
        lines.append("-" * 10)
        text = r.get('text', '').strip()
        # Ограничиваем длину текста для читаемости
        if len(text) > 500:
            text = text[:500] + "..."
        lines.append(text)
        lines.append("")
    
    return "\n".join(lines)

def main() -> None:
    # Базовая директория src/
    base_dir = Path(__file__).resolve().parents[1]

    # Папка с уже построенным индексом
    vector_store_dir = base_dir / "prepare_db" / "vector_store"

    vectorizer = HashVectorizer(dimension=256)

    retriever = SemanticRetriever(
        vectorizer=vectorizer,
        data_path=vector_store_dir / "data.json",
    )

    print("RAG CLI запущен.\n")

    # Список предопределённых запросов
    predefined_queries = [
        # Демография
        # "Численность населения Республики Беларусь",
        "Численность населения по областям Беларуси",
        "Население Минской области в 2024 году",
        "Число городского населения Беларуси",
        # "Сколько сельских жителей в Беларуси",
        # "Плотность населения по регионам",
        # "Естественный прирост населения в Беларуси",
        # "Число родившихся в Беларуси",
        # "Число умерших в Беларуси",
        # "Миграционный прирост населения",
        # "Численность населения Минска",
        # "Население областных центров Беларуси",
        # "Число мужчин и женщин в Беларуси",
        # "Возрастная структура населения",
        # "Число населения трудоспособного возраста",

        # Регионы (том 1 и том 2)
        # "Социально-экономическое развитие Брестской области",
        # "Основные показатели Витебской области",
        # "Экономика Гомельской области",
        # "Промышленность Гродненской области",
        # "Сельское хозяйство Минской области",
        # "Показатели Могилевской области",
        # "Численность населения по районам Минской области",
        # "Производство промышленной продукции по регионам",
        # "Инвестиции в основной капитал по областям",
        # "Объем розничного товарооборота по регионам",
        "Уровень безработицы по областям",
        "Средняя заработная плата по регионам",
        # "Число организаций по областям",
        "Число индивидуальных предпринимателей по регионам",

        # Экономика и макроэкономика
        "Валовой внутренний продукт Беларуси",
        "ВВП на душу населения",
        # "Темпы роста ВВП",
        # "Индекс потребительских цен",
        "Инфляция в Беларуси",
        # "Объем промышленного производства",
        # "Доля промышленности в ВВП",
        # "Экспорт товаров Беларуси",
        # "Импорт товаров Беларуси",
        # "Сальдо внешней торговли",
        # "Основные торговые партнеры Беларуси",

        # Беларусь и Россия
        # "Товарооборот между Беларусью и Россией",
        "Экспорт Беларуси в Россию",
        "Импорт из России в Беларусь",
        "Доля России во внешней торговле Беларуси",
        # "Сравнение ВВП Беларуси и России",
        # "Сравнение численности населения Беларуси и России",
        # "Сравнение уровня инфляции Беларусь Россия",
        # "Сравнение средней заработной платы Беларусь Россия",

        # Социальная сфера
        "Средняя номинальная заработная плата",
        "Реальная заработная плата в Беларуси",
        # "Доходы населения Беларуси",
        # "Расходы населения",
        # "Уровень бедности в Беларуси",
        "Число пенсионеров",
        "Средний размер пенсии",
        # "Число учащихся в школах",
        "Число студентов в вузах",
        # "Число учреждений общего среднего образования",
        "Число учреждений здравоохранения",
        # "Обеспеченность врачами населения",

        # Сельское хозяйство
        "Производство зерна в Беларуси",
        "Производство картофеля",
        "Производство молока",
        # "Поголовье крупного рогатого скота",
        # "Поголовье свиней",
        # "Урожайность сельскохозяйственных культур",

        # Промышленность и энергетика
        "Добыча нефти в Беларуси",
        "Производство электроэнергии",
        # "Потребление электроэнергии",
        # "Производство нефтепродуктов",
        # "Объем производства машиностроения",

        # Строительство и жилье
        "Ввод жилья в эксплуатацию",
        "Общая площадь введенного жилья",
        # "Жилищный фонд Беларуси",
        # "Обеспеченность жильем населения",

        # Транспорт и связь
        "Грузооборот транспорта",
        "Пассажирооборот транспорта",
        # "Перевозки железнодорожным транспортом",
        # "Автомобильные перевозки грузов",

        # Финансы
        # "Доходы консолидированного бюджета",
        # "Расходы консолидированного бюджета",
        "Государственный долг Беларуси",
        # "Инвестиции в основной капитал"
    ]

    if not predefined_queries_flag and not user_queries_flag:
        print("Оба режима выключены. Задайте хотя бы один режим: predefined_queries=True или user_queries=True")
        return

    try:
        # Предопределённые запросы
        if predefined_queries_flag:
            for query in predefined_queries:
                print(f"\n{'='*20}")
                print(f"Запрос: {query}")
                print('='*20)
                
                results = retriever.search(query, top_k=10, hybrid_weight=0.3)

                if not results:
                    print("[ОШИБКА] Ничего не найдено.\n")
                    continue

                # Формируем улучшенный ответ
                answer = compose_answer(query, results, top_k=3)
                print(answer)
                
                # Опционально: показываем детальные результаты
                # print("\n" + format_detailed_results(query, results))
                print()

        # Пользовательский интерактив
        if user_queries_flag:
            print("\n" + "="*70)
            print("Теперь можно вводить свои запросы (Ctrl+C для выхода)")
            print("="*70 + "\n")
            
            while True:
                query = input("> ").strip()
                if not query:
                    continue

                print(f"\n{'='*20}")
                print(f"Обработка запроса: {query}")
                print('='*20)
                
                results = retriever.search(query, top_k=10, hybrid_weight=0.3)
                
                if not results:
                    print("[ОШИБКА] Ничего не найдено.\n")
                    continue

                # Формируем улучшенный ответ
                answer = compose_answer(query, results, top_k=3)
                print(answer)
                
                # Опционально: показываем детальные результаты
                # print("\n" + format_detailed_results(query, results))
                print()

    except KeyboardInterrupt:
        print("\nВыход из программы.")


if __name__ == "__main__":
    main()
