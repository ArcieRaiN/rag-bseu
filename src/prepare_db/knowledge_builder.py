from __future__ import annotations

"""
PIPELINE 1: подготовка базы знаний (prepare_db).

Задачи (высокоуровневый skeleton):
1. Взять PDF из src/prepare_db/documents/
2. Разбить документ на чанки с помощью LlamaIndex
3. Передать ВСЕ чанки в LLM (Ollama) ОДНИМ ЗАПРОСОМ
   и получить для каждого чанка:
   - context (краткое описание на основе всего документа)
   - geo / metrics / years / time_granularity / oked
4. Сохранить чанки в data.json
5. Построить embedding ТОЛЬКО для поля context
6. Загрузить embeddings в FAISS (index.faiss)

Реализация намеренно оставлена на уровне skeleton, чтобы:
- зафиксировать интерфейсы и архитектурные границы
- не смешивать здесь сетевой/IO‑код с бизнес‑логикой
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import shutil
import faiss

from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SimpleNodeParser

from src.main.models import Chunk
from src.main.ollama_client import OllamaClient
from src.main.vectorizer import HashVectorizer


@dataclass
class BuildConfig:
    documents_dir: Path
    output_dir: Path
    vector_dim: int = 256


class KnowledgeBaseBuilder:
    """
    Высокоуровневый фасад для подготовки базы знаний.

    ВАЖНО:
    - этот класс НЕ занимается скачиванием PDF (site_parser.py остаётся заглушкой)
    - LlamaIndex интеграция описана как TODO‑интеграция (скелет)
    """

    def __init__(self, config: BuildConfig, llm_client: OllamaClient | None = None):
        self._config = config
        self._llm = llm_client or OllamaClient()
        self._vectorizer = HashVectorizer(dimension=config.vector_dim)

    # -------------------- Публичный интерфейс -------------------- #

    def build(self) -> None:
        """
        Основной entrypoint для подготовки базы знаний.

        Реальная реализация должна:
        - пройти по всем PDF
        - вызвать `_chunk_pdf_with_llamaindex`
        - затем `_enrich_chunks_with_llm_batch`
        - сохранить JSON и построить FAISS‑индекс
        """
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

        all_chunks: List[Chunk] = []
        chunk_id_counter = 0

        for pdf_path in sorted(self._config.documents_dir.glob("*.pdf")):
            # 1–2. Чанкинг PDF через LlamaIndex (skeleton)
            raw_chunks = self._chunk_pdf_with_llamaindex(pdf_path)

            # Присваиваем id на уровне всего корпуса
            for ch in raw_chunks:
                ch.id = f"{pdf_path.name}::page{ch.page}::chunk{chunk_id_counter}"
                chunk_id_counter += 1

            # 3. LLM‑enrichment для чанков ОДНИМ запросом
            enriched_chunks = self._enrich_chunks_with_llm_batch(pdf_path.name, raw_chunks)
            all_chunks.extend(enriched_chunks)

        # 4. Сохранение data.json
        data_json_path = self._config.output_dir / "data.json"
        with open(data_json_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "id": ch.id,
                        "context": ch.context,
                        "text": ch.text,
                        "source": ch.source,
                        "page": ch.page,
                        "geo": ch.geo,
                        "metrics": ch.metrics,
                        "years": ch.years,
                        "time_granularity": ch.time_granularity,
                        "oked": ch.oked,
                    }
                    for ch in all_chunks
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        # 5–6. Embeddings по context + FAISS
        self._build_faiss_index(all_chunks, self._config.output_dir / "index.faiss")

        # Дополнительная мета‑информация
        meta_path = self._config.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "vectorizer": type(self._vectorizer).__name__,
                    "dimension": self._vectorizer.dimension,
                    "chunks": len(all_chunks),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    # -------------------- Skeleton‑методы для интеграции -------------------- #

    def _chunk_pdf_with_llamaindex(self, pdf_path: Path) -> List[Chunk]:
        """
        Интеграция с LlamaIndex для чанкинга PDF.

        Использует:
        - PDFReader для чтения PDF
        - SimpleNodeParser для разбиения на чанки (chunk_size=512, chunk_overlap=50)
        - Извлечение метаданных страницы из node.metadata

        Возвращает список Chunk с заполненными:
        - text (текст чанка)
        - source (имя PDF)
        - page (номер страницы из метаданных)
        - context и метаданные пока пустые (заполнятся в _enrich_chunks_with_llm_batch)
        """
        # Создаём временную директорию с одним PDF для LlamaIndex
        # (PDFReader работает с директориями)
        temp_dir = pdf_path.parent / f"_temp_{pdf_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        temp_pdf = temp_dir / pdf_path.name

        try:
            # Копируем PDF во временную директорию
            shutil.copy2(pdf_path, temp_pdf)

            # Читаем PDF через LlamaIndex
            pdf_reader = PDFReader()
            documents = pdf_reader.load_data(file=str(temp_pdf))

            # Парсим документы на ноды (чанки)
            # Используем разумные параметры для статистических документов
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=512,  # размер чанка в символах
                chunk_overlap=50,  # перекрытие между чанками
            )

            chunks: List[Chunk] = []
            for doc in documents:
                nodes = node_parser.get_nodes_from_documents([doc])

                for node in nodes:
                    # Извлекаем номер страницы из метаданных
                    # LlamaIndex обычно сохраняет page_label или page_number
                    page_num = 0
                    if hasattr(node, "metadata"):
                        page_num = node.metadata.get("page_label", 0)
                        if not page_num:
                            page_num = node.metadata.get("page_number", 0)
                        if not page_num:
                            page_num = node.metadata.get("page", 0)
                        try:
                            page_num = int(page_num) if page_num else 0
                        except (ValueError, TypeError):
                            page_num = 0

                    # Создаём Chunk с пустыми полями для enrichment
                    chunk = Chunk(
                        id="",  # будет присвоен в build()
                        context="",  # заполнится в _enrich_chunks_with_llm_batch
                        text=node.text or "",
                        source=pdf_path.name,
                        page=page_num,
                        geo=None,
                        metrics=None,
                        years=None,
                        time_granularity=None,
                        oked=None,
                    )
                    chunks.append(chunk)

        finally:
            # Удаляем временную директорию
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        return chunks

    def _enrich_chunks_with_llm_batch(self, pdf_name: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Батчевое LLM‑обогащение чанков с контекстом.

        Для каждого чанка передается:
        - Страницы 3-8 (содержание и примечания)
        - 3 чанка до и 3 чанка после текущего
        - Текущий чанк для обогащения

        Обрабатывается батчами для ускорения.
        """
        if not chunks:
            return []
        
        # Получаем чанки со страниц 3-8 (содержание)
        toc_chunks = [ch for ch in chunks if 3 <= ch.page <= 8]
        
        # Обрабатываем батчами по 10 чанков
        BATCH_SIZE = 10
        all_enriched_chunks: List[Chunk] = []
        total = len(chunks)
        
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"   Обработка батча {batch_num}/{total_batches} (чанки {batch_start+1}-{batch_end} из {total})")
            
            enriched_batch = self._enrich_batch_with_context(pdf_name, batch_chunks, chunks, toc_chunks)
            all_enriched_chunks.extend(enriched_batch)
        
        return all_enriched_chunks
    
    def _enrich_batch_with_context(self, pdf_name: str, target_chunks: List[Chunk], all_chunks: List[Chunk], toc_chunks: List[Chunk]) -> List[Chunk]:
        """
        Обогащение батча чанков с контекстом (3 до, 3 после, страницы 3-8).
        """
        enriched_results: List[Chunk] = []
        
        for target_chunk in target_chunks:
            # Находим индекс текущего чанка
            target_idx = next((i for i, ch in enumerate(all_chunks) if ch.id == target_chunk.id), -1)
            if target_idx == -1:
                enriched_results.append(target_chunk)
                continue
            
            # Получаем контекстные чанки (3 до и 3 после)
            context_before = all_chunks[max(0, target_idx - 3):target_idx]
            context_after = all_chunks[target_idx + 1:min(len(all_chunks), target_idx + 4)]
            
            # Формируем данные для промпта
            context_data = {
                "toc": [{"page": ch.page, "text": ch.text[:200]} for ch in toc_chunks[:10]],  # Первые 10 чанков из содержания
                "before": [{"page": ch.page, "text": ch.text[:200]} for ch in context_before],
                "target": {
                    "chunk_id": target_chunk.id,
                    "page": target_chunk.page,
                    "text": target_chunk.text
                },
                "after": [{"page": ch.page, "text": ch.text[:200]} for ch in context_after]
            }
            
            # Обогащаем один чанк с контекстом
            enriched = self._enrich_single_with_context(pdf_name, target_chunk, context_data)
            enriched_results.append(enriched)
        
        return enriched_results
    
    def _enrich_single_with_context(self, pdf_name: str, chunk: Chunk, context_data: Dict[str, Any]) -> Chunk:
        """
        Обогащение одного чанка с контекстом.
        """
        system_prompt = (
            "Ты — аналитик по официальной статистике Республики Беларусь. "
            "Твоя задача — обогатить чанк документа структурированными метаданными. "
            "Верни ТОЛЬКО JSON-объект (не массив!) с полями: chunk_id, context, geo, metrics, years, time_granularity, oked."
        )
        
        prompt = (
            f"Документ: {pdf_name}\n\n"
            "СОДЕРЖАНИЕ ДОКУМЕНТА (страницы 3-8):\n"
            f"{json.dumps(context_data['toc'], ensure_ascii=False, indent=2)}\n\n"
            "КОНТЕКСТ ДО ТЕКУЩЕГО ЧАНКА (3 предыдущих чанка):\n"
            f"{json.dumps(context_data['before'], ensure_ascii=False, indent=2)}\n\n"
            "ТЕКУЩИЙ ЧАНК ДЛЯ ОБОГАЩЕНИЯ:\n"
            f"ID: {context_data['target']['chunk_id']}\n"
            f"Страница: {context_data['target']['page']}\n"
            f"Текст: {context_data['target']['text']}\n\n"
            "КОНТЕКСТ ПОСЛЕ ТЕКУЩЕГО ЧАНКА (3 следующих чанка):\n"
            f"{json.dumps(context_data['after'], ensure_ascii=False, indent=2)}\n\n"
            "Задача: опиши ТЕКУЩИЙ ЧАНК на основе его текста и контекста.\n"
            "Верни JSON-объект с полями:\n"
            "- chunk_id: точно такой же как ID выше\n"
            "- context: краткое описание чанка (1-2 предложения) на основе контекста\n"
            "- geo: географический объект или null\n"
            "- metrics: список показателей или null\n"
            "- years: список годов или null\n"
            "- time_granularity: 'year'/'quarter'/'month'/'day' или null\n"
            "- oked: код ОКЭД или null\n\n"
            "ВАЖНО: Верни ТОЛЬКО JSON-объект {}, НЕ массив!"
        )
        
        try:
            raw_response = self._llm.generate(prompt, system_prompt=system_prompt, format="json")
            enriched_data = self._parse_llm_single_enrichment(raw_response, chunk.id)
            
            if enriched_data:
                # Обновляем чанк данными от LLM
                if enriched_data.get("context"):
                    chunk.context = str(enriched_data["context"])[:200]
                elif chunk.text:
                    chunk.context = chunk.text[:200]
                else:
                    chunk.context = "нет текста"
                
                if "geo" in enriched_data:
                    chunk.geo = enriched_data["geo"]
                if "metrics" in enriched_data:
                    chunk.metrics = enriched_data["metrics"]
                if "years" in enriched_data:
                    chunk.years = self._normalize_years(enriched_data["years"])
                if "time_granularity" in enriched_data:
                    chunk.time_granularity = enriched_data["time_granularity"]
                if "oked" in enriched_data:
                    chunk.oked = enriched_data["oked"]
        except Exception as e:
            print(f"   ⚠️  Ошибка при обогащении чанка {chunk.id}: {e}")
            # Оставляем чанк без обогащения
            if not chunk.context:
                chunk.context = chunk.text[:200] if chunk.text else "нет текста"
        
        return chunk
    
    def _enrich_single_batch(self, pdf_name: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Обогащение одного батча чанков.
        """
        if not chunks:
            return []

        # Формируем промпт с описанием документа и всеми чанками
        chunks_data = []
        for i, ch in enumerate(chunks):
            chunks_data.append({
                "chunk_id": ch.id,
                "text": ch.text,
                "page": ch.page,
            })

        # System prompt для строгого контроля формата
        system_prompt = (
            "Ты — аналитик по официальной статистике Республики Беларусь. "
            "Твоя задача — обогатить чанки документа структурированными метаданными. "
            "ОБЯЗАТЕЛЬНО верни JSON-массив, который начинается с символа '[' и заканчивается ']'. "
            "НЕ возвращай объект '{...}'. НЕ добавляй пояснений. Только JSON-массив."
        )
        
        # Упрощенный промпт для одного или нескольких чанков
        if len(chunks_data) == 1:
            # Для одного чанка - более простой промпт
            chunk = chunks_data[0]
            prompt = (
                f"Документ: {pdf_name}\n\n"
                f"Чанк для обработки:\n"
                f"ID: {chunk['chunk_id']}\n"
                f"Страница: {chunk['page']}\n"
                f"Текст: {chunk['text'][:500]}...\n\n"
                "Верни JSON-объект с полями:\n"
                "- chunk_id: \"{chunk_id}\" (точно такой же как выше)\n"
                "- context: краткое описание чанка (1-2 предложения)\n"
                "- geo: географический объект или null\n"
                "- metrics: список показателей или null\n"
                "- years: список годов или null\n"
                "- time_granularity: 'year'/'quarter'/'month'/'day' или null\n"
                "- oked: код ОКЭД или null\n\n"
                "ВАЖНО: Верни ТОЛЬКО JSON-объект в фигурных скобках {}, НЕ массив!"
            )
        else:
            # Для нескольких чанков
            prompt = (
                f"Документ: {pdf_name}\n\n"
                "Задача: для каждого чанка верни объект с полями:\n"
                "- chunk_id: идентификатор чанка (обязательно)\n"
                "- context: краткое описание (1-2 предложения)\n"
                "- geo: географический объект или null\n"
                "- metrics: список показателей или null\n"
                "- years: список годов или null\n"
                "- time_granularity: 'year'/'quarter'/'month'/'day' или null\n"
                "- oked: код ОКЭД или null\n\n"
                "ФОРМАТ ОТВЕТА: JSON-массив объектов. Начинай с '[' и заканчивай ']'.\n"
                "Пример: [{\"chunk_id\": \"id1\", \"context\": \"...\", \"geo\": null, ...}, ...]\n\n"
                f"Чанки для обработки ({len(chunks_data)} шт.):\n"
                f"{json.dumps(chunks_data, ensure_ascii=False, indent=2)}\n\n"
                "Верни массив результатов для ВСЕХ чанков выше."
            )

        # Вызываем LLM с форсированием JSON формата
        # Ollama поддерживает параметр format для форсирования JSON
        raw_response = self._llm.generate(
            prompt, 
            system_prompt=system_prompt,
            format="json"  # Форсируем JSON формат
        )

        # Парсим JSON‑ответ
        enriched_data = self._parse_llm_batch_enrichment(raw_response)
        
        # Отладочная информация
        if not enriched_data:
            print(f"⚠️  WARNING: LLM не вернул данные для обогащения чанков документа {pdf_name}")
            print(f"   Количество чанков: {len(chunks)}")
            print(f"   Первые 500 символов ответа LLM: {raw_response[:500]}")

        # Фильтруем только словари из enriched_data
        valid_enriched_data = [item for item in enriched_data if isinstance(item, dict)]
        
        # Создаём словарь chunk_id -> enriched_data для быстрого поиска
        enriched_map: Dict[str, Dict[str, Any]] = {}
        for item in valid_enriched_data:
            chunk_id = item.get("chunk_id")
            if chunk_id:
                enriched_map[str(chunk_id)] = item
        
        # Проверяем, сколько чанков было обогащено
        if len(enriched_map) < len(chunks):
            print(f"⚠️  WARNING: Только {len(enriched_map)} из {len(chunks)} чанков были обогащены для документа {pdf_name}")
            # Показываем примеры chunk_id для отладки
            if chunks:
                print(f"   Пример chunk_id из чанков: {chunks[0].id}")
            if enriched_map:
                example_id = list(enriched_map.keys())[0]
                print(f"   Пример chunk_id из ответа LLM: {example_id}")

        # Обогащаем чанки данными от LLM
        enriched_chunks: List[Chunk] = []
        for ch in chunks:
            enriched = enriched_map.get(ch.id, {})
            
            # Если для одного чанка не нашли по ID, но есть один объект в enriched_data - используем его
            if not enriched and len(chunks) == 1 and valid_enriched_data:
                # Для одного чанка модель могла вернуть объект без chunk_id
                if len(valid_enriched_data) == 1:
                    enriched = valid_enriched_data[0]
                    # Добавляем chunk_id если его нет
                    if "chunk_id" not in enriched:
                        enriched["chunk_id"] = ch.id
                elif len(valid_enriched_data) > 0:
                    # Если несколько объектов, берем первый
                    enriched = valid_enriched_data[0]
                    if "chunk_id" not in enriched:
                        enriched["chunk_id"] = ch.id
            
            # Обновляем context (с fallback на text)
            if enriched.get("context"):
                ch.context = str(enriched.get("context"))[:200]
            elif ch.text:
                ch.context = ch.text[:200]
            else:
                ch.context = "нет текста"
            
            # Обновляем метаданные: всегда обновляем, если ключ присутствует в enriched
            # Это позволяет явно установить None для полей, которые LLM вернул как null
            if "geo" in enriched:
                ch.geo = enriched["geo"]
            if "metrics" in enriched:
                ch.metrics = enriched["metrics"]
            if "years" in enriched:
                ch.years = self._normalize_years(enriched["years"])
            if "time_granularity" in enriched:
                ch.time_granularity = enriched["time_granularity"]
            if "oked" in enriched:
                ch.oked = enriched["oked"]
            
            enriched_chunks.append(ch)

        return enriched_chunks

    @staticmethod
    def _parse_llm_single_enrichment(raw: str, expected_chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Парсит ответ LLM для одного чанка (объект, не массив).
        """
        if not raw:
            return None
        
        # Удаление markdown code blocks
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        # Попытка парсинга как объект
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # Нормализуем ключи (ID -> chunk_id, и т.д.)
                normalized = KnowledgeBaseBuilder._normalize_enrichment_object(data, expected_chunk_id)
                return normalized
        except json.JSONDecodeError:
            pass
        
        # Поиск объекта в тексте
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end+1])
                if isinstance(data, dict):
                    normalized = KnowledgeBaseBuilder._normalize_enrichment_object(data, expected_chunk_id)
                    return normalized
            except json.JSONDecodeError:
                pass
        
        return None
    
    @staticmethod
    def _normalize_enrichment_object(obj: Dict[str, Any], expected_chunk_id: str) -> Dict[str, Any]:
        """
        Нормализует объект обогащения: исправляет ключи, извлекает вложенные данные.
        """
        result = {}
        
        # Нормализация chunk_id (может быть ID, chunk_id, ИД и т.д.)
        for key in ["chunk_id", "ID", "ИД", "id", "chunkId"]:
            if key in obj:
                chunk_id_value = str(obj[key])
                # Исправляем формат если нужно (page153::chunk1351 -> полный формат)
                if "::" not in chunk_id_value and "::" in expected_chunk_id:
                    # Пытаемся извлечь номер из chunk_id и использовать expected_chunk_id
                    result["chunk_id"] = expected_chunk_id
                else:
                    result["chunk_id"] = chunk_id_value
                break
        if "chunk_id" not in result:
            result["chunk_id"] = expected_chunk_id
        
        # Извлечение остальных полей
        for field in ["context", "geo", "metrics", "years", "time_granularity", "oked"]:
            if field in obj:
                result[field] = obj[field]
            else:
                # Пробуем найти вложенные объекты
                found = False
                for key, value in obj.items():
                    if isinstance(value, dict):
                        if field in value:
                            result[field] = value[field]
                            found = True
                            break
                        # Рекурсивно ищем в глубоко вложенных объектах
                        elif any(isinstance(v, dict) for v in value.values() if isinstance(v, dict)):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, dict) and field in sub_value:
                                    result[field] = sub_value[field]
                                    found = True
                                    break
                            if found:
                                break
                if not found:
                    result[field] = None
        
        # Если объект содержит вложенные объекты с нужными полями, извлекаем первый
        # (например, когда модель возвращает {"Светлогорский": {"chunk_id": ..., "context": ...}})
        for key, value in obj.items():
            if isinstance(value, dict):
                # Проверяем, содержит ли вложенный объект нужные поля
                has_enrichment_fields = any(field in value for field in ["context", "geo", "metrics", "chunk_id"])
                if has_enrichment_fields:
                    # Используем вложенный объект
                    nested = KnowledgeBaseBuilder._normalize_enrichment_object(value, expected_chunk_id)
                    # Объединяем результаты (вложенные данные имеют приоритет)
                    for k, v in nested.items():
                        if v is not None or k not in result:
                            result[k] = v
                    break
        
        return result
    
    @staticmethod
    def _parse_llm_batch_enrichment(raw: str) -> List[Dict[str, Any]]:
        """
        Робастный парсер JSON‑ответа от LLM для батчевого enrichment.

        Ищет JSON‑массив в ответе и пытается его распарсить.
        Если найден объект вместо массива, пытается извлечь из него данные.
        При ошибке возвращает пустой список.
        """
        if not raw:
            return []

        # Стратегия 1: Удаление markdown code blocks
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]  # Удаляем ```json
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]  # Удаляем ```
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]  # Удаляем закрывающий ```
        cleaned = cleaned.strip()

        # Стратегия 2: Попытка прямого парсинга всего ответа
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data)
            elif isinstance(data, dict):
                # LLM вернул объект вместо массива - пытаемся извлечь массив из него
                # Ищем ключи, которые могут содержать массив
                for key in ["chunks", "data", "results", "items", "array"]:
                    if key in data and isinstance(data[key], list):
                        return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data[key])
                # Если объект содержит chunk_id, возможно это один элемент - оборачиваем в массив
                if "chunk_id" in data:
                    return KnowledgeBaseBuilder._validate_and_fix_enrichment_data([data])
        except json.JSONDecodeError:
            pass

        # Стратегия 3: Поиск JSON-массива в тексте
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, list):
                    return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data)
            except json.JSONDecodeError:
                pass

        # Стратегия 4: Поиск JSON-объекта и попытка извлечь из него данные
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = cleaned[start : end + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    # Пытаемся найти массив внутри объекта
                    for key in ["chunks", "data", "results", "items", "array"]:
                        if key in data and isinstance(data[key], list):
                            return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(data[key])
                    # Если объект содержит chunk_id, оборачиваем в массив
                    if "chunk_id" in data:
                        return KnowledgeBaseBuilder._validate_and_fix_enrichment_data([data])
                    # Если это один объект без chunk_id, но с нужными полями - тоже принимаем
                    if any(key in data for key in ["context", "geo", "metrics", "years"]):
                        # Пытаемся создать объект с chunk_id из первого чанка в запросе
                        # Но это не идеально, лучше вернуть как есть и обработать позже
                        return KnowledgeBaseBuilder._validate_and_fix_enrichment_data([data])
            except json.JSONDecodeError:
                pass

        # Стратегия 5: Попытка найти несколько JSON объектов в тексте
        # Иногда LLM возвращает несколько объектов подряд
        objects = []
        i = 0
        while i < len(cleaned):
            if cleaned[i] == '{':
                # Находим закрывающую скобку
                depth = 0
                j = i
                while j < len(cleaned):
                    if cleaned[j] == '{':
                        depth += 1
                    elif cleaned[j] == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(cleaned[i:j+1])
                                if isinstance(obj, dict) and "chunk_id" in obj:
                                    objects.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j
                            break
                    j += 1
            i += 1
        
        if objects:
            return KnowledgeBaseBuilder._validate_and_fix_enrichment_data(objects)

        # Если ничего не сработало
        print(f"⚠️  WARNING: Не найден JSON-массив в ответе LLM. Первые 500 символов: {raw[:500]}")
        return []
    
    @staticmethod
    def _validate_and_fix_enrichment_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Валидирует и исправляет данные обогащения.
        """
        if not isinstance(data, list):
            print(f"⚠️  WARNING: Парсер вернул не список, а {type(data).__name__}")
            return []
        
        # Фильтруем только словари
        valid_items = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                # Пропускаем не-словари без предупреждения (могут быть строки из неправильного парсинга)
                continue
            valid_items.append(item)
        
        # Проверяем, что каждый элемент содержит необходимые поля
        for item in valid_items:
            # Убеждаемся, что все поля присутствуют (даже если null)
            required_fields = ["chunk_id", "context", "geo", "metrics", "years", "time_granularity", "oked"]
            for field in required_fields:
                if field not in item:
                    # Если поле отсутствует, добавляем его как None
                    item[field] = None
        
        return valid_items

    @staticmethod
    def _normalize_years(value: Any) -> Optional[List[int]]:
        """Нормализует значение years в список целых чисел."""
        if value is None:
            return None
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            try:
                return [int(value)]
            except ValueError:
                return None
        if isinstance(value, list):
            years: List[int] = []
            for v in value:
                try:
                    years.append(int(v))
                except (TypeError, ValueError):
                    continue
            return years or None
        return None

    # -------------------- FAISS -------------------- #

    def _build_faiss_index(self, chunks: List[Chunk], index_path: Path) -> None:
        """
        Строит FAISS IndexFlatIP по embeddings поля `context`.

        ВАЖНО:
        - HashVectorizer уже нормализует вектора, поэтому IndexFlatIP == cosine similarity.
        """
        if not chunks:
            # создаём пустой индекс на случай пустой базы (отладка)
            index = faiss.IndexFlatIP(self._vectorizer.dimension)
            faiss.write_index(index, str(index_path))
            return

        texts = [ch.context for ch in chunks]
        embeddings = self._vectorizer.embed_many(texts).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))

