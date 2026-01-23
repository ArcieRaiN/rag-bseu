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

from llama_index.core import SimpleDirectoryReader
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
        - SimpleDirectoryReader для чтения PDF
        - SimpleNodeParser для разбиения на чанки (chunk_size=512, chunk_overlap=50)
        - Извлечение метаданных страницы из node.metadata

        Возвращает список Chunk с заполненными:
        - text (текст чанка)
        - source (имя PDF)
        - page (номер страницы из метаданных)
        - context и метаданные пока пустые (заполнятся в _enrich_chunks_with_llm_batch)
        """
        # Создаём временную директорию с одним PDF для LlamaIndex
        # (SimpleDirectoryReader работает с директориями)
        temp_dir = pdf_path.parent / f"_temp_{pdf_path.stem}"
        temp_dir.mkdir(exist_ok=True)
        temp_pdf = temp_dir / pdf_path.name

        try:
            # Копируем PDF во временную директорию
            shutil.copy2(pdf_path, temp_pdf)

            # Читаем PDF через LlamaIndex
            reader = SimpleDirectoryReader(
                input_files=[str(temp_pdf)],
                required_exts=[".pdf"],
                file_metadata=lambda x: {"file_path": str(pdf_path)},
            )
            documents = reader.load_data()

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
        Батчевое LLM‑обогащение чанков ОДНИМ запросом.

        Схема промпта:
        - кратко описать документ (pdf_name)
        - передать список чанков с их текстами и метаданными (page и пр.)
        - попросить модель вернуть JSON‑массив с объектами вида:
          {
            "chunk_id": "...",
            "context": "...",
            "geo": "...",
            "metrics": [...],
            "years": [...],
            "time_granularity": "year",
            "oked": null
          }

        ВАЖНО:
        - все чанки одного документа обрабатываются одним запросом
        - модель видит полный контекст документа для генерации context каждого чанка
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

        prompt = (
            "Ты аналитик по официальной статистике Республики Беларусь.\n"
            "Твоя задача — обогатить чанки документа структурированными метаданными.\n\n"
            f"Документ: {pdf_name}\n\n"
            "Для каждого чанка нужно:\n"
            "1. Написать краткий context (1-2 предложения) на основе ПОЛНОГО документа.\n"
            "   Context должен описывать суть чанка в контексте всего документа.\n"
            "2. Извлечь и заполнить поля:\n"
            "   - geo: географический объект (страна, область, город и т.п.) или null\n"
            "   - metrics: список показателей (например, ['добыча нефти', 'численность населения']) или null\n"
            "   - years: список целых годов, упомянутых в чанке (например, [2020, 2021, 2022]) или null\n"
            "   - time_granularity: 'year' | 'quarter' | 'month' | 'day' или null\n"
            "   - oked: код или описание из ОКЭД при наличии, иначе null\n\n"
            "Верни ТОЛЬКО JSON‑массив без пояснений. Формат:\n"
            "[\n"
            '  {\n'
            '    "chunk_id": "...",\n'
            '    "context": "...",\n'
            '    "geo": "..." или null,\n'
            '    "metrics": ["...", "..."] или null,\n'
            '    "years": [2020, 2021] или null,\n'
            '    "time_granularity": "year" или null,\n'
            '    "oked": "..." или null\n'
            '  },\n'
            "  ...\n"
            "]\n\n"
            f"Чанки документа:\n{json.dumps(chunks_data, ensure_ascii=False, indent=2)}"
        )

        # Вызываем LLM
        raw_response = self._llm.generate(prompt)

        # Парсим JSON‑ответ
        enriched_data = self._parse_llm_batch_enrichment(raw_response)

        # Создаём словарь chunk_id -> enriched_data для быстрого поиска
        enriched_map: Dict[str, Dict[str, Any]] = {}
        for item in enriched_data:
            chunk_id = item.get("chunk_id")
            if chunk_id:
                enriched_map[str(chunk_id)] = item

        # Обогащаем чанки данными от LLM
        enriched_chunks: List[Chunk] = []
        for ch in chunks:
            enriched = enriched_map.get(ch.id, {})
            ch.context = enriched.get("context", "") or ch.text[:200]  # fallback на начало text
            ch.geo = enriched.get("geo")
            ch.metrics = enriched.get("metrics")
            ch.years = self._normalize_years(enriched.get("years"))
            ch.time_granularity = enriched.get("time_granularity")
            ch.oked = enriched.get("oked")
            enriched_chunks.append(ch)

        return enriched_chunks

    @staticmethod
    def _parse_llm_batch_enrichment(raw: str) -> List[Dict[str, Any]]:
        """
        Робастный парсер JSON‑ответа от LLM для батчевого enrichment.

        Ищет JSON‑массив в ответе и пытается его распарсить.
        При ошибке возвращает пустой список.
        """
        if not raw:
            return []

        # Ищем JSON‑массив
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []

        snippet = raw[start : end + 1]
        try:
            data = json.loads(snippet)
            if not isinstance(data, list):
                return []
            return data
        except json.JSONDecodeError:
            return []

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

