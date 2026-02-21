# rag-bseu

Репозиторий содержит RAG-пайплайн для работы со статистическими сборниками:
1. парсинг статистических сборников, 
2. подготовка векторного хранилища,
3. интерактивный CLI-запрос,
4. Streamlit-просмотр таблиц и графиков на основе LLM-ответа.

## Установка зависимостей

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

После этого переходи в корень проекта:

```powershell
cd C:\Users\alex\Downloads\projects\rag-bseu
```


## CLI (numeric extraction)

Интерактивный CLI (`usage/query.py`) запускает `QueryPipeline`, отображает топ-чанки и теперь вызывает `OutputPipeline`, который генерирует `usage/outputs/output_df.json`. Пример запуска:

```powershell
.\.venv\Scripts\python.exe .\usage\query.py --strict --query "Численность населения Минска" --aggregate
```

Параметры:
- `--strict`: строгая фильтрация означает уверенные ответы.
- `--relaxed`: показывает больше кандидатов, включая менее уверенные.

Для логирования фрагментов используйте:

```powershell
.\.venv\Scripts\python.exe .\usage\query.py --query "Производство молока" --log-raw .\reference\raw_hits.jsonl
```

## Подготовка векторного хранилища

Скрипт `usage/prepare_vector_store.py` запускает `KnowledgeBaseBuilder`, который:

- читает PDF из `usage/documents/`,
- обогащает чанки через LLM,
- сохраняет `usage/vector_store/data.json`, `index.faiss`, `metadata.json`.

Запуск:

```powershell
.\.venv\Scripts\python.exe .\usage\prepare_vector_store.py
```

Можно использовать это перед запросами, чтобы обновить базу.

## Streamlit-интерфейс

Новая Streamlit-страница визуализирует JSON-ответ LLM в виде:
1. seaborn-barplot,
2. таблички `st.dataframe`,
3. кнопки скачивания `.xlsx`.

Запускается командой (из корня проекта):

```powershell
streamlit run usage/query.py
```

Интерфейс сам добавляет корень проекта в `sys.path`, поэтому дополнительный `PYTHONPATH` не нужен при таком запуске.

Статусы ошибок сгенерированных ответов пишутся в `usage/logs/output_df_fails.json`, если JSON оказался неверным.
