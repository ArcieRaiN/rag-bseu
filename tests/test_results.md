# Результаты тестирования rag-bseu

Сводный журнал для дипломной работы. Подробности архитектуры — в [README.md](../README.md).

## 1. Методология (набор тестов по 12 PDF)

- **Скрипты:** `python -m tests.run_v4_experiments` (основной прогон), `python -m tests.run_v4_ablation_experiments` (ablation), `python -m tests.validate_benchmark` (проверка набора).
- **Вспомогательный модуль:** [retrieval_eval.py](retrieval_eval.py) — загрузка `benchmark_v4.json`, Hit@k, MRR.
- **Данные:** `tests/benchmarks/benchmark_v4.json` — 1500 вопросов, эталон `source::page`.
- **База знаний:** `usage/vector_store` (`data.json`, `index.faiss`), 2974 чанка по 12 сборникам из `usage/documents`.
- **Метрики:** Hit@1/3/5, MRR, среднее время запроса (см. `QueryPipeline.run`).

## 2. Актуальные артефакты

| Артефакт | Путь |
|----------|------|
| Набор тестов | `tests/benchmarks/benchmark_v4.json` |
| Детальный прогон | `tests/results/v4/v4_experiments.json` |
| Ablation | `tests/results/v4/v4_ablation_experiments.json` |
| Сводка экспериментов | `reports/v4/experiment_summary.json` |
| Сводка ablation | `reports/v4/ablation_summary.json` |
| Отчёт | `reports/v4/v4_experiment_report.md` |
| Контроль search_context-only | `reports/v4/search_context_only_control_report.md` |

## 3. Извлечение (текущая конфигурация)

Источник чисел: `reports/v4/experiment_summary.json`.

| Метрика | Значение |
|---------|----------|
| Hit@1 | 36,4 % |
| Hit@3 | 53,5 % |
| Hit@5 | 61,3 % |
| MRR | 0,469 |
| Среднее время запроса | 0,146 с |

## 4. Ablation (индекс без пересборки, метаданные в памяти)

| Вариант | Hit@1 | Hit@3 | Hit@5 | MRR | Среднее время |
|---------|------:|------:|------:|----:|--------------:|
| regex_geo_only | 26,9 % | 47,0 % | 57,7 % | 0,394 | 0,081 с |
| years_metrics_units_only | 36,7 % | 53,5 % | 60,9 % | 0,470 | 0,139 с |
| search_context_only | 28,2 % | 48,3 % | 58,6 % | 0,407 | 0,072 с |
| final_best_full | 36,4 % | 53,5 % | 61,3 % | 0,469 | 0,147 с |

## 5. Обогащение (2974 чанка)

По `reports/v4/enrichment_quality_rules_v4.json`: `section` 98,42 %, `geo` 99,66 %, `metrics` 100,00 %, `units` 74,55 %, `years` 99,19 %, `search_context` 100,00 %.

## 6. Воспроизведение

1. `pip install -r requirements.txt`
2. При отсутствии индекса: `python usage/rebuild_index.py`
3. Проверка набора: `python -m tests.validate_benchmark`
4. Полный прогон извлечения: `python -m tests.run_v4_experiments` (долго; опционально `--limit 50`)

**Замечание:** на путях с кириллицей Windows иногда падает запись `index.faiss` — см. README.
