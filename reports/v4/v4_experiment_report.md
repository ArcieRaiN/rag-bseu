# V4 experiment report

## Dataset
- Documents: 12 local PDF publications in `usage/documents`.
- Vector store: 2974 chunks in `usage/vector_store`, 12 sources.
- Test set: 1500 generated retrieval questions with chunk/page-level expected sources (`tests/benchmarks/benchmark_v4.json`).
- Chunk schema uses `search_context` only (legacy `context` removed).

## Enrichment coverage (current store)
| Field | Present |
|---|---:|
| section | 98.42% |
| geo | 99.66% |
| metrics | 100.00% |
| units | 74.55% |
| years | 99.19% |
| search_context | 100.00% |

## Retrieval results (current)
| Variant | Queries | Hit@1 | Hit@3 | Hit@5 | MRR | Avg time, s |
|---|---:|---:|---:|---:|---:|---:|
| current | 1500 | 0.364 | 0.535 | 0.613 | 0.469 | 0.146 |

## Ablations
Ablations reuse the FAISS index under `usage/vector_store` and change retrieval metadata in memory.

| Variant | Queries | Hit@1 | Hit@3 | Hit@5 | MRR | Avg time, s |
|---|---:|---:|---:|---:|---:|---:|
| regex_geo_only | 1500 | 0.269 | 0.470 | 0.577 | 0.394 | 0.081 |
| years_metrics_units_only | 1500 | 0.367 | 0.535 | 0.609 | 0.470 | 0.139 |
| search_context_only | 1500 | 0.282 | 0.483 | 0.586 | 0.407 | 0.072 |
| final_best_full | 1500 | 0.364 | 0.535 | 0.613 | 0.469 | 0.147 |

## Timing
- Rule metadata pass: 5.866 s total, 0.001972 s/chunk.
- FAISS rebuild with `intfloat/multilingual-e5-large`: 1606.859 s.
- Retrieval average time on the test set: 0.146 s/query.
