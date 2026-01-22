# rag-bseu

## CLI (numeric extraction)

Запуск (Windows PowerShell):

```powershell
$env:PYTHONPATH="C:\Users\alex\Downloads\projects\rag-bseu"
.\.venv\Scripts\python.exe .\src\usage\cli.py --strict --query "Численность населения Минска" --aggregate
```

Режимы:
- `--strict`: строгая фильтрация (по умолчанию) — возвращает только уверенные извлечения.
- `--relaxed`: показывает больше извлечений (включая низкую уверенность).

Логирование raw-фрагментов для ручного ревью:

```powershell
.\.venv\Scripts\python.exe .\src\usage\cli.py --query "Производство молока" --log-raw .\reference\raw_hits.jsonl
```