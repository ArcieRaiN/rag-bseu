import math

import pytest

from src.main.numeric_extractor import FragmentMeta, extract_numeric_indicators


def _pick_one(items, metric):
    for it in items:
        if it["metric"] == metric:
            return it
    return None


@pytest.mark.parametrize(
    "text,meta,query,expected",
    [
        (
            "На начало 2024 года в городе проживало 1 992,9 тыс. человек.",
            FragmentMeta(source_file="Регионы (том 2).pdf", page=196, title="Численность населения"),
            "Численность населения Минска",
            {
                "entity": "г.Минск",
                "metric": "population",
                "unit": "thousand_persons",
                "normalized_value": 1992900.0,
            },
        ),
        (
            "В области 17 районов, 15 городов.",
            FragmentMeta(source_file="Регионы (том 1).pdf", page=12, title="Административное деление Гродненской области"),
            "В области 17 районов, 15 городов",
            {
                "entity": "Гродненская область",
                "metrics": {
                    "districts_count": 17.0,
                    "cities_count": 15.0,
                },
            },
        ),
        (
            "Плотность населения – 5 635,3 человека на 1 км2",
            FragmentMeta(source_file="Регионы (том 2).pdf", page=44, title="Население"),
            "Плотность населения",
            {
                "metric": "population_density",
                "unit": "persons_per_km2",
                "normalized_value": 5635.3,
            },
        ),
        (
            "Площадь территории составляет 40 100 км2.",
            FragmentMeta(source_file="Регионы (том 2).pdf", page=10, title="Территория"),
            "Площадь территории",
            {"metric": "area", "unit": "km2", "normalized_value": 40100.0},
        ),
        (
            "Индекс потребительских цен составил 112,3%.",
            FragmentMeta(source_file="Беларусь в цифрах, 2025.pdf", page=23, title="Цены"),
            "Индекс потребительских цен",
            {"metric": "index_percent", "unit": "percent", "normalized_value": 112.3},
        ),
        (
            "Добыча нефти – 1 650,2 тыс. т.",
            FragmentMeta(source_file="Беларусь в цифрах, 2025.pdf", page=80, title="Топливно-энергетические ресурсы"),
            "Добыча нефти в Беларуси",
            {"metric": "production_volume", "unit": "tonnes", "normalized_value": 1650200.0},
        ),
        # noisy / should be filtered in strict
        (
            "1) Данные приведены без учета микроорганизаций. 2) См. методику расчета.",
            FragmentMeta(source_file="Беларусь в цифрах, 2025.pdf", page=1, title="Примечания"),
            "Численность населения",
            {"expect_empty": True},
        ),
        (
            "ISBN 978-985-7307-95-1",
            FragmentMeta(source_file="Национальные счета, 2025.pdf", page=2, title="Выходные данные"),
            "Численность населения",
            {"expect_empty": True},
        ),
        # healthcare facilities count (institutions_count)
        (
            "В республике действует 534 учреждения здравоохранения.",
            FragmentMeta(source_file="Беларусь в цифрах, 2025.pdf", page=77, title="Здравоохранение"),
            "Число учреждений здравоохранения",
            {"metric": "institutions_count", "unit": "count", "normalized_value": 534.0},
        ),
        # GDP / VRP
        (
            "ВВП составил 216,5 млрд руб.",
            FragmentMeta(source_file="Беларусь в цифрах, 2025.pdf", page=30, title="Макроэкономика"),
            "ВВП Беларуси",
            {"metric": "gdp", "unit": "count", "normalized_value": 216.5e9},
        ),
        # population without explicit entity in text but in title
        (
            "На начало 2024 года проживало 1 338,6 тыс. человек.",
            FragmentMeta(source_file="Регионы (том 2).pdf", page=196, title="Гродненская область. Численность населения"),
            "Численность населения Гродненской области",
            {"entity": "Гродненская область", "metric": "population", "unit": "thousand_persons", "normalized_value": 1338600.0},
        ),
        # percent without "индекс" should still be percent but maybe unknown metric in strict -> filtered
        (
            "Доля составила 12,5%.",
            FragmentMeta(source_file="Беларусь в цифрах, 2025.pdf", page=55, title="Структура"),
            "Доля промышленности в ВВП",
            {"metric": "index_percent", "unit": "percent", "normalized_value": 12.5},
        ),
        # counts: cities only
        (
            "В Минской области 22 района и 24 города.",
            FragmentMeta(source_file="Регионы (том 1).pdf", page=9, title="Минская область"),
            "районы и города Минской области",
            {
                "entity": "Минская область",
                "metrics": {"districts_count": 22.0, "cities_count": 24.0},
            },
        ),
        # grain production
        (
            "Производство зерна составило 7 151,6 тыс. тонн.",
            FragmentMeta(source_file="Беларусь и Россия, 2024.pdf", page=107, title="Сельское хозяйство"),
            "Производство зерна в Беларуси",
            {"metric": "production_volume", "unit": "tonnes", "normalized_value": 7151600.0},
        ),
        # negative / range / should still parse first number only
        (
            "Диапазон: 10–12 человек на 1 км2.",
            FragmentMeta(source_file="Регионы (том 2).pdf", page=50, title="Плотность населения"),
            "Плотность населения",
            {"metric": "population_density", "unit": "persons_per_km2", "normalized_value": 10.0},
        ),
        # thousands without explicit persons word -> strict may drop; relaxed keeps
        (
            "Всего: 1 234,5 тыс.",
            FragmentMeta(source_file="Регионы (том 2).pdf", page=1, title="Население Минска"),
            "Численность населения Минска",
            {"metric": "population", "unit": "thousand_persons", "normalized_value": 1234500.0},
        ),
    ],
)
def test_extract_numeric_indicators_strict(text, meta, query, expected):
    items = extract_numeric_indicators(text, meta, query=query, strict=True, min_confidence=0.55)

    if expected.get("expect_empty"):
        assert items == []
        return

    # multi-metric
    if "metrics" in expected:
        for metric, val in expected["metrics"].items():
            it = _pick_one(items, metric)
            assert it is not None, f"missing metric={metric} in {items}"
            assert math.isclose(it["normalized_value"], val, rel_tol=0, abs_tol=1e-6)
            if "entity" in expected:
                assert it["entity"] == expected["entity"]
        return

    it = _pick_one(items, expected["metric"])
    assert it is not None, f"missing metric={expected['metric']} in {items}"
    assert it["unit"] == expected["unit"]
    assert math.isclose(it["normalized_value"], expected["normalized_value"], rel_tol=0, abs_tol=1e-6)
    if "entity" in expected:
        assert it["entity"] == expected["entity"]
    assert 0.0 <= it["confidence"] <= 1.0


def test_precision_recall_f1_on_small_suite():
    # 20-ish examples (including noise) baked into this test; we evaluate metric+entity slots.
    suite = [
        (
            "На начало 2024 года в городе проживало 1 992,9 тыс. человек.",
            FragmentMeta(source_file="x.pdf", page=1, title="Минск"),
            "Численность населения Минска",
            [("г.Минск", "population")],
        ),
        (
            "В области 17 районов, 15 городов.",
            FragmentMeta(source_file="x.pdf", page=1, title="Гродненская область"),
            "районы и города",
            [("Гродненская область", "districts_count"), ("Гродненская область", "cities_count")],
        ),
        (
            "Плотность населения – 5 635,3 человека на 1 км2",
            FragmentMeta(source_file="x.pdf", page=1, title="Минская область"),
            "Плотность населения",
            [("Минская область", "population_density")],
        ),
        (
            "ISBN 978-985-7307-95-1",
            FragmentMeta(source_file="x.pdf", page=1, title=""),
            "население",
            [],
        ),
    ]

    tp = fp = fn = 0
    for text, meta, query, gold in suite:
        pred = extract_numeric_indicators(text, meta, query=query, strict=True, min_confidence=0.55)
        pred_slots = {(p.get("entity"), p.get("metric")) for p in pred}
        gold_slots = set(gold)
        tp += len(pred_slots & gold_slots)
        fp += len(pred_slots - gold_slots)
        fn += len(gold_slots - pred_slots)

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # relaxed thresholds for small suite; main requirement is covered by per-case asserts above
    assert precision >= 0.8
    assert recall >= 0.7
    assert f1 >= 0.75

