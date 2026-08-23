from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads(
    (ROOT / "benchmarks" / "README_BENCHMARK_SUMMARY.json").read_text(encoding="utf-8")
)


def _script_constant(name: str):
    tree = ast.parse((ROOT / "scripts" / "benchmark_readme.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"Missing static benchmark constant: {name}")


METRIC_KEYS = tuple(_script_constant("METRIC_KEYS"))
METRIC_LABELS = _script_constant("METRIC_LABELS")


def _first_table(path: Path) -> dict[str, tuple[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("| Metric |") or line.startswith("| 指標 |") or line.startswith("| 指标 |"):
            rows: dict[str, tuple[str, str]] = {}
            for row in lines[index + 2 :]:
                if not row.startswith("|"):
                    break
                cells = [cell.strip() for cell in row.strip("|").split("|")]
                if len(cells) != 3:
                    raise AssertionError(f"Unexpected benchmark row in {path.name}: {row}")
                rows[cells[0]] = (cells[1], cells[2])
            return rows
    raise AssertionError(f"Benchmark table not found in {path.name}")


def _numbers(cell: str) -> list[float]:
    return [
        float(value.replace(",", ""))
        for value in re.findall(r"\d[\d,]*(?:\.\d+)?", cell)
    ]


def _tolerance(key: str) -> float:
    if key.endswith("_ms"):
        return 0.000501
    if key == "wall_rtf_p50":
        return 0.000000501
    if key.startswith("gpu_busy_"):
        return 0.050001
    if key == "vram_p50_mib":
        return 0.500001
    raise AssertionError(f"Unknown benchmark metric: {key}")


def test_readme_tables_match_canonical_benchmark_summary_without_running_benchmark() -> None:
    canonical = SUMMARY["session_distribution"]
    assert METRIC_KEYS == tuple(canonical)[: len(METRIC_KEYS)]
    assert len(METRIC_KEYS) == 9
    readmes = {
        "en": ROOT / "README.md",
        "zh-HK": ROOT / "README_ZH_HK.md",
        "zh-CN": ROOT / "README_ZH_CN.md",
    }
    for locale, path in readmes.items():
        table = _first_table(path)
        assert list(table) == list(METRIC_LABELS[locale])
        for key, label in zip(METRIC_KEYS, METRIC_LABELS[locale], strict=True):
            values = canonical[key]
            headline, range_text = table[label]
            assert headline.startswith("**") and headline.endswith("**")
            assert len(_numbers(headline)) == 1
            assert len(_numbers(range_text)) == 2
            tolerance = _tolerance(key)
            assert abs(_numbers(headline)[0] - values["session_median"]) <= tolerance
            assert abs(_numbers(range_text)[0] - values["best"]) <= tolerance
            assert abs(_numbers(range_text)[1] - values["worst"]) <= tolerance
