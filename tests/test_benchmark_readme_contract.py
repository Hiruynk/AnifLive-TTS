from __future__ import annotations

import ast
import importlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = json.loads(
    (ROOT / "benchmarks" / "README_BENCHMARK_SUMMARY.json").read_text(encoding="utf-8")
)
V1_5_SUMMARY = json.loads(
    (ROOT / "benchmarks" / "V1_5_BENCHMARK_SUMMARY.json").read_text(encoding="utf-8")
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
    raise AssertionError(f"Unknown benchmark metric: {key}")


def _import_module(import_root: Path, module_name: str):
    import sys

    sys.path.insert(0, str(import_root))
    try:
        return importlib.import_module(module_name)
    finally:
        sys.path.remove(str(import_root))


def test_historical_metric_contract_and_canonical_renderer_without_running_benchmark(
    tmp_path,
) -> None:
    canonical = SUMMARY["session_distribution"]
    assert METRIC_KEYS == tuple(canonical)[: len(METRIC_KEYS)]
    assert len(METRIC_KEYS) == 14
    table_rows = SUMMARY["table"]
    assert tuple(row["key"] for row in table_rows) == METRIC_KEYS
    aggregates = {}
    for row in table_rows:
        values = canonical[row["key"]]
        assert row["median"] == values["session_median"]
        assert row["range_min"] == values["best"]
        assert row["range_max"] == values["worst"]
        aggregates[row["key"]] = (values["session_median"], values["best"], values["worst"])

    benchmark_readme = _import_module(ROOT / "scripts", "benchmark_readme")
    for locale in METRIC_LABELS:
        rendered = benchmark_readme.render_markdown(
            {
                key: benchmark_readme.Aggregate(
                    median=median,
                    range_min=best,
                    range_max=worst,
                )
                for key, (median, best, worst) in aggregates.items()
            },
            locale=locale,
            sessions=SUMMARY["methodology"]["sessions"],
        )
        path = tmp_path / f"README_{locale}.md"
        path.write_text(rendered, encoding="utf-8")
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


def test_current_benchmark_document_and_readme_summaries_match_v1_5_without_running_benchmark() -> None:
    release_preflight = _import_module(ROOT, "scripts.release_preflight")
    result = release_preflight.check_readme_benchmark_values(ROOT, V1_5_SUMMARY)

    assert result["status"] == "passed", result["findings"]
    assert result["detailed_rows_checked"] == 4
    assert result["comparison_rows_checked"] == 36
    assert result["rows_checked"] == 40
    assert result["per_condition_samples"] == 1000
    assert result["measured_total"] == 3000


def test_compact_readme_benchmark_value_mutation_is_reported(tmp_path) -> None:
    release_preflight = _import_module(ROOT, "scripts.release_preflight")
    readme_paths = ("README.md", "README_ZH_HK.md", "README_ZH_CN.md")
    for relative_path in (*readme_paths, "docs/BENCHMARKS.md"):
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            (ROOT / relative_path).read_text(encoding="utf-8"), encoding="utf-8"
        )

    readme_path = tmp_path / "README.md"
    lines = readme_path.read_text(encoding="utf-8").splitlines()
    row_index = next(
        index
        for index, line in enumerate(lines)
        if "v2proplus" in line.casefold()
        and "V1_5_BENCHMARK_SUMMARY.json" in line
        and "75.70" in line
    )
    lines[row_index] = lines[row_index].replace("75.70", "75.71", 1)
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = release_preflight.check_readme_benchmark_values(tmp_path, V1_5_SUMMARY)

    assert result["status"] == "failed"
    assert any(
        finding["code"] == "readme_benchmark_value_mismatch"
        and finding.get("path") == "README.md"
        and finding.get("workload") == "windows_v2proplus"
        and finding.get("metric") == "audible_ttfa_p50_ms"
        for finding in result["findings"]
    )
