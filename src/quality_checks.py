"""Validates data quality against expectations defined in quality.yaml."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from src.config import QualityConfig, ColumnExpectation


def run_quality_checks(df: pd.DataFrame, quality: QualityConfig) -> list[dict]:
    """Runs all table and column expectations. Returns list of results."""
    results = []

    count = len(df)
    results.append({
        "type": "table_row_count", "column": None,
        "passed": quality.table.min_rows <= count <= quality.table.max_rows,
        "detail": f"expected [{quality.table.min_rows}, {quality.table.max_rows}], got {count}",
    })

    for col_name, exp in quality.columns.items():
        if col_name not in df.columns:
            results.append({"type": "column_exists", "column": col_name, "passed": False, "detail": "not found"})
            continue

        col = df[col_name]
        if not exp.nullable:
            n = col.isnull().sum()
            results.append({"type": "not_null", "column": col_name, "passed": n == 0, "detail": f"{n} nulls"})
        if exp.min_value is not None:
            n = (col < exp.min_value).sum()
            results.append({"type": "min_value", "column": col_name, "passed": n == 0, "detail": f"{n} below {exp.min_value}"})
        if exp.max_value is not None:
            n = (col > exp.max_value).sum()
            results.append({"type": "max_value", "column": col_name, "passed": n == 0, "detail": f"{n} above {exp.max_value}"})
        if exp.allowed_values is not None:
            unexpected = set(col.dropna().unique()) - set(exp.allowed_values)
            results.append({"type": "allowed_values", "column": col_name, "passed": len(unexpected) == 0, "detail": f"unexpected: {unexpected}" if unexpected else "ok"})

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print(f"[quality] {passed} passed, {failed} failed ({len(results)} checks)")
    return results


def save_report(results: list[dict], output_dir: str = "outputs/quality_reports", execution_id: str = "") -> Path:
    """Saves quality report as JSON."""
    if execution_id:
        output_dir = f"{output_dir}/{execution_id}"
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = path / f"quality_report_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total_checks": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "failed": sum(1 for r in results if not r["passed"]),
            "results": results,
        }, f, indent=2, ensure_ascii=False, default=str)

    print(f"[quality] report: {filename}")
    return filename
