"""Compares current production run with previous. Detects degradation."""

import logging
import sys

logging.getLogger("mlflow").setLevel(logging.CRITICAL)

import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")


def analyze():
    mlflow.set_experiment("loan_approval_production")
    runs = mlflow.search_runs(order_by=["start_time DESC"])

    if runs.empty:
        print("no production runs found. Run 'make prod' first.")
        return

    if len(runs) < 2:
        current = runs.iloc[0]
        print(f"\nPOST-DEPLOY ANALYSIS")
        print(f"  current: {current.get('tags.mlflow.runName', '?')}")
        print(f"  execution_id: {current.get('tags.execution_id', '?')}")
        print(f"  f1: {current.get('metrics.f1_score', 0):.4f}")
        print(f"  no previous run to compare. This is the first deployment.")
        return

    runs = runs.drop_duplicates(subset=["tags.execution_id"], keep="first")
    runs = runs.sort_values("start_time", ascending=False)

    if len(runs) < 2:
        print("  only one execution_id found. Need at least 2 monthly runs to compare.")
        return

    current = runs.iloc[0]
    previous = runs.iloc[1]
    cur_eid = current.get("tags.execution_id", "?")
    prev_eid = previous.get("tags.execution_id", "?")

    metrics = ["f1_score", "accuracy", "precision", "recall", "train_time_seconds"]

    print(f"\nPOST-DEPLOY ANALYSIS")
    print(f"  {'metric':<20} {'previous (' + prev_eid + ')':>18} {'current (' + cur_eid + ')':>18} {'change':>12}")

    degraded = False
    for m in metrics:
        prev_val = previous.get(f"metrics.{m}", 0)
        cur_val = current.get(f"metrics.{m}", 0)
        diff = cur_val - prev_val
        sign = "+" if diff > 0 else ""

        if m == "train_time_seconds":
            change = f"{sign}{diff:.1f}s"
        else:
            change = f"{sign}{diff:.4f}"
            if m == "f1_score" and diff < -0.02:
                change += " !"
                degraded = True

        print(f"  {m:<20} {prev_val:>18.4f} {cur_val:>18.4f} {change:>12}")

    if degraded:
        print(f"\n  STATUS: DEGRADED. f1 dropped more than 0.02. Investigate drift.")
        print(f"  ACTION: run 'make test' to re-evaluate all models.")
    else:
        print(f"\n  STATUS: STABLE. Model performing within expected range.")


if __name__ == "__main__":
    analyze()
