"""
Post-deploy analysis: compares current production run with previous.
Shows if the model improved, degraded, or stayed stable.

Usage: python scripts/post_deploy.py
"""

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
        print(f"\n{'='*70}")
        print(f"  POST-DEPLOY ANALYSIS")
        print(f"{'='*70}")
        print(f"\n  current: {current.get('tags.mlflow.runName', '?')}")
        print(f"  execution_id: {current.get('tags.execution_id', '?')}")
        print(f"  f1: {current.get('metrics.f1_score', 0):.4f}")
        print(f"\n  no previous run to compare. This is the first deployment.")
        return

    # Deduplicate by execution_id, keep latest
    runs = runs.drop_duplicates(subset=["tags.execution_id"], keep="first")
    runs = runs.sort_values("start_time", ascending=False)

    if len(runs) < 2:
        print("  only one execution_id found. Need at least 2 monthly runs to compare.")
        return

    current = runs.iloc[0]
    previous = runs.iloc[1]

    cur_name = current.get("tags.mlflow.runName", "?")
    cur_eid = current.get("tags.execution_id", "?")
    prev_name = previous.get("tags.mlflow.runName", "?")
    prev_eid = previous.get("tags.execution_id", "?")

    metrics = ["f1_score", "accuracy", "precision", "recall", "train_time_seconds"]

    print(f"\n{'='*70}")
    print(f"  POST-DEPLOY ANALYSIS")
    print(f"{'='*70}")
    print(f"\n  {'metric':<20} {'previous':>15} {'current':>15} {'change':>15}")
    print(f"  {'':─<20} {'(' + prev_eid + ')':>15} {'(' + cur_eid + ')':>15} {'':>15}")
    print(f"  {'─'*65}")

    degraded = False
    for m in metrics:
        prev_val = previous.get(f"metrics.{m}", 0)
        cur_val = current.get(f"metrics.{m}", 0)
        diff = cur_val - prev_val

        if m == "train_time_seconds":
            sign = "+" if diff > 0 else ""
            change = f"{sign}{diff:.1f}s"
        else:
            sign = "+" if diff > 0 else ""
            change = f"{sign}{diff:.4f}"
            if m == "f1_score" and diff < -0.02:
                change += " ⚠"
                degraded = True

        print(f"  {m:<20} {prev_val:>15.4f} {cur_val:>15.4f} {change:>15}")

    print(f"\n  {'─'*65}")

    if degraded:
        print(f"  STATUS: ⚠ DEGRADED - f1 dropped more than 0.02. Investigate drift.")
        print(f"  ACTION: run 'make test' to re-evaluate all models.")
    else:
        print(f"  STATUS: STABLE - model performing within expected range.")
        print(f"  ACTION: no action needed. Continue monitoring.")


if __name__ == "__main__":
    analyze()
