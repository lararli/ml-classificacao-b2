"""Compare models from MLflow. Shows latest run per model, exports CSV."""

import sys
from pathlib import Path

import mlflow
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db")

ALIASES = {
    "test": "loan_approval_experimentation",
    "prod": "loan_approval_production",
}

arg = sys.argv[1] if len(sys.argv) > 1 else "test"
experiment = ALIASES.get(arg, arg)
execution_id = sys.argv[2] if len(sys.argv) > 2 else None

filter_str = f"tags.execution_id = '{execution_id}'" if execution_id else ""
runs = mlflow.search_runs(experiment_names=[experiment], filter_string=filter_str, order_by=["start_time DESC"])

if runs.empty:
    print(f"no runs found in {experiment}")
    sys.exit(1)

runs = runs.drop_duplicates(subset=["tags.mlflow.runName"], keep="first")
runs = runs.sort_values("metrics.f1_score", ascending=False)

table = pd.DataFrame({
    "model": runs["tags.mlflow.runName"],
    "f1": runs["metrics.f1_score"].round(4),
    "accuracy": runs["metrics.accuracy"].round(4),
    "precision": runs["metrics.precision"].round(4),
    "recall": runs["metrics.recall"].round(4),
    "gap": runs["metrics.f1_gap"].round(4),
    "time_s": runs["metrics.train_time_seconds"].round(1),
    "run_id": runs["run_id"],
})

if "metrics.auc_roc" in runs.columns:
    table["auc_roc"] = runs["metrics.auc_roc"].round(4)
if "tags.execution_id" in runs.columns:
    table["execution_id"] = runs["tags.execution_id"]

print(f"\nEXPERIMENT: {experiment}")
if execution_id:
    print(f"EXECUTION: {execution_id}")
print(f"MODELS: {len(table)}\n")

print(table.drop(columns=["run_id"]).to_string(index=False))

best = table.iloc[0]
print(f"\nBEST: {best['model']}  f1={best['f1']}  run_id={best['run_id'][:16]}...")
print(f"\nnext: make promote MODEL={best['model']}")

output_dir = Path("outputs/results")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / f"comparison_{experiment.split('_')[-1]}.csv"
table.to_csv(csv_path, index=False)
print(f"saved: {csv_path}")
