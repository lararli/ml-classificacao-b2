"""Promotes a model to production. Gets best params from MLflow if search was used."""

import sys
import logging

logging.getLogger("mlflow").setLevel(logging.CRITICAL)

import mlflow
import yaml

mlflow.set_tracking_uri("sqlite:///mlflow.db")

if len(sys.argv) < 2:
    print("usage: python promote.py MODEL_NAME")
    sys.exit(1)

model_name = sys.argv[1]

with open("config/experiments_test.yaml") as f:
    test_config = yaml.safe_load(f)

model_entry = None
for m in test_config["models"]:
    if m["name"] == model_name:
        model_entry = m.copy()
        break

if not model_entry:
    print(f"model '{model_name}' not found in experiments_test.yaml")
    print(f"available: {[m['name'] for m in test_config['models']]}")
    sys.exit(1)

if model_entry.get("search_params"):
    runs = mlflow.search_runs(
        experiment_names=["loan_approval_experimentation"],
        filter_string=f"tags.mlflow.runName = '{model_name}'",
        order_by=["metrics.f1_score DESC"],
    )
    if not runs.empty:
        best_run = runs.iloc[0]
        for col in runs.columns:
            if col.startswith("params.best_"):
                key = col.replace("params.best_", "")
                val = best_run[col]
                if val is not None and str(val) != "nan":
                    try:
                        val = int(val)
                    except (ValueError, TypeError):
                        try:
                            val = float(val)
                        except (ValueError, TypeError):
                            if val == "None":
                                val = None
                    model_entry["params"][key] = val
        print(f"PROMOTE: using optimized params from MLflow")

    del model_entry["search_params"]

if "reduction" in model_entry:
    del model_entry["reduction"]

with open("config/experiments_prod.yaml", "w") as f:
    f.write("# experiments_prod.yaml\n# Production model.\n\n")
    yaml.dump({"models": [model_entry]}, f, default_flow_style=False, sort_keys=False)

print(f"PROMOTE: {model_name} promoted to production\n")
with open("config/experiments_prod.yaml") as f:
    print(f.read())
print("next: make prod")
