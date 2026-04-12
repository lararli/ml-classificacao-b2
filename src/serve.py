"""Inference service. Loads model from MLflow and serves predictions."""

import json
import logging

import mlflow
import mlflow.sklearn
import pandas as pd

logging.getLogger("mlflow").setLevel(logging.CRITICAL)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def load_model(run_id: str):
    """Loads a saved model by run_id."""
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    print(f"SERVE: model loaded from run {run_id}")
    return model


def predict(model, preprocessor, input_data: dict) -> dict:
    """Predicts for a single client. Returns label + probability."""
    X = preprocessor.transform(pd.DataFrame([input_data]))
    prediction = int(model.predict(X)[0])
    result = {"prediction": prediction, "label": "APPROVED" if prediction == 1 else "REJECTED"}
    if hasattr(model, "predict_proba"):
        result["probability"] = round(float(model.predict_proba(X)[0][1]), 4)
    return result


def run_service():
    """Interactive CLI service. Loads production model, accepts JSON input."""
    from src.config import DataConfig, PipelineConfig
    from src.preprocessing import create_preprocessor, load_and_split
    from pathlib import Path

    data = DataConfig.load()
    pipeline = PipelineConfig.load()

    prod_dir = Path(pipeline.paths.processed_data_dir) / "production"
    run_path = ""
    if prod_dir.exists():
        runs = sorted(prod_dir.iterdir(), reverse=True)
        run_path = f"production/{runs[0].name}" if runs else ""

    X_train, _, _, _ = load_and_split(data, pipeline, run_path)
    preprocessor = create_preprocessor(data)
    preprocessor.fit(X_train)

    mlflow.set_experiment("loan_approval_production")
    runs = mlflow.search_runs(order_by=["start_time DESC"])
    if runs.empty:
        print("SERVE: no production models. Run 'make prod' first.")
        return

    best = runs.iloc[0]
    model = load_model(best["run_id"])
    name = best.get("tags.mlflow.runName", "?")
    f1 = best.get("metrics.f1_score", 0)

    print(f"\n{'='*60}")
    print(f"  INFERENCE SERVICE")
    print(f"  model: {name}  |  f1: {f1:.4f}")
    print(f"{'='*60}")
    print(f"\nPaste JSON (inline), or 'quit' to exit.\n")
    print('example:')
    print('{"person_age": 29, "person_income": 25018, "person_emp_exp": 7, "loan_amnt": 8069, "loan_int_rate": 12.32, "loan_percent_income": 0.32, "cb_person_cred_hist_length": 10, "credit_score": 650, "person_gender": "male", "person_education": "Associate", "person_home_ownership": "RENT", "loan_intent": "MEDICAL", "previous_loan_defaults_on_file": "No"}')

    while True:
        print("\n> ", end="")
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break
        if line.strip().lower() in ("quit", "exit", "q"):
            break
        try:
            result = predict(model, preprocessor, json.loads(line))
            prob = f"  (probability: {result['probability']})" if "probability" in result else ""
            print(f"\n  {result['label']}{prob}")
        except json.JSONDecodeError:
            print("  invalid JSON")
        except Exception as e:
            print(f"  error: {e}")


if __name__ == "__main__":
    run_service()
