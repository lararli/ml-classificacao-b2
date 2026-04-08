"""Compares models and calculates financial impact."""

import logging

import mlflow
import pandas as pd

logging.getLogger("mlflow").setLevel(logging.CRITICAL)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def compare_models(experiment_name: str, execution_id: str | None = None) -> pd.DataFrame:
    """Returns DataFrame with all runs ordered by F1."""
    filter_str = f"tags.execution_id = '{execution_id}'" if execution_id else ""
    runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string=filter_str, order_by=["metrics.f1_score DESC"])

    if runs.empty:
        print("EVALUATE: no runs found")
        return pd.DataFrame()

    cols = {"tags.mlflow.runName": "model", "tags.execution_id": "execution_id",
            "metrics.accuracy": "accuracy", "metrics.precision": "precision",
            "metrics.recall": "recall", "metrics.f1_score": "f1",
            "metrics.f1_train": "f1_train", "metrics.f1_gap": "gap",
            "metrics.train_time_seconds": "time_s"}

    result = runs[[k for k in cols if k in runs.columns]].rename(columns=cols)
    if "metrics.auc_roc" in runs.columns:
        result["auc_roc"] = runs["metrics.auc_roc"]

    print(f"EVALUATE: {len(result)} models found")
    return result


def financial_analysis(y_test, predictions: dict, X_test: pd.DataFrame) -> pd.DataFrame:
    """Calculates FP cost (loan lost) and FN cost (interest lost) per model."""
    y_arr = y_test.values if hasattr(y_test, "values") else y_test
    results = []

    for name, y_pred in predictions.items():
        fp = (y_pred == 1) & (y_arr == 0)
        fn = (y_pred == 0) & (y_arr == 1)
        fp_loss = X_test.loc[fp, "loan_amnt"].sum() if "loan_amnt" in X_test.columns else 0
        fn_loss = (X_test.loc[fn, "loan_amnt"] * X_test.loc[fn, "loan_int_rate"] / 100).sum() if "loan_int_rate" in X_test.columns else 0
        results.append({"model": name, "fp_count": int(fp.sum()), "fn_count": int(fn.sum()),
                        "fp_loss": float(fp_loss), "fn_loss": float(fn_loss), "total_impact": float(fp_loss + fn_loss)})

    print(f"EVALUATE: financial analysis for {len(results)} models")
    return pd.DataFrame(results).sort_values("total_impact")
