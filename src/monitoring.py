"""Drift detection: compares train vs production data statistics."""

import pandas as pd


def detect_drift(df_train: pd.DataFrame, df_prod: pd.DataFrame,
                 numerical_cols: list[str], threshold: float = 10.0) -> list[dict]:
    """Compares mean of each numerical column. Flags if variation > threshold%."""
    results = []
    for col in numerical_cols:
        if col not in df_train.columns or col not in df_prod.columns:
            continue
        mean_train = df_train[col].mean()
        mean_prod = df_prod[col].mean()
        var = abs(mean_prod - mean_train) / abs(mean_train) * 100 if mean_train != 0 else 0.0
        results.append({"feature": col, "mean_train": round(mean_train, 4),
                        "mean_production": round(mean_prod, 4), "variation_pct": round(var, 2),
                        "drift": var > threshold})

    n = sum(1 for r in results if r["drift"])
    print(f"MONITORING: {n} features with drift (threshold={threshold}%) out of {len(results)}")
    return results


def check_new_categories(df_train: pd.DataFrame, df_prod: pd.DataFrame,
                         categorical_cols: list[str]) -> list[dict]:
    """Checks if production data has categories the model never saw."""
    results = []
    for col in categorical_cols:
        if col not in df_train.columns or col not in df_prod.columns:
            continue
        new = set(df_prod[col].dropna().unique()) - set(df_train[col].dropna().unique())
        results.append({"feature": col, "new_categories": list(new) if new else [], "has_new": len(new) > 0})
        if new:
            print(f"MONITORING: ALERT: {col} has new categories: {new}")
    return results
