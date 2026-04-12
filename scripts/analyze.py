"""Analyze results for an experiment. Storytelling view with data samples, KPIs, and CSV export."""

import logging
import sys
from pathlib import Path

# ensure project root is in path (needed to load models that reference src.*)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("mlflow").setLevel(logging.CRITICAL)

import mlflow
import pandas as pd
from src.colors import ok, fail, warn, info, bold, header

mlflow.set_tracking_uri("sqlite:///mlflow.db")

ALIASES = {
    "test": "loan_approval_experimentation",
    "prod": "loan_approval_production",
}

arg = sys.argv[1] if len(sys.argv) > 1 else "test"
experiment = ALIASES.get(arg, arg)
mode = "experimentation" if "experimentation" in experiment else "production"

runs = mlflow.search_runs(experiment_names=[experiment], order_by=["start_time DESC"])

if runs.empty:
    print(fail(f"no runs found in {experiment}"))
    sys.exit(1)

# keep only latest run per execution_id + model
runs = runs.drop_duplicates(subset=["tags.execution_id", "tags.mlflow.runName"], keep="first")

# ── 1. THE DATA ──
print(f"\n{'=' * 64}")
print(f"  {bold('ANALYSIS REPORT:')} {experiment}")
print(f"{'=' * 64}")

# find the most recent parquet to show data samples
exec_ids = sorted(runs["tags.execution_id"].dropna().unique(), reverse=True)
latest_eid = exec_ids[0] if exec_ids else None

data_path = Path(f"data/processed/{mode}/{latest_eid}/loan_data.parquet")
if not data_path.exists():
    # fallback to raw
    data_path = Path("data/raw/loan_data.csv")

if data_path.exists():
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    total = len(df)
    approved = (df["loan_status"] == 1).sum()
    rejected = (df["loan_status"] == 0).sum()

    print(header("  1. THE DATA"))
    print(f"  {'─' * 58}")
    print(f"  source:    taweilo/loan-approval-classification-data")
    print(f"  rows:      {total:,}")
    print(f"  features:  {len(df.columns) - 1} ({df.select_dtypes('number').shape[1] - 1} numeric, {df.select_dtypes('object').shape[1]} categorical)")
    print(f"  target:    loan_status")
    print(f"  approved:  {ok(f'{approved:,} ({approved/total:.1%})')}")
    print(f"  rejected:  {fail(f'{rejected:,} ({rejected/total:.1%})')}")

    # sample approved and rejected
    sample_approved = df[df["loan_status"] == 1].sample(5, random_state=42)
    sample_rejected = df[df["loan_status"] == 0].sample(5, random_state=42)

    display_cols = [
        "person_age", "person_income", "person_emp_exp", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "credit_score",
        "person_education", "person_home_ownership", "loan_intent",
        "previous_loan_defaults_on_file", "loan_status",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    samples = pd.concat([sample_approved, sample_rejected])[display_cols].copy()
    samples["loan_status"] = samples["loan_status"].map({1: "APPROVED", 0: "REJECTED"})
    samples = samples.rename(columns={
        "person_age": "age",
        "person_income": "income",
        "person_emp_exp": "exp_yrs",
        "loan_amnt": "loan",
        "loan_int_rate": "rate%",
        "loan_percent_income": "loan/inc",
        "credit_score": "score",
        "person_education": "education",
        "person_home_ownership": "housing",
        "loan_intent": "intent",
        "previous_loan_defaults_on_file": "defaults",
        "loan_status": "status",
    })
    samples["income"] = samples["income"].apply(lambda x: f"${x:,.0f}")
    samples["loan"] = samples["loan"].apply(lambda x: f"${x:,.0f}")
    samples["age"] = samples["age"].astype(int)
    samples["exp_yrs"] = samples["exp_yrs"].astype(int)

    print(f"\n  sample data:")
    table_str = samples.to_string(index=False)
    for line in table_str.split("\n"):
        print(f"    {line}")

    # key stats
    print(f"\n  data highlights:")
    print(f"    avg income:       ${df['person_income'].mean():,.0f}")
    print(f"    avg loan amount:  ${df['loan_amnt'].mean():,.0f}")
    print(f"    avg credit score: {df['credit_score'].mean():.0f}")
    print(f"    avg interest rate:{df['loan_int_rate'].mean():>6.1f}%")
else:
    print(f"\n  {warn('(data file not found, skipping data samples)')}")

# ── 2. THE EXPERIMENT ──
print(header("  2. THE EXPERIMENT"))
print(f"  {'─' * 58}")
print(f"  mode:           {mode}")
print(f"  total runs:     {len(runs)}")
print(f"  executions:     {len(exec_ids)}")
print(f"  train/test:     80% / 20% (stratified)")

# ── 3. RESULTS PER EXECUTION ──
print(header("  3. RESULTS"))
print(f"  {'─' * 58}")

all_results = []

for eid in exec_ids:
    subset = runs[runs["tags.execution_id"] == eid].sort_values("metrics.f1_score", ascending=False)
    best = subset.iloc[0]

    print(f"\n  execution: {eid}  |  models: {len(subset)}")
    print(f"  {'model':<28} {'f1':>8} {'acc':>8} {'prec':>8} {'rec':>8}")

    for rank, (_, row) in enumerate(subset.iterrows()):
        name = row.get("tags.mlflow.runName", "?")
        f1 = row.get("metrics.f1_score", 0)
        acc = row.get("metrics.accuracy", 0)
        prec = row.get("metrics.precision", 0)
        rec = row.get("metrics.recall", 0)

        # color the best model green, worst red, rest default
        if rank == 0:
            line = ok(f"  {name:<28} {f1:>7.2%} {acc:>7.2%} {prec:>7.2%} {rec:>7.2%}")
        elif f1 < 0.01:
            line = fail(f"  {name:<28} {f1:>7.2%} {acc:>7.2%} {prec:>7.2%} {rec:>7.2%}")
        else:
            line = f"  {name:<28} {f1:>7.2%} {acc:>7.2%} {prec:>7.2%} {rec:>7.2%}"
        print(line)

        all_results.append({
            "execution_id": eid,
            "model": name,
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_gap": round(row.get("metrics.f1_gap", 0), 4),
            "train_time_s": round(row.get("metrics.train_time_seconds", 0), 1),
            "run_id": row.get("run_id", ""),
        })

    # ── 4. KPIs ──
    best_f1 = best.get("metrics.f1_score", 0)
    best_acc = best.get("metrics.accuracy", 0)
    best_prec = best.get("metrics.precision", 0)
    best_rec = best.get("metrics.recall", 0)
    best_gap = best.get("metrics.f1_gap", 0)
    best_name = best.get("tags.mlflow.runName", "?")
    best_time = best.get("metrics.train_time_seconds", 0)

    avg_f1 = subset["metrics.f1_score"].mean()
    worst_f1 = subset["metrics.f1_score"].min()
    worst_name = subset.loc[subset["metrics.f1_score"].idxmin(), "tags.mlflow.runName"]

    print(header(f"  4. KPIs (execution {eid})"))
    print(f"  {'─' * 58}")
    print(f"  best model:              {bold(best_name)}")
    print(f"  best f1:                 {ok(f'{best_f1:.2%}')}")
    print(f"  best accuracy:           {best_acc:.2%}")
    print(f"  best precision:          {best_prec:.2%}  (false positives control)")
    print(f"  best recall:             {best_rec:.2%}  (missed approvals)")

    gap_color = warn if best_gap > 0.10 else ok
    gap_note = "(high - review regularization)" if best_gap > 0.10 else "(acceptable)"
    print(f"  overfitting gap:         {gap_color(f'{best_gap:.4f}')}  {gap_note}")

    print(f"  train time:              {best_time:.1f}s")
    print(f"  avg f1 (all models):     {avg_f1:.2%}")
    print(f"  worst model:             {fail(f'{worst_name} ({worst_f1:.2%})')}")
    print(f"  f1 spread:               {best_f1 - worst_f1:.2%}")

    # interpretation
    if best_f1 >= 0.85:
        status_color = ok
        status = "EXCELLENT"
        note = "model is production-ready with strong performance across all metrics."
    elif best_f1 >= 0.70:
        status_color = ok
        status = "GOOD"
        note = "model performs well. consider tuning hyperparameters or adding features to push above 85%."
    elif best_f1 >= 0.50:
        status_color = warn
        status = "MODERATE"
        note = "model needs improvement. review feature engineering, class balance, or try different algorithms."
    else:
        status_color = fail
        status = "POOR"
        note = "model is underperforming. check data quality, feature relevance, and pipeline correctness."

    print(f"\n  STATUS: {status_color(status)}")
    print(f"  {note}")

    if best_gap > 0.10:
        print(f"  {warn('warning:')} overfitting gap of {best_gap:.2%} suggests the model memorizes training data.")
        print(f"  consider: stronger regularization, more data, or simpler model.")

    if best_rec < 0.70:
        print(f"  {warn('warning:')} recall at {best_rec:.2%} means {1-best_rec:.1%} of good loans are being rejected.")

# ── 5. EXPORT ──
output_dir = Path("outputs/results")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / f"analysis_{mode}.csv"

results_df = pd.DataFrame(all_results)
results_df.to_csv(csv_path, index=False)

print(header("  5. EXPORTED"))
print(f"  {'─' * 58}")
print(f"  csv: {ok(str(csv_path))}")
print(f"  rows: {len(results_df)} (all models, all executions)")
print(f"  use this file to build dashboards or share results.")

# ── 6. INSIGHTS ──
if data_path.exists():
    print(header("  6. INSIGHTS"))
    print(f"  {'─' * 58}")

    # approved vs rejected profile comparison
    approved_df = df[df["loan_status"] == 1]
    rejected_df = df[df["loan_status"] == 0]

    print(f"\n  approved vs rejected profile:")
    print(f"  {'metric':<24} {'approved':>12} {'rejected':>12} {'diff':>12}")
    print(f"  {'─' * 54}")

    comparisons = [
        ("avg income", "person_income", "$"),
        ("avg loan amount", "loan_amnt", "$"),
        ("avg interest rate", "loan_int_rate", "%"),
        ("avg credit score", "credit_score", ""),
        ("avg loan/income", "loan_percent_income", ""),
        ("avg experience (yrs)", "person_emp_exp", ""),
        ("avg age", "person_age", ""),
    ]

    for label, col, prefix in comparisons:
        avg_app = approved_df[col].mean()
        avg_rej = rejected_df[col].mean()
        diff = avg_app - avg_rej
        sign = "+" if diff > 0 else ""

        if prefix == "$":
            print(f"  {label:<24} {prefix}{avg_app:>10,.0f} {prefix}{avg_rej:>10,.0f} {sign}{diff:>+10,.0f}")
        elif prefix == "%":
            print(f"  {label:<24} {avg_app:>11.1f}% {avg_rej:>11.1f}% {sign}{diff:>+10.1f}%")
        else:
            print(f"  {label:<24} {avg_app:>12.1f} {avg_rej:>12.1f} {sign}{diff:>+12.1f}")

    # default rate comparison
    app_defaults = (approved_df["previous_loan_defaults_on_file"] == "Yes").mean()
    rej_defaults = (rejected_df["previous_loan_defaults_on_file"] == "Yes").mean()
    print(f"\n  previous defaults:")
    print(f"    approved group: {ok(f'{app_defaults:.1%}')} had previous defaults")
    print(f"    rejected group: {fail(f'{rej_defaults:.1%}')} had previous defaults")

    # top reasons for rejection (strongest separators)
    print(f"\n  {bold('key rejection drivers:')}")
    separators = []
    for col in ["loan_int_rate", "loan_percent_income", "credit_score", "person_income", "person_emp_exp"]:
        avg_app = approved_df[col].mean()
        avg_rej = rejected_df[col].mean()
        if avg_app != 0:
            pct_diff = abs(avg_app - avg_rej) / abs(avg_app) * 100
            direction = "higher" if avg_rej > avg_app else "lower"
            separators.append((col, pct_diff, direction))

    separators.sort(key=lambda x: x[1], reverse=True)
    for col, pct, direction in separators[:3]:
        print(f"    - rejected loans have {warn(f'{pct:.0f}%')} {direction} {col.replace('_', ' ')}")

# ── 7. MODEL DECISION FACTORS ──
# Load best model and extract feature importances
best_run = runs.sort_values("metrics.f1_score", ascending=False).iloc[0]
best_run_id = best_run["run_id"]
best_model_name = best_run.get("tags.mlflow.runName", "?")

print(header("  7. WHY DOES THE MODEL APPROVE OR REJECT?"))
print(f"  {'─' * 58}")
print(f"  model: {bold(best_model_name)} (run {best_run_id[:12]}...)")

try:
    model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

    # handle Pipeline (reduction models wrap clf inside a pipeline)
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(clf, "feature_importances_"):
        from src.config import DataConfig
        from src.preprocessing import create_preprocessor, get_feature_names

        data_cfg = DataConfig.load()
        preprocessor = create_preprocessor(data_cfg)

        # fit preprocessor to get feature names
        if data_path.exists():
            X_fit = df[data_cfg.all_features]
            preprocessor.fit(X_fit)
            feature_names = get_feature_names(preprocessor, data_cfg)
        else:
            feature_names = [f"feature_{i}" for i in range(len(clf.feature_importances_))]

        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False)

        imp_df["importance_pct"] = imp_df["importance"] / imp_df["importance"].sum() * 100

        print(f"\n  feature importance ranking:")
        print(f"  {'feature':<36} {'weight':>8} {'impact':>8}")
        print(f"  {'─' * 54}")

        for rank, (_, row) in enumerate(imp_df.head(10).iterrows()):
            bar = "#" * int(row["importance_pct"] / 2)
            if rank < 3:
                print(f"  {ok(f'{row[\"feature\"]:<36}')} {row['importance']:>8.4f} {row['importance_pct']:>6.1f}%  {ok(bar)}")
            else:
                print(f"  {row['feature']:<36} {row['importance']:>8.4f} {row['importance_pct']:>6.1f}%  {bar}")

        top3 = imp_df.head(3)["feature"].tolist()
        print(f"\n  top 3 decision factors: {bold(', '.join(top3))}")

        # explain each sample decision using top features
        if data_path.exists():
            print(f"\n  {bold('per-sample reasoning')} (top factors for each decision):")
            print(f"  {'─' * 58}")

            X_samples = pd.concat([sample_approved, sample_rejected])[data_cfg.all_features]
            X_transformed = preprocessor.transform(X_samples)

            top_features = imp_df.head(5)["feature"].tolist()
            top_indices = [feature_names.index(f) for f in top_features]

            for i, (idx, row) in enumerate(pd.concat([sample_approved, sample_rejected]).iterrows()):
                status = "APPROVED" if row["loan_status"] == 1 else "REJECTED"
                status_colored = ok(status) if status == "APPROVED" else fail(status)
                transformed_values = X_transformed[i]

                # show original values for the top features
                reasons = []
                for feat_name, feat_idx in zip(top_features, top_indices):
                    # get original value if it's a raw feature
                    if feat_name in row.index:
                        val = row[feat_name]
                        if isinstance(val, float):
                            reasons.append(f"{feat_name}={val:.1f}")
                        else:
                            reasons.append(f"{feat_name}={val}")
                    else:
                        # one-hot encoded feature
                        encoded_val = transformed_values[feat_idx]
                        reasons.append(f"{feat_name}={encoded_val:.2f}")

                print(f"\n    [{status_colored}] age={int(row['person_age'])}, income=${row['person_income']:,.0f}, loan=${row['loan_amnt']:,.0f}")
                print(f"      factors: {' | '.join(reasons)}")

    elif hasattr(clf, "coef_"):
        print(f"\n  {info('(linear model — uses coefficient weights, not tree importances)')}")
    else:
        print(f"\n  {info('(model type does not expose feature importances)')}")

except Exception as e:
    print(f"  {fail('could not load model for analysis:')} {e}")

print(f"\n{'=' * 64}\n")
