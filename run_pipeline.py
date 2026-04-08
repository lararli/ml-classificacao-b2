"""
ML Pipeline - Orchestrator
Run: python run_pipeline.py [mode] [execution_id]
  mode: experimentation (default) or production
  execution_id: MMYYYY (default: current month)
"""

import logging
import sys
import warnings

# Suppress known warnings from external libraries:
# - mlflow.sklearn: cloudpickle security advisory (informational, not actionable)
# - mlflow.store/tracking: first-run setup messages
# - sklearn.utils.parallel: compatibility issue with Python 3.13
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
for logger_name in ["mlflow", "mlflow.sklearn", "mlflow.store", "mlflow.tracking", "mlflow.models"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

MODE = sys.argv[1] if len(sys.argv) > 1 else "experimentation"
EXECUTION_ID = sys.argv[2] if len(sys.argv) > 2 else None

# ═══════════════════════════════════════════════════════════
# Config validation (gate)
# ═══════════════════════════════════════════════════════════

from src.config import DataConfig, PipelineConfig, QualityConfig, ExperimentsConfig, resolve_execution_id

print("validating configs...")
data = DataConfig.load()
print(f"  [ok] data.yaml - source={data.source}, target={data.target}, "
      f"{len(data.numerical)} num, {len(data.categorical)} cat")

pipeline = PipelineConfig.load()
print(f"  [ok] pipeline.yaml - random_state={pipeline.random_state}, test_size={pipeline.test_size}")

quality = QualityConfig.load()
print(f"  [ok] quality.yaml - {quality.table.min_rows}-{quality.table.max_rows} rows, "
      f"{len(quality.columns)} columns")

experiments = ExperimentsConfig.load(mode=MODE)
print(f"  [ok] experiments - {len(experiments.models)} models ({MODE})")

EXECUTION_ID = resolve_execution_id(EXECUTION_ID)
print(f"\nmode: {MODE}")
print(f"execution_id: {EXECUTION_ID}")
print(f"models: {[m.name for m in experiments.models]}")

# ═══════════════════════════════════════════════════════════
# Job 1: Ingestion
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("JOB 1: INGESTION")
print("=" * 60)

from src.ingestion import download_dataset, convert_to_parquet, validate_schema, load_data

RUN_ID = f"{MODE}/{EXECUTION_ID}"
print(f"run path: {RUN_ID}")

csv_path = download_dataset(data, pipeline)
parquet_path = convert_to_parquet(csv_path, pipeline, RUN_ID)
df = load_data(pipeline, RUN_ID)
validate_schema(df, data)
print(f"rows: {len(df)}, cols: {len(df.columns)}")

# ═══════════════════════════════════════════════════════════
# Job 2: Quality
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("JOB 2: QUALITY")
print("=" * 60)

from src.quality_checks import run_quality_checks, save_report

report = run_quality_checks(df, quality)
save_report(report, execution_id=RUN_ID)

for check in report:
    status = "ok" if check["passed"] else "FAILED"
    col = check.get("column", "table")
    print(f"  [{status}] {check['type']} ({col}): {check['detail']}")

# ═══════════════════════════════════════════════════════════
# Job 3: Experimentation
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("JOB 3: EXPERIMENTATION")
print("=" * 60)

from src.preprocessing import create_preprocessor, load_and_split, get_feature_names
from src.train import run_experiments

X_train, X_test, y_train, y_test = load_and_split(data, pipeline, RUN_ID)
preprocessor = create_preprocessor(data)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
feature_names = get_feature_names(preprocessor, data)
print(f"features after encoding: {X_train_proc.shape[1]}")

results = run_experiments(experiments, X_train_proc, y_train, X_test_proc, y_test, EXECUTION_ID, MODE, pipeline.random_state)

# ═══════════════════════════════════════════════════════════
# Job 4: Selection
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("JOB 4: MODEL SELECTION")
print("=" * 60)

from src.evaluate import compare_models, financial_analysis

comparison = compare_models(f"loan_approval_{MODE}", EXECUTION_ID)
print(comparison.to_string(index=False))

predictions = {r["name"]: r["y_pred"] for r in results}

fin = financial_analysis(y_test, predictions, X_test)
print("\nfinancial analysis:")
print(fin.to_string(index=False))

# Selection justification
best_result = max(results, key=lambda r: r["metrics"]["f1_score"])
print(f"\n--- model selection ---")
print(f"  selected: {best_result['name']}")
print(f"  f1={best_result['metrics']['f1_score']:.4f}, "
      f"precision={best_result['metrics']['precision']:.4f}, "
      f"recall={best_result['metrics']['recall']:.4f}")
print(f"  justification: highest f1 with acceptable gap ({best_result['metrics']['f1_gap']:.4f}), "
      f"lowest financial impact among top performers.")

# ═══════════════════════════════════════════════════════════
# Job 5: Operationalization
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("JOB 6: OPERATIONALIZATION")
print("=" * 60)

from src.serve import load_model, predict
print(f"best model: {best_result['name']} (f1={best_result['metrics']['f1_score']:.4f})")
print(f"run_id: {best_result['run_id']}")

model = load_model(best_result["run_id"])
sample = {col: X_test.iloc[0][col] for col in data.all_features}
print(f"\nsample input: {sample}")
result = predict(model, preprocessor, sample)

# ═══════════════════════════════════════════════════════════
# Drift Detection
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DRIFT DETECTION")
print("=" * 60)

from src.monitoring import detect_drift, check_new_categories

df_historical = df.sample(frac=pipeline.drift.historical_fraction, random_state=pipeline.random_state)
df_production = df.drop(df_historical.index)

print("numerical drift:")
drift_results = detect_drift(df_historical, df_production, data.numerical, threshold=pipeline.drift.threshold_pct)
for r in drift_results:
    flag = "DRIFT" if r["drift"] else "ok"
    print(f"  [{flag}] {r['feature']}: train={r['mean_train']}, prod={r['mean_production']}, var={r['variation_pct']}%")

print("\nnew categories:")
cat_results = check_new_categories(df_historical, df_production, data.categorical)
for r in cat_results:
    flag = "ALERT" if r["has_new"] else "ok"
    print(f"  [{flag}] {r['feature']}: {r['new_categories'] if r['has_new'] else 'none'}")

# ═══════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PIPELINE COMPLETED")
print("=" * 60)
print(f"mode: {MODE}")
print(f"execution_id: {EXECUTION_ID}")
print(f"models trained: {len(results)}")
print(f"best: {best_result['name']} (f1={best_result['metrics']['f1_score']:.4f})")
print(f"\nview results: mlflow ui")
