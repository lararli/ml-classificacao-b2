"""
ML Pipeline Orchestrator.
Usage: python run_pipeline.py [mode] [execution_id]
"""

import logging
import sys
import warnings

warnings.filterwarnings("ignore")
for name in ["mlflow", "mlflow.sklearn", "mlflow.store", "mlflow.tracking", "mlflow.models"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

import os
os.environ["TQDM_DISABLE"] = "1"

MODE = sys.argv[1] if len(sys.argv) > 1 else "experimentation"
EXECUTION_ID = sys.argv[2] if len(sys.argv) > 2 else None


def validate_configs():
    from src.config import DataConfig, PipelineConfig, QualityConfig, ExperimentsConfig, resolve_execution_id

    print("validating configs...")
    data = DataConfig.load()
    pipeline = PipelineConfig.load()
    quality = QualityConfig.load()
    experiments = ExperimentsConfig.load(mode=MODE)
    execution_id = resolve_execution_id(EXECUTION_ID)

    print(f"  [ok] data: {data.source}, {len(data.numerical)} num, {len(data.categorical)} cat")
    print(f"  [ok] pipeline: random_state={pipeline.random_state}, test_size={pipeline.test_size}")
    print(f"  [ok] quality: {len(quality.columns)} column checks")
    print(f"  [ok] experiments: {len(experiments.models)} models ({MODE})")
    print(f"\n  mode={MODE}  execution_id={execution_id}")

    return data, pipeline, quality, experiments, execution_id


def ingest(data, pipeline, run_id):
    from src.ingestion import download_dataset, convert_to_parquet, validate_schema, load_data

    print("\nJOB 1: INGESTION")
    csv_path = download_dataset(data, pipeline)
    convert_to_parquet(csv_path, pipeline, run_id)
    df = load_data(pipeline, run_id)
    validate_schema(df, data)
    return df


def check_quality(df, quality, run_id):
    from src.quality_checks import run_quality_checks, save_report

    print("\nJOB 2: QUALITY")
    report = run_quality_checks(df, quality)
    save_report(report, execution_id=run_id)
    for check in report:
        status = "ok" if check["passed"] else "FAILED"
        col = check.get("column", "table")
        print(f"  [{status}] {check['type']} ({col}): {check['detail']}")
    return report


def train_models(data, pipeline, experiments, execution_id, run_id):
    from src.preprocessing import create_preprocessor, load_and_split, get_feature_names
    from src.train import run_experiments

    print("\nJOB 3: TRAIN")
    X_train, X_test, y_train, y_test = load_and_split(data, pipeline, run_id)
    preprocessor = create_preprocessor(data)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"  features after encoding: {X_train_proc.shape[1]}")

    results = run_experiments(
        experiments, X_train_proc, y_train, X_test_proc, y_test,
        execution_id, MODE, pipeline.random_state
    )
    return results, X_test, y_test, preprocessor


def select_model(results, X_test, y_test, execution_id):
    from src.evaluate import compare_models, financial_analysis

    print("\nJOB 4: SELECTION")
    comparison = compare_models(f"loan_approval_{MODE}", execution_id)
    print(comparison.to_string(index=False))

    predictions = {r["name"]: r["y_pred"] for r in results}
    fin = financial_analysis(y_test, predictions, X_test)
    print("\nfinancial analysis:")
    print(fin.to_string(index=False))

    best = max(results, key=lambda r: r["metrics"]["f1_score"])
    print(f"\nselected: {best['name']} (f1={best['metrics']['f1_score']:.4f})")
    return best


def demonstrate_inference(best, data, preprocessor, X_test):
    from src.serve import load_model, predict

    print("\nJOB 5: INFERENCE")
    model = load_model(best["run_id"])
    sample = {col: X_test.iloc[0][col] for col in data.all_features}
    result = predict(model, preprocessor, sample)
    print(f"  result: {result}")


def detect_drift(df, data, pipeline):
    from src.monitoring import detect_drift as _detect_drift, check_new_categories

    print("\nJOB 6: DRIFT")
    df_hist = df.sample(frac=pipeline.drift.historical_fraction, random_state=pipeline.random_state)
    df_prod = df.drop(df_hist.index)

    drift = _detect_drift(df_hist, df_prod, data.numerical, threshold=pipeline.drift.threshold_pct)
    for r in drift:
        flag = "DRIFT" if r["drift"] else "ok"
        print(f"  [{flag}] {r['feature']}: var={r['variation_pct']}%")

    cats = check_new_categories(df_hist, df_prod, data.categorical)
    for r in cats:
        if r["has_new"]:
            print(f"  [ALERT] {r['feature']}: {r['new_categories']}")


def main():
    data, pipeline, quality, experiments, execution_id = validate_configs()
    run_id = f"{MODE}/{execution_id}"

    df = ingest(data, pipeline, run_id)
    check_quality(df, quality, run_id)
    results, X_test, y_test, preprocessor = train_models(data, pipeline, experiments, execution_id, run_id)
    best = select_model(results, X_test, y_test, execution_id)
    demonstrate_inference(best, data, preprocessor, X_test)
    detect_drift(df, data, pipeline)

    print(f"\nPIPELINE COMPLETED")
    print(f"  mode={MODE}  id={execution_id}  models={len(results)}  best={best['name']} (f1={best['metrics']['f1_score']:.4f})")


if __name__ == "__main__":
    main()
