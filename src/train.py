"""Trains models with MLflow tracking. Supports sklearn and custom models, with optional PCA/LDA reduction."""

import importlib
import logging
import time

import mlflow
import mlflow.sklearn

logging.getLogger("mlflow").setLevel(logging.CRITICAL)
logging.getLogger("mlflow.sklearn").setLevel(logging.CRITICAL)

TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(TRACKING_URI)

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import ModelSpec


def _resolve_class(class_path: str):
    module_path, class_name = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _valid_params(model_class, params: dict) -> dict:
    return {k: v for k, v in params.items() if k in model_class().get_params()}


def _build_model(spec: ModelSpec, random_state: int):
    """Builds model from spec. Wraps in PCA/LDA pipeline if reduction is set."""
    cls = _resolve_class(spec.model_class)
    model = cls(**_valid_params(cls, spec.params))

    if not spec.reduction:
        return model

    steps = [("scaler", StandardScaler())]
    if spec.reduction == "pca":
        steps.append(("pca", PCA(n_components=0.95, random_state=random_state)))
    elif spec.reduction == "lda":
        steps.append(("lda", LinearDiscriminantAnalysis()))
    steps.append(("clf", model))
    return Pipeline(steps)


def train_model(spec: ModelSpec, X_train, y_train, X_test, y_test,
                execution_id: str, mode: str = "production", random_state: int = 42,
                cv_folds: int = 5, search_n_iter: int = 30) -> dict:
    """Trains one model, logs everything to MLflow, returns metrics."""
    mlflow.set_experiment(f"loan_approval_{mode}")

    existing = mlflow.search_runs(
        filter_string=f"tags.execution_id = '{execution_id}' and tags.mlflow.runName = '{spec.name}'",
    )
    if not existing.empty:
        client = mlflow.tracking.MlflowClient()
        for _, old in existing.iterrows():
            client.delete_run(old["run_id"])

    with mlflow.start_run(run_name=spec.name) as run:
        start = time.time()

        mlflow.set_tag("execution_id", execution_id)
        mlflow.set_tag("mode", mode)
        mlflow.set_tag("model_class", spec.model_class)
        if spec.reduction:
            mlflow.set_tag("reduction", spec.reduction)

        for k, v in spec.params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("reduction", spec.reduction or "none")

        if spec.search_params:
            base = _build_model(spec, random_state)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            search_params = {f"clf__{k}": v for k, v in spec.search_params.items()} if spec.reduction else spec.search_params
            search = RandomizedSearchCV(base, search_params, n_iter=search_n_iter, cv=cv, scoring="f1", random_state=random_state, n_jobs=-1)
            search.fit(X_train, y_train)
            model = search.best_estimator_
            mlflow.log_metric("best_cv_f1", search.best_score_)
            for k, v in search.best_params_.items():
                mlflow.log_param(f"best_{k}", v)
        else:
            model = _build_model(spec, random_state)
            model.fit(X_train, y_train)

        train_time = time.time() - start
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "f1_train": f1_score(y_train, model.predict(X_train), zero_division=0),
            "train_time_seconds": train_time,
        }
        metrics["f1_gap"] = metrics["f1_train"] - metrics["f1_score"]

        if hasattr(model, "predict_proba"):
            metrics["auc_roc"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        if spec.reduction == "pca" and hasattr(model, "named_steps"):
            pca = model.named_steps["pca"]
            mlflow.log_param("pca_n_components", pca.n_components_)
            mlflow.log_metric("pca_variance_retained", sum(pca.explained_variance_ratio_))

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, name="model", registered_model_name=spec.name, serialization_format="cloudpickle")
        mlflow.log_dict({"tn": int((cm := confusion_matrix(y_test, y_pred))[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}, "confusion_matrix.json")

        reduction_tag = f" ({spec.reduction})" if spec.reduction else ""
        print(f"  [{spec.name}] f1={metrics['f1_score']:.4f} acc={metrics['accuracy']:.4f} gap={metrics['f1_gap']:.4f} time={train_time:.1f}s{reduction_tag}")

        return {"run_id": run.info.run_id, "model": model, "y_pred": y_pred, "metrics": metrics, "name": spec.name}


def run_experiments(experiments, X_train, y_train, X_test, y_test, execution_id, mode, random_state=42):
    """Trains all models from experiments config."""
    print(f"\n[train] running {len(experiments.models)} models (mode={mode}, id={execution_id})")
    results = [train_model(s, X_train, y_train, X_test, y_test, execution_id, mode, random_state) for s in experiments.models]
    print(f"[train] {len(results)} models registered in mlflow")
    return results
