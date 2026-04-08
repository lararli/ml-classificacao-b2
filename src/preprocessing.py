"""Preprocessing pipeline: ColumnTransformer + train/test split."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder

from src.config import DataConfig, PipelineConfig
from src.ingestion import load_data


def create_preprocessor(data: DataConfig) -> ColumnTransformer:
    """RobustScaler for numerical, OneHotEncoder for categorical."""
    return ColumnTransformer([
        ("num", RobustScaler(), data.numerical),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), data.categorical),
    ])


def get_feature_names(preprocessor: ColumnTransformer, data: DataConfig) -> list[str]:
    """Returns feature names after encoding."""
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(data.categorical))
    return data.numerical + cat_names


def load_and_split(data: DataConfig, pipeline: PipelineConfig, run_id: str = "") -> tuple:
    """Loads data and splits into train/test."""
    df = load_data(pipeline, run_id)
    X = df[data.all_features]
    y = df[data.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=pipeline.test_size,
        random_state=pipeline.random_state,
        stratify=y if pipeline.stratify else None,
    )
    print(f"PREPROCESSING: train={len(X_train)} | test={len(X_test)}")
    return X_train, X_test, y_train, y_test
