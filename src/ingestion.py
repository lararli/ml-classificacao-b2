"""Downloads dataset and converts to Parquet."""

import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import DataConfig, PipelineConfig


def download_dataset(data: DataConfig, pipeline: PipelineConfig) -> Path:
    """Downloads dataset from Kaggle to data/raw/."""
    import kagglehub

    raw_dir = Path(pipeline.paths.raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ingestion] downloading: {data.source}")
    path = kagglehub.dataset_download(data.source)
    downloaded = list(Path(path).glob(data.file_pattern))
    if not downloaded:
        raise FileNotFoundError(f"no {data.file_pattern} files in {path}")

    dest = raw_dir / downloaded[0].name
    shutil.copy2(downloaded[0], dest)
    print(f"[ingestion] csv: {dest}")
    return dest


def convert_to_parquet(csv_path: Path, pipeline: PipelineConfig, run_id: str = "") -> Path:
    """Converts CSV to Parquet with Snappy compression."""
    processed_dir = Path(pipeline.paths.processed_data_dir) / run_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = processed_dir / csv_path.with_suffix(".parquet").name
    df = pd.read_csv(csv_path)
    pq.write_table(pa.Table.from_pandas(df), parquet_path, compression="snappy")
    print(f"[ingestion] parquet: {parquet_path} ({len(df)} rows)")
    return parquet_path


def load_data(pipeline: PipelineConfig, run_id: str = "") -> pd.DataFrame:
    """Loads data from Parquet."""
    processed_dir = Path(pipeline.paths.processed_data_dir) / run_id
    parquet_files = list(processed_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no parquet in {processed_dir}")

    df = pd.read_parquet(parquet_files[0])
    print(f"[ingestion] loaded: {parquet_files[0].name} ({len(df)} rows, {len(df.columns)} cols)")
    return df


def validate_schema(df: pd.DataFrame, data: DataConfig) -> None:
    """Checks that all expected columns exist."""
    missing = set(data.all_columns) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {missing}")
    print(f"[ingestion] schema ok ({len(data.all_columns)} columns)")
