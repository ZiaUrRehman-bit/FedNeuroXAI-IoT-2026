"""
Dataset loading. Loads a tabular CSV, integer-encodes the target column and
sanitises feature names so that they survive PyAerial's rule mining
unchanged.
"""
import os
import re
from typing import Dict, Tuple

import pandas as pd

from . import config


def load_csv(file_path: str, target_col: str) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    """Returns (X, y, label_map) with y integer-encoded."""
    df = pd.read_csv(file_path)
    df.columns = [re.sub(r"[()[\]<>\s]+", "_", c).strip("_") for c in df.columns]
    target_col = re.sub(r"[()[\]<>\s]+", "_", target_col).strip("_")

    if df.isnull().sum().sum() > 0:
        print(f"[data] WARNING: {df.isnull().sum().sum()} missing values found.")

    if pd.api.types.is_numeric_dtype(df[target_col]):
        df[target_col] = df[target_col].astype(int)
        label_map = {v: int(v) for v in sorted(df[target_col].unique())}
    else:
        labels = sorted(df[target_col].unique())
        label_map = {lab: i for i, lab in enumerate(labels)}
        df[target_col] = df[target_col].map(label_map).astype(int)

    df = df.rename(columns={target_col: "class"})
    X = df.drop(columns=["class"])
    y = df["class"]
    print(f"[data] {os.path.basename(file_path)}: {X.shape[0]} rows, "
          f"{X.shape[1]} features, {y.nunique()} classes, labels={label_map}")
    return X, y, label_map


def sanitize_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Collapse feature names to alphanumerics and single underscores.

    PyAerial rewrites some characters in column names (and reserves "__" as
    its own separator) during rule mining; without this step a mined rule's
    feature name can silently differ from the DataFrame column it refers to.
    """
    X = X.copy()
    X.columns = [
        re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z_]+", "_", c)).strip("_")
        for c in X.columns
    ]
    return X


def load_dataset(dataset_key: str, sanitize: bool = True):
    """Load one of the paper's datasets by key (see config.DATASETS)."""
    cfg = config.DATASETS[dataset_key]
    X, y, label_map = load_csv(os.path.join(config.DATA_DIR, cfg["file"]), cfg["target"])
    if sanitize:
        X = sanitize_columns(X)
    return X, y, label_map
