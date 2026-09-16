
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "referral_friends",
    "water_sources",
    "shelters",
    "fauna_friendly",
    "island_size",
    "creation_time",
    "region",
    "happiness_metric",
    "features",
    "happiness_index",
    "loyalty_score",
    "total_refunds_requested",
    "trade_goods",
    "x_coordinate",
    "avg_time_in_euphoria",
    "y_coordinate",
    "island_id",
    "entry_fee",
    "nearest_city",
]

NUMERIC_COLUMNS = [
    "referral_friends",
    "water_sources",
    "shelters",
    "island_size",
    "creation_time",
    "happiness_index",
    "loyalty_score",
    "total_refunds_requested",
    "x_coordinate",
    "avg_time_in_euphoria",
    "y_coordinate",
    "island_id",
]


def _parse_row(row: list[str], expected_len: int) -> list[str]:
    """Recover rows that were stored as one quoted CSV field."""
    if len(row) == 1:
        row = next(csv.reader([row[0]]))
    if len(row) != expected_len:
        raise ValueError(f"Expected {expected_len} fields, found {len(row)}")
    return row


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the supplied course dataset without fragile positional splitting."""
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != EXPECTED_COLUMNS:
            raise ValueError(
                "Unexpected dataset schema. "
                f"Expected {EXPECTED_COLUMNS}, found {header}."
            )
        rows = [_parse_row(row, len(header)) for row in reader]

    df = pd.DataFrame(rows, columns=header)
    df = df.replace(r"^\s*$", np.nan, regex=True)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["creation_datetime"] = pd.to_datetime(
        out["creation_time"], unit="s", errors="coerce", utc=True
    )
    out["creation_year"] = out["creation_datetime"].dt.year
    out["creation_month"] = out["creation_datetime"].dt.month

    out["amenity_count"] = out["features"].fillna("").map(
        lambda x: 0 if not x else len([item for item in str(x).split(",") if item.strip()])
    )

    fauna = out["fauna_friendly"].fillna("")
    out["cats_allowed"] = fauna.str.contains("Cats", case=False, regex=False).astype(int)
    out["dogs_allowed"] = fauna.str.contains("Dogs", case=False, regex=False).astype(int)

    return out


def prepare_regression_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create X/y for the corrected supervised-learning task.

    In the supplied dataset, ``happiness_index`` is a continuous variable
    (not a class label), so the portfolio version models it as regression.
    Identifier-like and near-constant fields are excluded.
    """
    data = engineer_features(df)
    data = data[data["happiness_index"].notna()].copy()

    feature_cols = [
        "referral_friends",
        "water_sources",
        "shelters",
        "island_size",
        "loyalty_score",
        "total_refunds_requested",
        "x_coordinate",
        "avg_time_in_euphoria",
        "y_coordinate",
        "creation_year",
        "creation_month",
        "amenity_count",
        "cats_allowed",
        "dogs_allowed",
        "region",
        "entry_fee",
        "nearest_city",
    ]

    X = data[feature_cols].copy()
    y = data["happiness_index"].astype(float)
    return X, y


def clustering_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = engineer_features(df)
    cols = [
        "referral_friends",
        "water_sources",
        "shelters",
        "island_size",
        "happiness_index",
        "loyalty_score",
        "total_refunds_requested",
        "avg_time_in_euphoria",
        "x_coordinate",
        "y_coordinate",
        "amenity_count",
    ]
    return data[cols].copy()
