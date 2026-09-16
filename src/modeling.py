
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor

from .data_cleaning import prepare_regression_frame


@dataclass
class RegressionResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame


def _onehot_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=20,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )


def _tree_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numerical),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def run_regression(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    random_state: int = 42,
) -> RegressionResult:
    X, y = prepare_regression_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state
    )
    y_train_log = np.log1p(y_train.to_numpy())

    rows = []
    pred_table = pd.DataFrame({"actual": y_test.to_numpy()})

    # A simple benchmark makes model gains interpretable.
    dummy = DummyRegressor(strategy="median")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    dummy_pred = dummy.predict(np.zeros((len(y_test), 1)))
    rows.append(_metric_row("Median baseline", y_test.to_numpy(), dummy_pred))
    pred_table["Median baseline"] = dummy_pred

    # Random Forest: categorical variables are encoded inside a preprocessing
    # pipeline fit only on the training partition.
    tree_pre = _tree_preprocessor(X_train)
    X_train_tree = tree_pre.fit_transform(X_train)
    X_test_tree = tree_pre.transform(X_test)

    rf = ExtraTreesRegressor(
        n_estimators=100,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train_tree, y_train_log)
    rf_pred = np.expm1(rf.predict(X_test_tree))
    rf_pred = np.clip(rf_pred, 0, None)
    rows.append(_metric_row("Extra Trees", y_test.to_numpy(), rf_pred))
    pred_table["Extra Trees"] = rf_pred

    # XGBoost: one-hot encoded categorical variables remain sparse.
    xgb_pre = _onehot_preprocessor(X_train)
    X_train_xgb = xgb_pre.fit_transform(X_train)
    X_test_xgb = xgb_pre.transform(X_test)

    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(X_train_xgb, y_train_log)
    xgb_pred = np.expm1(xgb.predict(X_test_xgb))
    xgb_pred = np.clip(xgb_pred, 0, None)
    rows.append(_metric_row("XGBoost", y_test.to_numpy(), xgb_pred))
    pred_table["XGBoost"] = xgb_pred

    metrics = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    rf_feature_names = tree_pre.get_feature_names_out()
    importance = pd.DataFrame(
        {"feature": rf_feature_names, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out / "model_metrics.csv", index=False)
        pred_table.to_csv(out / "test_predictions.csv", index=False)
        importance.to_csv(out / "feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        order = metrics.sort_values("MAE")
        ax.bar(order["model"], order["MAE"])
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Regression model comparison")
        fig.tight_layout()
        fig.savefig(out / "model_comparison.png", dpi=180)
        plt.close(fig)

        # The Extra Trees model provides a strong nonlinear benchmark.
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(y_test, rf_pred, s=10, alpha=0.25)
        lo = min(float(y_test.min()), float(rf_pred.min()))
        hi = np.quantile(np.concatenate([y_test.to_numpy(), rf_pred]), 0.995)
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual happiness_index")
        ax.set_ylabel("Predicted happiness_index")
        ax.set_title("Extra Trees: actual vs predicted")
        fig.tight_layout()
        fig.savefig(out / "actual_vs_predicted.png", dpi=180)
        plt.close(fig)

        top = importance.head(15).sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top["importance"])
        ax.set_xlabel("Extra Trees feature importance")
        ax.set_title("Top model features")
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    return RegressionResult(metrics, pred_table, importance)
