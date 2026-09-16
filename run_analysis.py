
from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from src.clustering import run_clustering
from src.data_cleaning import engineer_features, load_raw_dataset
from src.modeling import run_regression


def save_dataset_summary(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    engineered = engineer_features(df)

    summary = pd.DataFrame(
        {
            "dtype": engineered.dtypes.astype(str),
            "missing_pct": engineered.isna().mean().mul(100).round(2),
            "n_unique": engineered.nunique(dropna=True),
        }
    )
    summary.to_csv(out_dir / "dataset_summary.csv")

    missing = engineered.isna().mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(missing.index[::-1], missing.values[::-1])
    ax.set_xlabel("Missing values (%)")
    ax.set_title("Dataset missingness")
    fig.tight_layout()
    fig.savefig(out_dir / "missingness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    target = engineered["happiness_index"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(target, bins=60)
    ax.set_xlim(0, target.quantile(0.995))
    ax.set_xlabel("happiness_index")
    ax.set_ylabel("Count")
    ax.set_title("Target distribution (x-axis capped at 99.5th percentile)")
    fig.tight_layout()
    fig.savefig(out_dir / "target_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parent
    raw_path = root / "data" / "raw" / "euphoria_dataset.csv"
    out_dir = root / "outputs"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. "
            "See data/README.md for setup instructions."
        )

    df = load_raw_dataset(raw_path)
    save_dataset_summary(df, out_dir)

    regression = run_regression(df, out_dir)
    clustering = run_clustering(df, out_dir)

    print("\nRegression results")
    print(regression.metrics.to_string(index=False))
    print(f"\nSelected number of clusters: {clustering.best_k}")
    print("\nClustering scores")
    print(clustering.scores.to_string(index=False))


if __name__ == "__main__":
    main()
