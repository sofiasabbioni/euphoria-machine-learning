
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_cleaning import clustering_frame


@dataclass
class ClusteringResult:
    scores: pd.DataFrame
    profiles: pd.DataFrame
    best_k: int


def run_clustering(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    random_state: int = 42,
    max_sample: int = 6000,
) -> ClusteringResult:
    X = clustering_frame(df)

    # K-Means is sensitive to extreme values. For clustering only, numerical
    # features are winsorized at the 1st/99th percentiles before imputation
    # and standardization. The regression task uses the original values.
    X = X.copy()
    for col in X.columns:
        low, high = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(low, high)

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    X_scaled = pipe.fit_transform(X)

    rng = np.random.default_rng(random_state)
    if len(X_scaled) > max_sample:
        idx = rng.choice(len(X_scaled), size=max_sample, replace=False)
        X_eval = X_scaled[idx]
    else:
        idx = np.arange(len(X_scaled))
        X_eval = X_scaled

    rows = []
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_eval)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(
                    X_eval,
                    labels,
                    sample_size=min(5000, len(X_eval)),
                    random_state=random_state,
                ),
                "davies_bouldin": davies_bouldin_score(X_eval, labels),
            }
        )

    scores = pd.DataFrame(rows)
    best_k = int(
        scores.sort_values(
            ["silhouette", "davies_bouldin"], ascending=[False, True]
        ).iloc[0]["k"]
    )

    final_model = KMeans(n_clusters=best_k, n_init=10, random_state=random_state)
    labels_all = final_model.fit_predict(X_scaled)

    clustered = X.copy()
    clustered["cluster"] = labels_all
    profiles = clustered.groupby("cluster").mean(numeric_only=True).round(2)
    profiles.insert(0, "count", clustered.groupby("cluster").size())

    pca = PCA(n_components=2, random_state=random_state)
    X_plot = X_scaled[idx]
    labels_plot = labels_all[idx]
    coords = pca.fit_transform(X_plot)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "clustering_scores.csv", index=False)
        profiles.to_csv(out / "cluster_profiles.csv")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(scores["k"], scores["silhouette"], marker="o", label="Silhouette")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title("KMeans cluster selection")
        ax.axvline(best_k, linestyle="--", label=f"Selected k={best_k}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "cluster_selection.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(coords[:, 0], coords[:, 1], c=labels_plot, s=8, alpha=0.45)
        ax.set_xlabel("PCA component 1")
        ax.set_ylabel("PCA component 2")
        ax.set_title(f"KMeans segmentation (k={best_k})")
        fig.tight_layout()
        fig.savefig(out / "clusters_pca.png", dpi=180)
        plt.close(fig)

    return ClusteringResult(scores, profiles, best_k)
