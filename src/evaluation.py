from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def compute_clustering_metrics(z: np.ndarray, labels_pred: np.ndarray, labels_true: list[str] | None = None) -> dict:
    metrics: dict[str, float | int] = {}

    # Some algorithms (DBSCAN) can output -1 for noise; metrics may be undefined if < 2 clusters
    unique_clusters = np.unique(labels_pred)
    n_clusters_eff = len(unique_clusters[unique_clusters >= 0])
    metrics["n"] = int(z.shape[0])
    metrics["n_clusters_eff"] = int(n_clusters_eff)

    # For DBSCAN, drop noise points for internal metrics; otherwise noise can dominate.
    mask = labels_pred >= 0
    z_m = z[mask]
    y_m = labels_pred[mask]
    uniq_m = np.unique(y_m)
    n_clusters_m = len(uniq_m)

    if n_clusters_m >= 2 and z_m.shape[0] >= 2:
        metrics["silhouette"] = float(silhouette_score(z_m, y_m))
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(z_m, y_m))
        metrics["davies_bouldin"] = float(davies_bouldin_score(z_m, y_m))
    else:
        metrics["silhouette"] = float("nan")
        metrics["calinski_harabasz"] = float("nan")
        metrics["davies_bouldin"] = float("nan")

    if labels_true is not None:
        y_true = np.asarray(labels_true)

        # Supervised metrics should ignore missing/degenerate label sets.
        if len(np.unique(y_true)) >= 2:
            metrics["ari"] = float(adjusted_rand_score(y_true, labels_pred))
            metrics["nmi"] = float(normalized_mutual_info_score(y_true, labels_pred))

        # Cluster purity: sum over clusters of the dominant true label count, / N.
        # For DBSCAN, ignore noise points for purity (consistent with internal metrics masking).
        if labels_pred.shape[0] == y_true.shape[0]:
            mask_p = labels_pred >= 0
            if int(mask_p.sum()) > 0:
                y_p = labels_pred[mask_p]
                t_p = y_true[mask_p]
                total = int(t_p.shape[0])
                purity_sum = 0
                for c in np.unique(y_p):
                    idx = y_p == c
                    if not np.any(idx):
                        continue
                    vals, counts = np.unique(t_p[idx], return_counts=True)
                    purity_sum += int(counts.max()) if counts.size else 0
                metrics["purity"] = float(purity_sum / max(1, total))
            else:
                metrics["purity"] = float("nan")

    return metrics


def append_metrics_csv(path: str, row: dict) -> None:
    df_row = pd.DataFrame([row])
    p = path
    try:
        existing = pd.read_csv(p)
        out = pd.concat([existing, df_row], ignore_index=True)
    except FileNotFoundError:
        out = df_row
    out.to_csv(p, index=False)
