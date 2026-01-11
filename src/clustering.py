from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering


@dataclass
class ClusterConfig:
    method: str = "kmeans"  # kmeans | agglomerative | dbscan | spectral
    n_clusters: int = 4
    random_state: int = 42

    # DBSCAN params
    eps: float = 0.5
    min_samples: int = 5

    # Spectral params
    spectral_n_neighbors: int = 10


def cluster_embeddings(z: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    method = cfg.method.lower()

    if method == "kmeans":
        model = KMeans(n_clusters=cfg.n_clusters, n_init="auto", random_state=cfg.random_state)
        return model.fit_predict(z)

    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=cfg.n_clusters)
        return model.fit_predict(z)


    if method == "dbscan":
        # Try initial parameters
        model = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples)
        labels = model.fit_predict(z)
        # If all points are noise, try a range of eps values
        if np.all(labels == -1):
            print(f"[DBSCAN] All points labeled as noise with eps={cfg.eps}. Trying alternative eps values...")
            # Try a geometric range of eps values
            eps_grid = np.geomspace(0.05, 5.0, num=10)
            for eps_try in eps_grid:
                model_try = DBSCAN(eps=eps_try, min_samples=cfg.min_samples)
                labels_try = model_try.fit_predict(z)
                n_clusters = len(set(labels_try)) - (1 if -1 in labels_try else 0)
                if n_clusters >= 2:
                    print(f"[DBSCAN] Found eps={eps_try:.3f} yielding {n_clusters} clusters.")
                    return labels_try
            print("[DBSCAN] Warning: All tried eps values resulted in all-noise or <2 clusters. Returning original labels.")
        return labels

    if method == "spectral":
        if z.shape[0] < 2:
            raise ValueError("Spectral clustering requires at least 2 samples")
        n_neighbors = int(min(max(2, cfg.spectral_n_neighbors), z.shape[0] - 1))
        model = SpectralClustering(
            n_clusters=int(cfg.n_clusters),
            affinity="nearest_neighbors",
            n_neighbors=n_neighbors,
            assign_labels="kmeans",
            random_state=int(cfg.random_state),
        )
        return model.fit_predict(z)

    raise ValueError(f"Unknown clustering method: {cfg.method}")
