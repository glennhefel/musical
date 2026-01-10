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
        model = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples)
        return model.fit_predict(z)

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
