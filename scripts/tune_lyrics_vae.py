"""Tune a lyrics-only VAE + KMeans pipeline.

Tries a small grid of:
- VAE latent_dim
- VAE beta
- KMeans k (n_clusters)

And reports metrics (Silhouette, Calinski-Harabasz) for each run.

This is meant for lyrics-only runs (no audio). It uses TF-IDF features from
`lyrics_clean` (fallback: `lyrics`) via src.features.build_feature_matrix_lyrics_tfidf.

Usage:
  python scripts/tune_lyrics_vae.py \
    --metadata data/metadata_known_categories_preprocessed_balanced.csv \
    --outdir results_lyrics_tuning \
    --epochs 10

Outputs:
- <outdir>/tuning_results.csv
- prints best configs by silhouette and calinski_harabasz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path as _Path
import sys as _sys

# Allow running as: python scripts/tune_lyrics_vae.py
_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from src.clustering import ClusterConfig, cluster_embeddings
from src.dataset import load_metadata, standardize_features
from src.evaluation import compute_clustering_metrics
from src.features import build_feature_matrix_lyrics_tfidf
from src.train_vae import TrainConfig, train_vae
from src.vae import VAEConfig, encode_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata",
        type=str,
        default="data/metadata_known_categories_preprocessed_balanced.csv",
    )
    ap.add_argument("--outdir", type=str, default="results_lyrics_tuning")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--latent-dims",
        type=str,
        default="8,16,32",
        help="Comma-separated latent dims to try",
    )
    ap.add_argument(
        "--betas",
        type=str,
        default="0.1,1.0",
        help="Comma-separated beta values to try",
    )
    ap.add_argument(
        "--ks",
        type=str,
        default="2,4,6,8",
        help="Comma-separated KMeans cluster counts to try",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.metadata)

    # Optional labels for ARI/NMI (if label present)
    label_map = {str(row["id"]): row.get("label", None) for _, row in df.iterrows()}

    x, ids = build_feature_matrix_lyrics_tfidf(df=df)
    x_std, mean, std = standardize_features(x)

    y_true = [label_map.get(track_id, None) for track_id in ids]
    y_true_clean = None
    if all(v is not None and not (isinstance(v, float) and np.isnan(v)) for v in y_true):
        y_true_clean = [str(v) for v in y_true]

    latent_dims = [int(s.strip()) for s in args.latent_dims.split(",") if s.strip()]
    betas = [float(s.strip()) for s in args.betas.split(",") if s.strip()]
    ks = [int(s.strip()) for s in args.ks.split(",") if s.strip()]

    rows = []

    for latent_dim in latent_dims:
        for beta in betas:
            vae_cfg = VAEConfig(input_dim=x_std.shape[1], latent_dim=latent_dim, beta=beta)
            train_cfg = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )

            model = train_vae(x_std, vae_cfg, train_cfg)
            z = encode_dataset(model, x_std, batch_size=256, device=args.device)

            for k in ks:
                cfg = ClusterConfig(method="kmeans", n_clusters=k)
                clusters = cluster_embeddings(z, cfg)
                metrics = compute_clustering_metrics(z, clusters, labels_true=y_true_clean)
                rows.append(
                    {
                        "model": "vae",
                        "feature": "lyrics_tfidf",
                        "latent_dim": latent_dim,
                        "beta": beta,
                        "k": k,
                        **metrics,
                    }
                )

    out = pd.DataFrame(rows)
    out_path = outdir / "tuning_results.csv"
    out.to_csv(out_path, index=False)

    # Best by silhouette and calinski
    out_valid = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["silhouette", "calinski_harabasz"])

    best_s = out_valid.sort_values("silhouette", ascending=False).iloc[0]
    best_c = out_valid.sort_values("calinski_harabasz", ascending=False).iloc[0]

    print(f"Wrote: {out_path}")
    print("\nBest by silhouette:")
    print(best_s[["latent_dim", "beta", "k", "silhouette", "calinski_harabasz"]].to_string())
    print("\nBest by calinski_harabasz:")
    print(best_c[["latent_dim", "beta", "k", "silhouette", "calinski_harabasz"]].to_string())


if __name__ == "__main__":
    main()
