from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pick_latest(df: pd.DataFrame, run_name: str) -> pd.Series:
    sub = df[df["run"].astype(str).str.lower() == run_name.lower()].copy()
    if sub.empty:
        raise SystemExit(f"No rows found for run={run_name!r} in metrics CSV")
    return sub.iloc[-1]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare VAE vs PCA baseline using Silhouette and Calinski–Harabasz from results/clustering_metrics.csv"
    )
    ap.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/clustering_metrics.csv"),
        help="Path to clustering_metrics.csv (default: results/clustering_metrics.csv)",
    )
    args = ap.parse_args()

    if not args.metrics.exists():
        raise SystemExit(f"Metrics file not found: {args.metrics}")

    df = pd.read_csv(args.metrics)

    vae = _pick_latest(df, "vae")
    pca = _pick_latest(df, "pca")

    out = pd.DataFrame(
        [
            {
                "run": "vae",
                "n": int(vae["n"]),
                "n_clusters_eff": int(vae["n_clusters_eff"]),
                "silhouette": float(vae["silhouette"]),
                "calinski_harabasz": float(vae["calinski_harabasz"]),
            },
            {
                "run": "pca",
                "n": int(pca["n"]),
                "n_clusters_eff": int(pca["n_clusters_eff"]),
                "silhouette": float(pca["silhouette"]),
                "calinski_harabasz": float(pca["calinski_harabasz"]),
            },
        ]
    )

    out["silhouette"] = out["silhouette"].map(lambda v: f"{v:.6f}")
    out["calinski_harabasz"] = out["calinski_harabasz"].map(lambda v: f"{v:.6f}")

    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
