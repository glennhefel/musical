from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


def save_easy_task_cluster_viz(run_dir: Path, methods: list[str]) -> None:
    """Generate 2D cluster visualizations from an existing latents.npz.

    This is intentionally a post-processing step so we don't need to re-run
    training/clustering to produce both t-SNE and UMAP plots.
    """

    latents_path = run_dir / "latents.npz"
    if not latents_path.exists():
        return

    import numpy as np

    from src.visualization import project_2d, save_scatter

    data = np.load(latents_path, allow_pickle=True)
    z = data["z"]
    clusters = data["clusters"]

    for method in methods:
        xy = project_2d(z, method=method)
        out_path = run_dir / "latent_visualization" / f"clusters_{method}.png"
        save_scatter(
            xy,
            clusters,
            out_path,
            title=f"Cluster visualization ({method.upper()})",
        )
        print(f"[easy-viz] saved plot: {out_path}")


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def write_en_bn_subset(
    src_csv: Path,
    out_csv: Path,
    n_en: int = 20,
    n_bn: int = 20,
    seed: int = 42,
) -> None:
    df = pd.read_csv(src_csv)
    # normalize
    df["language"] = df["language"].astype("string").fillna("").str.strip()

    df_en = df[df["language"].eq("en")].copy()
    df_bn = df[df["language"].eq("bn")].copy()

    if df_en.empty or df_bn.empty:
        raise SystemExit(
            f"Need both en and bn rows in {src_csv}. Found en={len(df_en)} bn={len(df_bn)}"
        )

    df_en = df_en.sample(n=min(n_en, len(df_en)), random_state=seed)
    df_bn = df_bn.sample(n=min(n_bn, len(df_bn)), random_state=seed)

    out = pd.concat([df_en, df_bn], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")


def aggregate_metrics(root: Path, out_csv: Path) -> None:
    paths = sorted(root.glob("**/clustering_metrics.csv"))
    frames: list[pd.DataFrame] = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "run_dir", p.parent.name)
        frames.append(df)

    if not frames:
        raise SystemExit(f"No clustering_metrics.csv found under {root}")

    out_df = pd.concat(frames, ignore_index=True)
    sort_cols = [
        c
        for c in ["feature", "run", "cluster_method", "run_dir"]
        if c in out_df.columns
    ]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run Easy/Medium/Hard task experiment suites and write results into\n"
            "results_easy_task/, results_medium_task/, results_hard_task/."
        )
    )
    ap.add_argument(
        "--mixed",
        type=Path,
        default=Path("data/metadata_audio_lyrics_mixed.csv"),
        help="Mixed audio+lyrics manifest (must include en + bn)",
    )
    ap.add_argument(
        "--lyrics-balanced",
        type=Path,
        default=Path("data/metadata_known_categories_preprocessed_balanced.csv"),
        help="Lyrics-only balanced metadata (used by Easy task)",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--epochs-easy", type=int, default=30)
    ap.add_argument("--epochs-medium", type=int, default=30)
    ap.add_argument("--epochs-hard", type=int, default=30)
    ap.add_argument(
        "--clusters-easy",
        type=int,
        default=2,
        help="Number of clusters for Easy task (default 2 for en vs bn)",
    )
    ap.add_argument("--clusters", type=int, default=6)
    ap.add_argument("--viz", type=str, default="tsne", choices=["tsne", "umap", "none"])
    ap.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=2,
        help="DBSCAN min_samples override for DBSCAN runs (lower reduces chance of all-noise)",
    )
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    # Write results next to this runner script (inside scripts/)
    results_easy = script_dir / "results_easy_task"
    results_medium = script_dir / "results_medium_task"
    results_hard = script_dir / "results_hard_task"

    # -------- Easy task dataset: lyrics-only (balanced en+bn) --------
    easy_manifest = args.lyrics_balanced
    easy_root = results_easy / "lyrics_only_known_balanced_runs"
    easy_root.mkdir(parents=True, exist_ok=True)

    # -------- Easy task runs --------
    # Basic VAE (MLP) on lyrics TF-IDF + KMeans
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(easy_manifest),
            "--feature",
            "lyrics_tfidf",
            "--vae-arch",
            "mlp",
            "--baseline",
            "none",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters_easy),
            "--epochs",
            str(args.epochs_easy),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--dist-cols",
            "language",
            "--outdir",
            str(easy_root / "vae_lyrics_tfidf_kmeans"),
        ]
    )

    # PCA + KMeans baseline
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(easy_manifest),
            "--feature",
            "lyrics_tfidf",
            "--baseline",
            "pca",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters_easy),
            "--epochs",
            str(args.epochs_easy),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--dist-cols",
            "language",
            "--outdir",
            str(easy_root / "baseline_pca_lyrics_tfidf_kmeans"),
        ]
    )

    # -------- Easy task: visualize clusters (t-SNE + UMAP) --------
    # The pipeline already supports --viz, but this explicit post-step guarantees
    # that Easy task always produces both plots from the saved latents.
    if args.viz != "none":
        for d in [
            easy_root / "vae_lyrics_tfidf_kmeans",
            easy_root / "baseline_pca_lyrics_tfidf_kmeans",
        ]:
            save_easy_task_cluster_viz(d, methods=["tsne", "umap"])

    aggregate_metrics(easy_root, results_easy / "easy_task_comparison.csv")

    # -------- Medium task runs --------
    # Conv VAE on spectrograms + multiple clusterers
    for method in ["kmeans", "agglomerative", "dbscan"]:
        cmd = [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "none",
            "--cluster-method",
            method,
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_medium),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--save-recon",
            "--n-recon",
            "6",
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_medium / f"convvae_logmel_{method}"),
        ]

        if method == "dbscan":
            cmd += ["--dbscan-min-samples", str(args.dbscan_min_samples)]

        run(cmd)

    # Hybrid: audio (logmelspec VAE embedding) + lyrics embedding
    for method in ["kmeans", "agglomerative", "dbscan"]:
        cmd = [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "hybrid",
            "--audio-feature",
            "logmelspec",
            "--lyrics-embed-dim",
            "128",
            "--baseline",
            "none",
            "--cluster-method",
            method,
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_medium),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_medium / f"hybrid_logmel_{method}"),
        ]

        if method == "dbscan":
            cmd += ["--dbscan-min-samples", str(args.dbscan_min_samples)]

        run(cmd)

    aggregate_metrics(results_medium, results_medium / "medium_task_comparison.csv")

    # -------- Medium task baselines (VAE vs baseline) --------
    # Keep baselines lightweight: single clustering method per baseline.
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "pca",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_medium / "baseline_pca_kmeans_logmel"),
        ]
    )

    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "ae",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_medium),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--save-recon",
            "--n-recon",
            "6",
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_medium / "baseline_ae_kmeans_logmel"),
        ]
    )

    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "raw",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_medium / "baseline_raw_kmeans_logmel"),
        ]
    )

    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "raw",
            "--cluster-method",
            "spectral",
            "--spectral-n-neighbors",
            "10",
            "--clusters",
            str(args.clusters),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_medium / "baseline_spectral_raw_logmel"),
        ]
    )

    aggregate_metrics(results_medium, results_medium / "medium_task_comparison.csv")

    # -------- Hard task runs --------
    # Beta-VAE (disentangled) on logmelspec
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--beta",
            "4",
            "--baseline",
            "none",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--save-recon",
            "--n-recon",
            "6",
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_hard / "hard_task_runs" / "betavae_logmel_kmeans"),
        ]
    )

    # CVAE conditioned on language
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--cond-col",
            "language_group",
            "--baseline",
            "none",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--save-recon",
            "--n-recon",
            "6",
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_hard / "hard_task_runs" / "cvae_logmel_lang_kmeans"),
        ]
    )

    # Multi-modal: audio + lyrics + genre/category
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "multimodal",
            "--audio-feature",
            "logmelspec",
            "--genre-col",
            "category",
            "--baseline",
            "none",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language",
            "--viz",
            args.viz,
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_hard / "hard_task_runs" / "multimodal_logmel_kmeans"),
        ]
    )

    # Baselines: PCA + KMeans, AE + KMeans, raw + KMeans, raw + Spectral
    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "pca",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_hard / "hard_task_runs" / "baseline_pca_kmeans_logmel"),
        ]
    )

    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "ae",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--save-recon",
            "--n-recon",
            "6",
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_hard / "hard_task_runs" / "baseline_ae_kmeans_logmel"),
        ]
    )

    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "raw",
            "--cluster-method",
            "kmeans",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_hard / "hard_task_runs" / "baseline_raw_kmeans_logmel"),
        ]
    )

    run(
        [
            py,
            "-m",
            "src.run_pipeline",
            "--metadata",
            str(args.mixed),
            "--feature",
            "logmelspec",
            "--vae-arch",
            "conv2d",
            "--baseline",
            "raw",
            "--cluster-method",
            "spectral",
            "--spectral-n-neighbors",
            "10",
            "--clusters",
            str(args.clusters),
            "--epochs",
            str(args.epochs_hard),
            "--device",
            args.device,
            "--label-col",
            "language_group",
            "--viz",
            args.viz,
            "--dist-cols",
            "language_group",
            "--outdir",
            str(results_hard / "hard_task_runs" / "baseline_spectral_raw_logmel"),
        ]
    )

    aggregate_metrics(
        results_hard / "hard_task_runs", results_hard / "hard_task_comparison.csv"
    )

    print("\nDone.")
    print("Easy:", results_easy)
    print("Medium:", results_medium)
    print("Hard:", results_hard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
