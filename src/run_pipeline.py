from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .clustering import ClusterConfig, cluster_embeddings
from .dataset import load_metadata, standardize_features
from .evaluation import append_metrics_csv, compute_clustering_metrics
from .features import (
    build_feature_matrix_mfcc,
    build_feature_matrix_lyrics_tfidf,
    build_feature_tensor_logmelspec,
    build_feature_tensor_mfcc_frames,
    build_onehot_from_column,
    build_lyrics_embedding_svd,
)
from .train_vae import TrainConfig, train_vae
from .vae import VAEConfig, encode_dataset
from .visualization import (
    project_2d,
    save_cluster_distribution,
    save_reconstruction_examples,
    save_scatter,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VAE-based clustering for hybrid language music"
    )
    p.add_argument(
        "--metadata", type=str, required=True, help="Path to data/metadata.csv"
    )
    p.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Column to use as ground-truth labels for ARI/NMI (if present).",
    )
    p.add_argument(
        "--feature",
        type=str,
        default="mfcc",
        choices=[
            "mfcc",
            "mfcc_frames",
            "logmelspec",
            "lyrics_tfidf",
            "hybrid",
            "multimodal",
        ],
        help="Feature type",
    )
    p.add_argument(
        "--audio-feature",
        type=str,
        default="mfcc",
        choices=["mfcc", "mfcc_frames", "logmelspec"],
        help="Audio feature to use when --feature hybrid",
    )
    p.add_argument(
        "--lyrics-embed-dim",
        type=int,
        default=128,
        help="Lyrics embedding dim for hybrid mode (TF-IDF -> SVD)",
    )

    p.add_argument(
        "--genre-col",
        type=str,
        default="genre",
        help="Column name for genre/category info (used by --feature multimodal)",
    )

    p.add_argument(
        "--cond-col",
        type=str,
        default="",
        help="If set, trains a conditional VAE conditioned on this metadata column (one-hot).",
    )
    p.add_argument(
        "--cache",
        type=str,
        default="data/.cache/features_mfcc",
        help="Feature cache dir",
    )

    p.add_argument(
        "--vae-arch",
        type=str,
        default="mlp",
        choices=["mlp", "conv2d"],
        help="VAE architecture (mlp for vectors, conv2d for time-frequency tensors)",
    )

    p.add_argument(
        "--tf-frames",
        type=int,
        default=512,
        help="Target frames for mfcc_frames/logmelspec",
    )
    p.add_argument("--n-mfcc", type=int, default=40, help="n_mfcc for mfcc_frames")
    p.add_argument("--n-mels", type=int, default=64, help="n_mels for logmelspec")

    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument(
        "--baseline",
        type=str,
        default="none",
        choices=["none", "pca", "ae", "raw"],
        help="Baseline method",
    )

    p.add_argument(
        "--cluster-method",
        type=str,
        default="kmeans",
        choices=["kmeans", "agglomerative", "dbscan", "spectral"],
    )
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--dbscan-eps", type=float, default=0.5)
    p.add_argument("--dbscan-min-samples", type=int, default=5)

    p.add_argument(
        "--spectral-n-neighbors",
        type=int,
        default=10,
        help="n_neighbors for spectral clustering (affinity=nearest_neighbors)",
    )

    p.add_argument("--viz", type=str, default="umap", choices=["umap", "tsne", "none"])
    p.add_argument(
        "--save-recon",
        action="store_true",
        help="Save reconstruction example plots (only for VAE/AE runs)",
    )
    p.add_argument(
        "--n-recon", type=int, default=8, help="Number of reconstruction examples"
    )
    p.add_argument(
        "--dist-cols",
        type=str,
        default="language",
        help="Comma-separated metadata columns for cluster distribution plots",
    )
    p.add_argument("--outdir", type=str, default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.metadata)

    def _labels_for(ids_local: list[str]) -> list[str] | None:
        label_col = str(getattr(args, "label_col", "label") or "label")
        if label_col not in df.columns:
            return None

        label_map = {
            str(row["id"]): row.get(label_col, None) for _, row in df.iterrows()
        }
        y_true = [label_map.get(track_id, None) for track_id in ids_local]
        if all(
            v is not None and not (isinstance(v, float) and np.isnan(v)) for v in y_true
        ):
            return [str(v) for v in y_true]
        return None

    feature = args.feature.lower()
    mean = None
    std = None

    # Optional conditioning vector (CVAE)
    c_std = None

    if feature == "mfcc":
        x, ids = build_feature_matrix_mfcc(
            repo_root=repo_root, df=df, cache_dir=Path(args.cache)
        )
        x_std, mean, std = standardize_features(x)

    elif feature == "lyrics_tfidf":
        x, ids = build_feature_matrix_lyrics_tfidf(df=df)
        x_std, mean, std = standardize_features(x)

    elif feature == "mfcc_frames":
        x, ids = build_feature_tensor_mfcc_frames(
            repo_root=repo_root,
            df=df,
            cache_dir=Path(args.cache),
            n_mfcc=int(args.n_mfcc),
            target_frames=int(args.tf_frames),
        )
        mean = x.mean(axis=0, keepdims=True)
        std = np.maximum(x.std(axis=0, keepdims=True), 1e-8)
        x_std = (x - mean) / std

    elif feature == "logmelspec":
        x, ids = build_feature_tensor_logmelspec(
            repo_root=repo_root,
            df=df,
            cache_dir=Path(args.cache),
            n_mels=int(args.n_mels),
            target_frames=int(args.tf_frames),
        )
        mean = x.mean(axis=0, keepdims=True)
        std = np.maximum(x.std(axis=0, keepdims=True), 1e-8)
        x_std = (x - mean) / std

    elif feature == "hybrid":
        audio_feat = args.audio_feature.lower()

        # Audio embedding from a VAE
        if audio_feat == "mfcc":
            x_a, ids_a = build_feature_matrix_mfcc(
                repo_root=repo_root, df=df, cache_dir=Path(args.cache)
            )
            x_a, _, _ = standardize_features(x_a)
            vae_cfg_a = VAEConfig(
                arch="mlp",
                input_dim=int(x_a.shape[1]),
                latent_dim=int(args.latent_dim),
                beta=float(args.beta),
            )
            train_cfg_a = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
            model_a = train_vae(x_a, vae_cfg_a, train_cfg_a)
            z_a = encode_dataset(model_a, x_a, batch_size=256, device=args.device)
        else:
            if audio_feat == "mfcc_frames":
                x_a, ids_a = build_feature_tensor_mfcc_frames(
                    repo_root=repo_root,
                    df=df,
                    cache_dir=Path(args.cache),
                    n_mfcc=int(args.n_mfcc),
                    target_frames=int(args.tf_frames),
                )
            elif audio_feat == "logmelspec":
                x_a, ids_a = build_feature_tensor_logmelspec(
                    repo_root=repo_root,
                    df=df,
                    cache_dir=Path(args.cache),
                    n_mels=int(args.n_mels),
                    target_frames=int(args.tf_frames),
                )
            else:
                raise ValueError(
                    f"Unknown audio feature for hybrid: {args.audio_feature}"
                )

            mean_a = x_a.mean(axis=0, keepdims=True)
            std_a = np.maximum(x_a.std(axis=0, keepdims=True), 1e-8)
            x_a = (x_a - mean_a) / std_a
            vae_cfg_a = VAEConfig(
                arch="conv2d",
                input_shape=tuple(int(v) for v in x_a.shape[1:]),
                latent_dim=int(args.latent_dim),
                beta=float(args.beta),
            )
            train_cfg_a = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
            model_a = train_vae(x_a, vae_cfg_a, train_cfg_a)
            z_a = encode_dataset(model_a, x_a, batch_size=128, device=args.device)

        # Lyrics embedding via TF-IDF -> SVD
        z_l, ids_l = build_lyrics_embedding_svd(
            df=df, n_components=int(args.lyrics_embed_dim)
        )
        z_l, _, _ = standardize_features(z_l)

        # Align by intersection of IDs
        idx_a = {tid: i for i, tid in enumerate(ids_a)}
        idx_l = {tid: i for i, tid in enumerate(ids_l)}
        ids = [tid for tid in ids_a if tid in idx_l]
        if not ids:
            raise ValueError(
                "Hybrid mode found no overlapping IDs with both audio_path and lyrics. "
                "Provide a manifest where the same IDs have both audio and lyrics."
            )
        z_a2 = np.stack([z_a[idx_a[tid]] for tid in ids], axis=0)
        z_l2 = np.stack([z_l[idx_l[tid]] for tid in ids], axis=0)
        x_std = np.concatenate([z_a2, z_l2], axis=1)
        x_std, mean, std = standardize_features(x_std)

    elif feature == "multimodal":
        # Audio embedding from a VAE (same as hybrid), plus lyrics embedding, plus genre one-hot.
        audio_feat = args.audio_feature.lower()

        if audio_feat == "mfcc":
            x_a, ids_a = build_feature_matrix_mfcc(
                repo_root=repo_root, df=df, cache_dir=Path(args.cache)
            )
            x_a, _, _ = standardize_features(x_a)
            vae_cfg_a = VAEConfig(
                arch="mlp",
                input_dim=int(x_a.shape[1]),
                latent_dim=int(args.latent_dim),
                beta=float(args.beta),
            )
            train_cfg_a = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
            model_a = train_vae(x_a, vae_cfg_a, train_cfg_a)
            z_a = encode_dataset(model_a, x_a, batch_size=256, device=args.device)
        else:
            if audio_feat == "mfcc_frames":
                x_a, ids_a = build_feature_tensor_mfcc_frames(
                    repo_root=repo_root,
                    df=df,
                    cache_dir=Path(args.cache),
                    n_mfcc=int(args.n_mfcc),
                    target_frames=int(args.tf_frames),
                )
            elif audio_feat == "logmelspec":
                x_a, ids_a = build_feature_tensor_logmelspec(
                    repo_root=repo_root,
                    df=df,
                    cache_dir=Path(args.cache),
                    n_mels=int(args.n_mels),
                    target_frames=int(args.tf_frames),
                )
            else:
                raise ValueError(
                    f"Unknown audio feature for multimodal: {args.audio_feature}"
                )

            mean_a = x_a.mean(axis=0, keepdims=True)
            std_a = np.maximum(x_a.std(axis=0, keepdims=True), 1e-8)
            x_a = (x_a - mean_a) / std_a
            vae_cfg_a = VAEConfig(
                arch="conv2d",
                input_shape=tuple(int(v) for v in x_a.shape[1:]),
                latent_dim=int(args.latent_dim),
                beta=float(args.beta),
            )
            train_cfg_a = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
            model_a = train_vae(x_a, vae_cfg_a, train_cfg_a)
            z_a = encode_dataset(model_a, x_a, batch_size=128, device=args.device)

        z_l, ids_l = build_lyrics_embedding_svd(
            df=df, n_components=int(args.lyrics_embed_dim)
        )
        z_l, _, _ = standardize_features(z_l)

        # Align by intersection of IDs
        idx_a = {tid: i for i, tid in enumerate(ids_a)}
        idx_l = {tid: i for i, tid in enumerate(ids_l)}
        ids0 = [tid for tid in ids_a if tid in idx_l]
        if not ids0:
            raise ValueError(
                "Multimodal mode found no overlapping IDs with both audio_path and lyrics. "
                "Provide a manifest where the same IDs have both audio and lyrics."
            )
        z_a2 = np.stack([z_a[idx_a[tid]] for tid in ids0], axis=0)
        z_l2 = np.stack([z_l[idx_l[tid]] for tid in ids0], axis=0)

        # Genre/category one-hot (keep rows; missing values become 'unknown')
        g, ids_g, _ = build_onehot_from_column(
            df=df, ids=ids0, col=str(args.genre_col), drop_missing=False
        )
        idx_g = {tid: i for i, tid in enumerate(ids_g)}
        idx0 = {tid: i for i, tid in enumerate(ids0)}
        ids = [tid for tid in ids0 if tid in idx_g]
        if not ids:
            raise ValueError(
                "Multimodal mode found no overlapping IDs with genre values. "
                "Provide a manifest with a populated genre/category column."
            )
        z_a3 = np.stack([z_a2[idx0[tid]] for tid in ids], axis=0)
        z_l3 = np.stack([z_l2[idx0[tid]] for tid in ids], axis=0)
        g3 = np.stack([g[idx_g[tid]] for tid in ids], axis=0)

        x_std = np.concatenate([z_a3, z_l3, g3], axis=1)
        x_std, mean, std = standardize_features(x_std)

    else:
        raise ValueError(f"Unsupported feature: {args.feature}")

    y_true_clean = _labels_for(ids)

    # Build conditioning one-hot aligned to ids (if requested)
    cond_col = str(getattr(args, "cond_col", "") or "").strip()
    if cond_col:
        c_raw, ids_c, _ = build_onehot_from_column(
            df=df, ids=ids, col=cond_col, drop_missing=True
        )
        idx_c = {tid: i for i, tid in enumerate(ids_c)}
        ids2 = [tid for tid in ids if tid in idx_c]
        if not ids2:
            raise ValueError(
                f"No conditioning values found for --cond-col {cond_col!r}"
            )
        keep = [ids.index(tid) for tid in ids2]
        x_std = x_std[keep]
        if mean is not None and std is not None:
            mean = mean
            std = std
        ids = ids2
        if y_true_clean is not None:
            y_true_clean = [y_true_clean[i] for i in keep]
        c_std = c_raw[[idx_c[tid] for tid in ids2]]

        # Standardize one-hot (helps when concatenated internally)
        c_std, _, _ = standardize_features(c_std)

    run_tag = "vae" if args.baseline == "none" else str(args.baseline)

    if args.baseline == "pca":
        if x_std.ndim == 4:
            x_std = x_std.reshape(x_std.shape[0], -1)
        elif x_std.ndim != 2:
            raise ValueError(
                f"PCA baseline expects 2D or 4D features, got rank={x_std.ndim}"
            )
        n_components = min(32, int(x_std.shape[0]), int(x_std.shape[1]))
        pca = PCA(n_components=n_components, random_state=42)
        z = pca.fit_transform(x_std)
    elif args.baseline == "raw":
        if x_std.ndim == 4:
            z = x_std.reshape(x_std.shape[0], -1)
        elif x_std.ndim == 2:
            z = x_std
        else:
            raise ValueError(
                f"Raw baseline expects 2D or 4D features, got rank={x_std.ndim}"
            )

    else:
        arch = args.vae_arch.lower()

        if x_std.ndim == 2:
            # Vector features always use MLP VAE (conv2d expects 4D tensors).
            arch = "mlp"
            vae_cfg = VAEConfig(
                arch=arch,
                input_dim=int(x_std.shape[1]),
                latent_dim=int(args.latent_dim),
                beta=float(args.beta),
                condition_dim=int(c_std.shape[1]) if c_std is not None else 0,
                deterministic=bool(args.baseline == "ae"),
                use_kl=bool(args.baseline != "ae"),
            )
        elif x_std.ndim == 4:
            if arch != "conv2d":
                raise ValueError("For mfcc_frames/logmelspec, set --vae-arch conv2d")
            vae_cfg = VAEConfig(
                arch=arch,
                input_shape=tuple(int(v) for v in x_std.shape[1:]),
                latent_dim=int(args.latent_dim),
                beta=float(args.beta),
                condition_dim=int(c_std.shape[1]) if c_std is not None else 0,
                deterministic=bool(args.baseline == "ae"),
                use_kl=bool(args.baseline != "ae"),
            )
        else:
            raise ValueError(f"Unexpected feature tensor rank: {x_std.ndim}")

        train_cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        model = train_vae(x_std, vae_cfg, train_cfg, c_train=c_std)
        z = encode_dataset(model, x_std, c=c_std, batch_size=256, device=args.device)

        if bool(args.save_recon):
            save_reconstruction_examples(
                model=model,
                x=x_std,
                c=c_std,
                out_dir=outdir,
                n=int(args.n_recon),
                device=args.device,
                title=f"{run_tag.upper()} recon ({feature})",
            )

    cluster_cfg = ClusterConfig(
        method=args.cluster_method,
        n_clusters=args.clusters,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
        spectral_n_neighbors=int(args.spectral_n_neighbors),
    )
    clusters = cluster_embeddings(z, cluster_cfg)

    metrics = compute_clustering_metrics(z, clusters, labels_true=y_true_clean)
    metrics_row = {
        "run": run_tag,
        "feature": args.feature,
        "audio_feature": getattr(args, "audio_feature", ""),
        "vae_arch": getattr(args, "vae_arch", ""),
        "cluster_method": args.cluster_method,
        "clusters_requested": int(args.clusters),
        "dbscan_eps": float(args.dbscan_eps),
        "dbscan_min_samples": int(args.dbscan_min_samples),
        "latent_dim": int(args.latent_dim),
        "beta": float(args.beta),
        "cond_col": str(cond_col),
        **metrics,
    }

    metrics_csv = (outdir / "clustering_metrics.csv").as_posix()
    append_metrics_csv(metrics_csv, metrics_row)
    print("metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    np.savez(
        outdir / "latents.npz",
        z=z,
        ids=np.array(ids),
        clusters=clusters,
        mean=mean,
        std=std,
    )

    if args.viz != "none":
        xy = project_2d(z, method=args.viz)
        out_path = (
            outdir
            / "latent_visualization"
            / f"{run_tag}_{args.cluster_method}_{args.viz}.png"
        )
        save_scatter(
            xy, clusters, out_path, title=f"{run_tag.upper()} latent space ({args.viz})"
        )
        print(f"saved plot: {out_path}")

    # Cluster distribution plots (language/genre/etc.)
    dist_cols = [c.strip() for c in str(args.dist_cols).split(",") if c.strip()]
    for col in dist_cols:
        if col in df.columns:
            out_path = outdir / "cluster_distributions" / f"clusters_by_{col}.png"
            save_cluster_distribution(
                df=df, ids=ids, clusters=clusters, group_col=col, out_path=out_path
            )


if __name__ == "__main__":
    main()
