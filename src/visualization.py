from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def _configure_matplotlib_fonts() -> None:
    """Configure Matplotlib with sensible Unicode font fallbacks.

    On Windows, Matplotlib's default DejaVu Sans often lacks Bengali glyphs,
    producing very noisy warnings and unreadable labels. We prefer a Bengali-
    capable system font when available, otherwise fall back to the default.
    """

    try:
        from matplotlib import font_manager

        # Important: do NOT force a single Bengali-only font for all text.
        # Some Bengali fonts contain limited Latin glyph coverage, which can
        # turn even English labels into tofu squares. Instead, provide an
        # ordered fallback list and let Matplotlib/fontconfig resolve glyphs.
        preferred = [
            # Good Latin coverage
            "Noto Sans",
            "DejaVu Sans",
            "Liberation Sans",
            # Bengali-capable fonts
            "Noto Sans Bengali UI",
            "Noto Sans Bengali",
            "Noto Serif Bengali",
            # Windows (ships with Bengali support)
            "Nirmala UI",
            # Older installs
            "Arial Unicode MS",
        ]

        available: list[str] = []
        for name in preferred:
            try:
                prop = font_manager.FontProperties(family=name)
                font_manager.findfont(prop, fallback_to_default=False)
                available.append(name)
            except Exception:
                continue

        if available:
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = available
    except Exception:
        # If font discovery fails for any reason, keep defaults.
        pass

    # Avoid Unicode minus rendering issues for some fonts.
    mpl.rcParams.setdefault("axes.unicode_minus", False)


_configure_matplotlib_fonts()


def _save_figure(fig, out_path: Path) -> None:
    import warnings

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Matplotlib can emit noisy layout/font warnings when saving figures with
    # long Unicode tick labels (e.g., Bengali categories). We rely on bbox_inches
    # and suppress these non-actionable warnings to keep runs clean.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*Tight layout not applied.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*constrained_layout not applied.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Glyph .* missing from font.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Matplotlib currently does not support Bengali.*",
            category=UserWarning,
        )
        fig.savefig(out_path.as_posix(), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_cluster_distribution(
    df,
    ids: list[str],
    clusters: np.ndarray,
    group_col: str,
    out_path: Path,
) -> None:
    """Save a stacked bar plot of cluster counts split by a metadata column."""

    if group_col not in df.columns:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    group_map = {str(row["id"]): row.get(group_col, None) for _, row in df.iterrows()}
    groups: list[str] = []
    for tid in ids:
        v = group_map.get(str(tid), None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            groups.append("(missing)")
        else:
            s = str(v).strip()
            groups.append(s if s else "(missing)")

    uniq_clusters = [int(c) for c in np.unique(clusters) if int(c) >= 0]
    if not uniq_clusters:
        uniq_clusters = [int(c) for c in np.unique(clusters)]

    uniq_groups = sorted(set(groups))

    # Build count matrix: (n_groups, n_clusters)
    counts = np.zeros((len(uniq_groups), len(uniq_clusters)), dtype=int)
    g_index = {g: i for i, g in enumerate(uniq_groups)}
    c_index = {c: i for i, c in enumerate(uniq_clusters)}
    for g, c in zip(groups, clusters, strict=False):
        c_int = int(c)
        if c_int not in c_index:
            continue
        counts[g_index[g], c_index[c_int]] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(uniq_groups), dtype=int)
    for j, c in enumerate(uniq_clusters):
        ax.bar(uniq_groups, counts[:, j], bottom=bottom, label=f"cluster {c}")
        bottom += counts[:, j]
    ax.set_title(f"Cluster distribution by {group_col}")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    _save_figure(fig, out_path)


def save_reconstruction_examples(
    model,
    x: np.ndarray,
    c: np.ndarray | None,
    out_dir: Path,
    n: int = 8,
    device: str = "cpu",
    title: str = "Reconstruction examples",
) -> None:
    """Save a small grid of original vs reconstructed examples."""

    import torch

    out_path = out_dir / "reconstructions" / "recon_examples.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = int(max(1, min(n, x.shape[0])))
    idx = np.linspace(0, x.shape[0] - 1, n, dtype=int)
    xb = torch.from_numpy(x[idx]).float().to(device)
    cb = None
    if c is not None:
        cb = torch.from_numpy(c[idx]).float().to(device)

    model.eval()
    model.to(device)
    with torch.no_grad():
        recon, _, _, _ = model(xb, cb)
    recon_np = recon.detach().cpu().numpy()
    orig_np = xb.detach().cpu().numpy()

    # Vector features: plot as line charts
    if orig_np.ndim == 2:
        fig, axes = plt.subplots(n, 1, figsize=(10, max(2, n * 1.4)), sharex=True)
        if n == 1:
            axes = [axes]
        for i in range(n):
            axes[i].plot(orig_np[i], label="orig", linewidth=1)
            axes[i].plot(recon_np[i], label="recon", linewidth=1)
            axes[i].set_yticks([])
            if i == 0:
                axes[i].legend(fontsize=8)
        fig.suptitle(title)
        _save_figure(fig, out_path)
        return

    # Time-frequency tensors: show heatmaps (channel 0)
    if orig_np.ndim == 4:
        fig, axes = plt.subplots(n, 2, figsize=(8, max(2, n * 1.5)))
        if n == 1:
            axes = np.array([axes])
        for i in range(n):
            o = orig_np[i, 0]
            r = recon_np[i, 0]
            axes[i, 0].imshow(o, aspect="auto", origin="lower")
            axes[i, 0].set_title("orig", fontsize=8)
            axes[i, 0].axis("off")
            axes[i, 1].imshow(r, aspect="auto", origin="lower")
            axes[i, 1].set_title("recon", fontsize=8)
            axes[i, 1].axis("off")
        fig.suptitle(title)
        _save_figure(fig, out_path)
        return


def project_2d(
    z: np.ndarray, method: str = "umap", random_state: int = 42
) -> np.ndarray:
    method = method.lower()

    if method == "umap":
        # On some Windows terminals, numba's error formatting can interact badly
        # with console VT processing. Disabling the highlighting keeps failures
        # from crashing the whole run.
        import os

        os.environ.setdefault("NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING", "1")

        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=random_state)
            return reducer.fit_transform(z)
        except Exception as e:
            print(f"[viz] UMAP failed ({type(e).__name__}: {e}); falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE

        # sklearn requires perplexity < n_samples
        n = int(z.shape[0])
        # Keep a reasonable default but safe for small datasets.
        # Common heuristic: perplexity ~ n/3, and must be < n.
        perplexity = min(30.0, max(5.0, (n - 1) / 3.0))
        if perplexity >= n:
            perplexity = max(1.0, float(n - 1))

        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
        )
        return reducer.fit_transform(z)

    raise ValueError(f"Unknown projection method: {method}")


def save_scatter(
    xy: np.ndarray,
    clusters: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = np.unique(clusters)
    for c in uniq:
        mask = clusters == c
        label = f"cluster {c}" if c >= 0 else "noise"
        ax.scatter(xy[mask, 0], xy[mask, 1], s=18, alpha=0.8, label=label)

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(markerscale=1.2, fontsize=9)
    _save_figure(fig, out_path)
