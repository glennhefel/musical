from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _flag_failure_modes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize expected columns (they should exist, but keep this resilient).
    for c in [
        "n",
        "n_clusters_eff",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "ari",
        "nmi",
        "purity",
    ]:
        if c in df.columns:
            df[c] = _as_float(df[c])

    cluster_method = df.get("cluster_method", pd.Series([""] * len(df)))

    n_clusters_eff = df.get("n_clusters_eff", pd.Series([np.nan] * len(df)))
    df["flag_degenerate_clusters"] = n_clusters_eff.fillna(0).astype(int) < 2

    is_dbscan = cluster_method.astype(str).str.lower().eq("dbscan")
    df["flag_dbscan_all_noise"] = is_dbscan & (
        n_clusters_eff.fillna(0).astype(int) == 0
    )

    # Internal metrics are undefined if <2 non-noise clusters.
    internal_cols = [
        c
        for c in ["silhouette", "calinski_harabasz", "davies_bouldin"]
        if c in df.columns
    ]
    if internal_cols:
        df["flag_nan_internal"] = df[internal_cols].isna().any(axis=1)
    else:
        df["flag_nan_internal"] = True

    # Supervised-ish metrics vs label_col can also be NaN in degenerate cases.
    supervised_cols = [c for c in ["ari", "nmi", "purity"] if c in df.columns]
    if supervised_cols:
        df["flag_nan_supervised"] = df[supervised_cols].isna().any(axis=1)
    else:
        df["flag_nan_supervised"] = False

    # Helpful combined flag string.
    flag_cols = [
        "flag_dbscan_all_noise",
        "flag_degenerate_clusters",
        "flag_nan_internal",
        "flag_nan_supervised",
    ]
    labels = {
        "flag_dbscan_all_noise": "DBSCAN_ALL_NOISE",
        "flag_degenerate_clusters": "<2_CLUSTERS",
        "flag_nan_internal": "NAN_INTERNAL",
        "flag_nan_supervised": "NAN_SUPERVISED",
    }

    def _join_flags(row: pd.Series) -> str:
        active = [labels[c] for c in flag_cols if bool(row.get(c, False))]
        return "|".join(active) if active else "OK"

    df["failure_flags"] = df.apply(_join_flags, axis=1)
    return df


def _combined_rank(df: pd.DataFrame) -> pd.Series:
    """Lower is better."""

    def rank_metric(col: str, ascending: bool) -> pd.Series:
        s = _as_float(df[col])
        # NaNs should sort to the bottom.
        ranked = s.rank(ascending=ascending, method="average", na_option="bottom")
        # If all NaN, rank() returns all NaN; treat as worst.
        if ranked.isna().all():
            return pd.Series([float(len(df))] * len(df), index=df.index)
        return ranked.fillna(float(len(df)))

    parts: list[pd.Series] = []
    if "silhouette" in df.columns:
        parts.append(rank_metric("silhouette", ascending=False))
    if "calinski_harabasz" in df.columns:
        parts.append(rank_metric("calinski_harabasz", ascending=False))
    if "davies_bouldin" in df.columns:
        parts.append(rank_metric("davies_bouldin", ascending=True))

    if not parts:
        return pd.Series([float(len(df))] * len(df), index=df.index)

    return pd.concat(parts, axis=1).mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze results/results_medium_task/medium_task_comparison.csv: "
            "print ranked summary and common failure modes (e.g., DBSCAN all-noise -> NaNs)."
        )
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("results/results_medium_task/medium_task_comparison.csv"),
        help="Path to medium_task_comparison.csv",
    )
    ap.add_argument("--top", type=int, default=15, help="How many top rows to print")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if df.empty:
        raise SystemExit(f"CSV is empty: {args.csv}")

    df = _flag_failure_modes(df)

    # Prefer rows that are not obviously degenerate.
    ok_mask = ~df["flag_degenerate_clusters"].fillna(True)
    ok = df[ok_mask].copy()
    bad = df[~ok_mask].copy()

    if not ok.empty:
        ok["combined_rank"] = _combined_rank(ok)
        ok = ok.sort_values(["combined_rank"], ascending=True)

    # Output
    print(f"Loaded {len(df)} rows from: {args.csv}")

    cols_preferred = [
        "run_dir",
        "run",
        "feature",
        "audio_feature",
        "cluster_method",
        "n",
        "n_clusters_eff",
        "silhouette",
        "calinski_harabasz",
        "davies_bouldin",
        "ari",
        "nmi",
        "purity",
        "dbscan_eps",
        "dbscan_min_samples",
        "failure_flags",
    ]
    cols = [c for c in cols_preferred if c in df.columns]

    if not ok.empty:
        print("\nTop ranked (non-degenerate) runs:")
        view = ok[cols].head(int(args.top)).copy()
        # Compact numeric formatting.
        for c in [
            "silhouette",
            "calinski_harabasz",
            "davies_bouldin",
            "ari",
            "nmi",
            "purity",
        ]:
            if c in view.columns:
                view[c] = _as_float(view[c]).map(
                    lambda v: f"{v:.6f}" if pd.notna(v) else "nan"
                )
        print(view.to_string(index=False))
    else:
        print("\nNo non-degenerate runs found (all have <2 effective clusters).")

    # Failure mode counts
    print("\nFailure mode counts:")
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    if flag_cols:
        counts = {c: int(df[c].fillna(False).astype(bool).sum()) for c in flag_cols}
        for k in sorted(counts.keys()):
            print(f"  {k}: {counts[k]}")

    if not bad.empty:
        print("\nDegenerate runs (<2 effective clusters):")
        view = bad[cols].copy()
        if "cluster_method" in view.columns:
            view = view.sort_values(["cluster_method"], ascending=True)
        print(view.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
