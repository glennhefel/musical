from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def build_onehot_from_column(
    df,
    ids: list[str],
    col: str,
    drop_missing: bool = True,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a one-hot matrix for a categorical metadata column.

    Returns:
      onehot: (N, D)
      ids_out: ids for rows included
      categories: category names in encoder order

    If drop_missing=True, rows with missing/empty values are dropped.
    """

    from sklearn.preprocessing import OneHotEncoder

    if col not in df.columns:
        raise ValueError(f"Column not found for one-hot encoding: {col}")

    val_map = {str(row["id"]): row.get(col, None) for _, row in df.iterrows()}
    ids_out: list[str] = []
    vals: list[str] = []
    for tid in ids:
        v = val_map.get(str(tid), None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            if drop_missing:
                continue
            v = "unknown"
        s = str(v).strip()
        if not s:
            if drop_missing:
                continue
            s = "unknown"
        ids_out.append(str(tid))
        vals.append(s)

    if not vals:
        raise ValueError(f"No values found for one-hot encoding col={col!r}")

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    onehot = enc.fit_transform(np.array(vals, dtype=object).reshape(-1, 1)).astype(np.float32)
    cats = [str(c) for c in enc.categories_[0].tolist()]
    return onehot, ids_out, cats


def build_lyrics_embedding_svd(
    df,
    text_col: str = "lyrics_clean",
    fallback_text_col: str = "lyrics",
    n_components: int = 128,
    max_features: int = 20000,
    analyzer: str = "char_wb",
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Build a compact lyrics embedding using TF-IDF -> TruncatedSVD.

    This is intended as a lightweight lyrics embedding for hybrid (audio+lyrics)
    experiments.
    """

    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    ids: list[str] = []
    texts: list[str] = []

    for _, row in df.iterrows():
        track_id = str(row["id"])
        val = row.get(text_col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = row.get(fallback_text_col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        s = str(val).strip()
        if not s:
            continue
        ids.append(track_id)
        texts.append(s)

    if not texts:
        raise ValueError(
            f"No lyrics found. Expected non-empty '{text_col}' (or '{fallback_text_col}')."
        )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=False,
    )
    x_tfidf = vectorizer.fit_transform(texts)

    # TruncatedSVD requires n_components < n_features
    n_comp = int(min(n_components, max(2, x_tfidf.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    x = svd.fit_transform(x_tfidf).astype(np.float32)
    return x, ids


def build_feature_matrix_lyrics_tfidf(
    df,
    text_col: str = "lyrics_clean",
    fallback_text_col: str = "lyrics",
    max_features: int = 5000,
    analyzer: str = "char_wb",
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """Build a dense TF-IDF feature matrix from lyrics.

    Designed for mixed-language text (English+Bangla) by defaulting to
    character n-grams within word boundaries (char_wb).

    Rows with missing/empty lyrics are skipped.
    Returns:
      x: float32 array of shape (n_samples, max_features)
      ids: list of track IDs aligned with x
    """

    from sklearn.feature_extraction.text import TfidfVectorizer

    ids: list[str] = []
    texts: list[str] = []

    for _, row in df.iterrows():
        track_id = str(row["id"])
        val = row.get(text_col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = row.get(fallback_text_col, None)

        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue

        s = str(val).strip()
        if not s:
            continue

        ids.append(track_id)
        texts.append(s)

    if not texts:
        raise ValueError(
            f"No lyrics found. Expected non-empty '{text_col}' (or '{fallback_text_col}')."
        )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=False,
    )

    x_sparse = vectorizer.fit_transform(texts)
    x = x_sparse.astype(np.float32).toarray()
    return x, ids


def _safe_path(repo_root: Path, maybe_path: Optional[str]) -> Optional[Path]:
    if maybe_path is None or (isinstance(maybe_path, float) and np.isnan(maybe_path)):
        return None
    s = str(maybe_path).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def extract_mfcc_features(
    repo_root: Path,
    audio_path: str,
    sample_rate: int = 22050,
    n_mfcc: int = 20,
    hop_length: int = 512,
    n_fft: int = 2048,
    max_seconds: float = 30.0,
) -> np.ndarray:
    """Return a fixed-length MFCC feature vector using summary stats.

    Output is shape (n_mfcc * 4,): mean, std, min, max per MFCC coefficient.
    """

    import librosa

    path = _safe_path(repo_root, audio_path)
    if path is None or (not path.exists()) or (not path.is_file()):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(path.as_posix(), sr=sample_rate, mono=True, duration=max_seconds)
    if y.size == 0:
        raise ValueError(f"Empty audio after loading: {audio_path}")

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    # mfcc shape: (n_mfcc, frames)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    mn = mfcc.min(axis=1)
    mx = mfcc.max(axis=1)
    feat = np.concatenate([mean, std, mn, mx], axis=0).astype(np.float32)
    return feat


def _pad_or_truncate_2d(x: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad or truncate time axis of a (freq, frames) array to target_frames."""

    if x.shape[1] == target_frames:
        return x
    if x.shape[1] > target_frames:
        return x[:, :target_frames]
    pad = target_frames - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="constant")


def build_feature_tensor_mfcc_frames(
    repo_root: Path,
    df,
    cache_dir: Optional[Path] = None,
    sample_rate: int = 22050,
    n_mfcc: int = 40,
    hop_length: int = 512,
    n_fft: int = 2048,
    max_seconds: float = 30.0,
    target_frames: int = 512,
) -> tuple[np.ndarray, list[str]]:
    """Build a 4D tensor (N, 1, n_mfcc, target_frames) from MFCC frames."""

    import librosa

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    tensors: list[np.ndarray] = []
    ids: list[str] = []

    for _, row in df.iterrows():
        track_id = str(row["id"])
        audio_path = row.get("audio_path", None)
        if audio_path is None or (isinstance(audio_path, float) and np.isnan(audio_path)):
            continue
        audio_path_str = str(audio_path).strip()
        if not audio_path_str:
            continue

        cached = None
        if cache_dir is not None:
            cached_path = cache_dir / f"{track_id}_mfccframes.npy"
            if cached_path.exists():
                cached = np.load(cached_path)

        if cached is None:
            path = _safe_path(repo_root, audio_path_str)
            if path is None or (not path.exists()) or (not path.is_file()):
                continue

            y, sr = librosa.load(
                path.as_posix(),
                sr=sample_rate,
                mono=True,
                duration=max_seconds,
            )
            if y.size == 0:
                continue

            mfcc = librosa.feature.mfcc(
                y=y,
                sr=sr,
                n_mfcc=n_mfcc,
                hop_length=hop_length,
                n_fft=n_fft,
            )
            mfcc = _pad_or_truncate_2d(mfcc, target_frames)
            t = mfcc.astype(np.float32)[None, :, :]  # (1, n_mfcc, frames)

            if cache_dir is not None:
                np.save(cached_path, t)
        else:
            t = cached

        tensors.append(t)
        ids.append(track_id)

    if not tensors:
        raise ValueError("No MFCC-frame tensors extracted. Check metadata audio_path values.")

    x = np.stack(tensors, axis=0)  # (N, 1, n_mfcc, frames)
    return x, ids


def build_feature_tensor_logmelspec(
    repo_root: Path,
    df,
    cache_dir: Optional[Path] = None,
    sample_rate: int = 22050,
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 2048,
    max_seconds: float = 30.0,
    target_frames: int = 512,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> tuple[np.ndarray, list[str]]:
    """Build a 4D tensor (N, 1, n_mels, target_frames) from log-mel spectrogram."""

    import librosa

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    tensors: list[np.ndarray] = []
    ids: list[str] = []

    for _, row in df.iterrows():
        track_id = str(row["id"])
        audio_path = row.get("audio_path", None)
        if audio_path is None or (isinstance(audio_path, float) and np.isnan(audio_path)):
            continue
        audio_path_str = str(audio_path).strip()
        if not audio_path_str:
            continue

        cached = None
        if cache_dir is not None:
            cached_path = cache_dir / f"{track_id}_logmelspec.npy"
            if cached_path.exists():
                cached = np.load(cached_path)

        if cached is None:
            path = _safe_path(repo_root, audio_path_str)
            if path is None or (not path.exists()) or (not path.is_file()):
                continue

            y, sr = librosa.load(
                path.as_posix(),
                sr=sample_rate,
                mono=True,
                duration=max_seconds,
            )
            if y.size == 0:
                continue

            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax,
                power=2.0,
            )
            logmel = librosa.power_to_db(mel, ref=np.max)
            logmel = _pad_or_truncate_2d(logmel, target_frames)
            t = logmel.astype(np.float32)[None, :, :]  # (1, n_mels, frames)

            if cache_dir is not None:
                np.save(cached_path, t)
        else:
            t = cached

        tensors.append(t)
        ids.append(track_id)

    if not tensors:
        raise ValueError("No log-mel tensors extracted. Check metadata audio_path values.")

    x = np.stack(tensors, axis=0)  # (N, 1, n_mels, frames)
    return x, ids


def build_feature_matrix_mfcc(
    repo_root: Path,
    df,
    cache_dir: Optional[Path] = None,
    **mfcc_kwargs,
) -> tuple[np.ndarray, list[str]]:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    features: list[np.ndarray] = []
    ids: list[str] = []

    for _, row in df.iterrows():
        track_id = str(row["id"])
        audio_path = row.get("audio_path", None)
        if audio_path is None or (isinstance(audio_path, float) and np.isnan(audio_path)):
            continue
        audio_path_str = str(audio_path).strip()
        if not audio_path_str:
            continue

        cached = None
        if cache_dir is not None:
            cached_path = cache_dir / f"{track_id}.npy"
            if cached_path.exists():
                cached = np.load(cached_path)

        if cached is None:
            feat = extract_mfcc_features(repo_root=repo_root, audio_path=audio_path_str, **mfcc_kwargs)
            if cache_dir is not None:
                np.save(cache_dir / f"{track_id}.npy", feat)
        else:
            feat = cached

        features.append(feat)
        ids.append(track_id)

    if not features:
        raise ValueError("No features extracted. Check metadata.csv audio_path values.")

    x = np.stack(features, axis=0)
    return x, ids
