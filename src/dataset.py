from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrackRecord:
    track_id: str
    audio_path: Optional[str]
    lyrics: Optional[str]
    language: Optional[str]
    genre: Optional[str]
    label: Optional[str]


def load_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)

    if "id" not in df.columns:
        raise ValueError("metadata.csv must have an 'id' column")

    # Normalize optional columns
    for col in [
        "audio_path",
        "lyrics",
        "lyrics_clean",
        "language",
        "genre",
        "label",
        "title",
        "category",
        "artist",
        "source",
    ]:
        if col not in df.columns:
            df[col] = None

    df["id"] = df["id"].astype(str)
    return df


def standardize_features(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std, mean, std
