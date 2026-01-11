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

    # Derived language grouping used by some experiment suites.
    # Goal: treat English/German/French/Italian as one group and Bangla as another.
    # Any other or missing languages are mapped to stable catch-all groups so
    # downstream evaluation/plots always have a label for every row.
    def _normalize_lang(value: object) -> str:
        if value is None:
            return ""
        try:
            text = str(value)
        except Exception:
            return ""

        text = text.strip().lower()
        if not text:
            return ""

        # handle tags like en-US, bn_BD, etc.
        for sep in ("-", "_"):
            if sep in text:
                text = text.split(sep, 1)[0]
                break
        return text

    lang_norm = df["language"].map(_normalize_lang)

    def _language_group(value: str) -> str:
        # All non-bangla languages (including unknown/other) are mapped to 'germanic'.
        if value in {"bn", "ben", "bangla", "bengali"}:
            return "bangla"
        return "germanic"

    df["language_group"] = lang_norm.map(_language_group)
    return df


def standardize_features(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    return (x - mean) / std, mean, std
