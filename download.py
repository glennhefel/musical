#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def _looks_like_placeholder(p: Path) -> bool:
    """Detect the tiny placeholder files created by earlier attempts."""
    try:
        if not p.exists() or not p.is_file():
            return False
        if p.stat().st_size >= 1024:
            return False
        head = p.open("rb").read(16)
        return head.startswith(b"../") or head.startswith(b"subsets")
    except Exception:
        return False


def _copy_file(
    src: Path, dst_dir: Path, *, overwrite_placeholders: bool = True
) -> bool:
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        return False
    dst = dst_dir / src.name
    if dst.exists():
        if overwrite_placeholders and _looks_like_placeholder(dst):
            dst.unlink()
        else:
            return False
    shutil.copy2(src, dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download the jamendolyrics/jamendolyrics dataset from Hugging Face and store "
            "audio in data/audio/ and lyrics in data/lyrics/."
        )
    )
    ap.add_argument(
        "--repo-id",
        default="jamendolyrics/jamendolyrics",
        help="Hugging Face dataset repo id",
    )
    ap.add_argument(
        "--language",
        default="English",
        help=(
            "Keep only this language (matches JamendoLyrics.csv Language column). "
            "Use 'all' to keep every language. Default: English"
        ),
    )
    ap.add_argument(
        "--max-tracks",
        type=int,
        default=0,
        help="Limit number of tracks to download (0 = no limit)",
    )
    ap.add_argument(
        "--data-dir",
        default="data",
        help="Base data directory (audio/ and lyrics/ will be created inside)",
    )
    ap.add_argument(
        "--cache-dir",
        default=".hf_cache",
        help="Local cache directory for Hugging Face snapshot download",
    )
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install with: python -m pip install 'huggingface_hub[cli]'"
        ) from e

    repo_root = Path(__file__).resolve().parent
    data_dir = (repo_root / args.data_dir).resolve()
    audio_dir = data_dir / "audio"
    lyrics_dir = data_dir / "lyrics"
    cache_dir = (repo_root / args.cache_dir).resolve()

    audio_dir.mkdir(parents=True, exist_ok=True)
    lyrics_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading JamendoLyrics from: {args.repo_id}")
    print("If access is gated, run: huggingface-cli login")

    # Download only the metadata CSV from the repo root.
    # Audio/lyrics live under subsets/en/ and must be fetched as real files (not pointer placeholders).
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            cache_dir=cache_dir.as_posix(),
            allow_patterns=["JamendoLyrics.csv"],
        )
    )

    csv_path = snapshot_path / "JamendoLyrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing JamendoLyrics.csv in snapshot at {csv_path}")

    df = pd.read_csv(csv_path)
    if "Filepath" not in df.columns or "Language" not in df.columns:
        raise ValueError(
            f"Unexpected JamendoLyrics.csv columns: {list(df.columns)} (need Filepath and Language)"
        )

    lang = str(args.language)
    if lang.lower() == "all":
        df_lang = df
    else:
        df_lang = df[df["Language"].astype(str) == lang]

    if df_lang.empty:
        available = sorted(df["Language"].dropna().astype(str).unique().tolist())
        raise ValueError(
            f"No rows found for language={args.language!r}. Available languages: {available}"
        )

    if int(args.max_tracks) and int(args.max_tracks) > 0:
        df_lang = df_lang.head(int(args.max_tracks))

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install with: python -m pip install 'huggingface_hub[cli]'"
        ) from e

    copied_audio = 0
    copied_lyrics = 0
    missing_audio = 0
    missing_lyrics = 0
    rows = []

    # JamendoLyrics.csv contains subset entries. Audio paths depend on language.
    # Map language names to subset folder codes.
    lang_to_code = {
        "english": "en",
        "french": "fr",
        "spanish": "es",
        "german": "de",
    }

    for _, r in df_lang.iterrows():
        name = str(r["Filepath"])
        stem = Path(name).stem

        lang_name = str(r.get("Language", "")).strip()
        lang_code = lang_to_code.get(lang_name.lower(), None)
        if lang_code is None:
            # Fallback heuristic: first 2 chars of language name
            lang_code = lang_name.lower()[:2] if lang_name else "en"

        # Real audio path in HF dataset repo
        audio_filename = f"subsets/{lang_code}/mp3/{name}"
        try:
            audio_cached = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename=audio_filename,
                    cache_dir=cache_dir.as_posix(),
                )
            )
            if _copy_file(audio_cached, audio_dir):
                copied_audio += 1
        except Exception:
            missing_audio += 1
            continue

        lyrics_filename = f"lyrics/{stem}.txt"
        lyrics_text = None
        try:
            lyrics_cached = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    filename=lyrics_filename,
                    cache_dir=cache_dir.as_posix(),
                )
            )
            if _copy_file(lyrics_cached, lyrics_dir):
                copied_lyrics += 1
            lyrics_text = lyrics_cached.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            missing_lyrics += 1

        # Build/refresh metadata row
        rows.append(
            {
                "id": stem,
                "audio_path": (audio_dir / name)
                .resolve()
                .relative_to(repo_root)
                .as_posix(),
                "lyrics": lyrics_text,
                "language": lang_code,
                "label": lang_code,
            }
        )

    print(f"Saved audio to: {audio_dir}")
    print(f"Saved lyrics to: {lyrics_dir}")
    print(f"Language filter: {args.language}")
    if int(args.max_tracks) and int(args.max_tracks) > 0:
        print(f"Track limit: {args.max_tracks}")
    print(f"New audio files copied: {copied_audio} (missing: {missing_audio})")
    print(f"New lyrics files copied: {copied_lyrics} (missing: {missing_lyrics})")

    # Write manifest the pipeline expects
    out_csv = data_dir / "metadata.csv"
    pd.DataFrame(rows).to_csv(out_csv.as_posix(), index=False)
    print(f"Wrote metadata: {out_csv}")


if __name__ == "__main__":
    main()
