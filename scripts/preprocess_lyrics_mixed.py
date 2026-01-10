"""Preprocess lyrics text while keeping English + Bangla mixed.

Creates a new CSV with an extra `lyrics_clean` column.

Default input is the known-categories dataset so you keep only labeled rows,
but you can point it at any metadata CSV.

What it does (language-agnostic):
- Unicode normalize (NFKC)
- Remove zero-width chars
- Normalize line endings
- Drop section tag lines like [Chorus], [Verse 1], (Chorus)
- Collapse repeated spaces/tabs inside lines
- Collapse 3+ blank lines to at most 1 blank line

Usage:
  python scripts/preprocess_lyrics_mixed.py
  python scripts/preprocess_lyrics_mixed.py --in data/metadata.csv --out data/metadata_preprocessed.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")
INLINE_WS_RE = re.compile(r"[\t\f\v ]+")
BLANKLINES_RE = re.compile(r"\n{3,}")

# Lines that are just section labels like [Chorus], (Verse 2), etc.
SECTION_LINE_RE = re.compile(
    r"^\s*(\[[^\]]{1,40}\]|\([^\)]{1,40}\))\s*$",
    flags=re.IGNORECASE,
)


def clean_lyrics(text: str) -> str:
    if text is None:
        return ""

    s = str(text)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = unicodedata.normalize("NFKC", s)
    s = ZERO_WIDTH_RE.sub("", s)

    out_lines: list[str] = []
    for line in s.split("\n"):
        line = line.strip()
        if not line:
            out_lines.append("")
            continue
        if SECTION_LINE_RE.match(line):
            continue
        line = INLINE_WS_RE.sub(" ", line)
        out_lines.append(line)

    s2 = "\n".join(out_lines).strip()
    s2 = BLANKLINES_RE.sub("\n\n", s2)
    return s2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/metadata_known_categories.csv")
    ap.add_argument("--out", dest="out", default="data/metadata_known_categories_preprocessed.csv")
    ap.add_argument("--min-len", type=int, default=50, help="If lyrics_clean shorter than this, blank it")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    in_path = (repo_root / args.inp).resolve()
    out_path = (repo_root / args.out).resolve()

    with in_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "lyrics" not in fieldnames:
        raise ValueError("Input CSV missing 'lyrics' column")

    if "lyrics_clean" not in fieldnames:
        fieldnames.append("lyrics_clean")

    changed = 0
    blanked_short = 0
    for r in rows:
        raw = r.get("lyrics") or ""
        cleaned = clean_lyrics(raw)
        if args.min_len > 0 and 0 < len(cleaned) < args.min_len:
            cleaned = ""
            blanked_short += 1
        if (r.get("lyrics_clean") or "") != cleaned:
            r["lyrics_clean"] = cleaned
            changed += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Input rows: {len(rows)}")
    print(f"lyrics_clean updated: {changed}")
    print(f"lyrics_clean blanked (too short): {blanked_short}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
