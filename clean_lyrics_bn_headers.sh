#!/usr/bin/env sh
set -eu

# Best-effort pipefail (not POSIX; supported by bash/zsh/ksh).
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TARGET_DIR="$ROOT_DIR/data/lyrics_bn"

if [ ! -d "$TARGET_DIR" ]; then
  echo "ERROR: Folder not found: $TARGET_DIR" >&2
  exit 1
fi

# Removes random/unpaired-lyrics header lines from each .txt file.
# This keeps the real lyrics body intact.
#
# Lines removed (if present):
# - WARNING: Lyrics below are randomly assigned and DO NOT correspond to this audio.
# - RANDOM_LYRICS_TITLE: ...
# - RANDOM_LYRICS_CATEGORY: ...
# - SOURCE: data/BanglaSongLyrics.csv (random sample; unpaired)

tmp="${TMPDIR:-/tmp}/lyrics_bn_clean.$$"
count=0

for f in "$TARGET_DIR"/*.txt; do
  [ -e "$f" ] || continue

  # Filter out the header lines.
  awk '
    /^WARNING: Lyrics below are randomly assigned and DO NOT correspond to this audio\./ { next }
    /^RANDOM_LYRICS_TITLE:[[:space:]]*/ { next }
    /^RANDOM_LYRICS_CATEGORY:[[:space:]]*/ { next }
    /^SOURCE:[[:space:]]*data\/BanglaSongLyrics\.csv \(random sample; unpaired\)[[:space:]]*$/ { next }
    { print }
  ' "$f" > "$tmp"

  # Only overwrite if content changed.
  if ! cmp -s "$f" "$tmp"; then
    mv "$tmp" "$f"
    count=$((count + 1))
  else
    rm -f "$tmp"
  fi

done

echo "Done. Cleaned $count file(s) in $TARGET_DIR"
