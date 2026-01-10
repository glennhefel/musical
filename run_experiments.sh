#!/usr/bin/env sh
set -eu

# Best-effort pipefail (not POSIX; supported by bash/zsh/ksh).
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail || true

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

# Prefer the repo-local venv interpreter if present (no activation needed).
if [ -x "$ROOT_DIR/venv/bin/python" ]; then
  PY="$ROOT_DIR/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "ERROR: Python not found. Create venv and install requirements first." >&2
  echo "  python3 -m venv venv" >&2
  echo "  . venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
Usage:
  sh ./run_experiments.sh [TASK] [cpu|gpu]
  ./run_experiments.sh [TASK] [cpu|gpu]

Tasks:
  easy    Run Easy task suite
  medium  Run Medium task suite
  hard    Run Hard task suite
  all     Run all task suites (default)

Device:
  gpu     Use CUDA (`--device cuda`) (default)
  cpu     Use CPU (`--device cpu`)

Examples:
  sh ./run_experiments.sh easy
  sh ./run_experiments.sh easy gpu
  sh ./run_experiments.sh easy cpu
  sh ./run_experiments.sh medium
  sh ./run_experiments.sh hard
  sh ./run_experiments.sh all
EOF
}

task=${1:-all}
device_mode=${2:-gpu}
if [ "$task" = "--help" ] || [ "$task" = "-h" ] || [ "$task" = "help" ]; then
  usage
  exit 0
fi

# ------------------------
# Hardcoded experiment defaults
# ------------------------
case "$device_mode" in
  gpu)
    DEVICE="cuda"
    ;;
  cpu)
    DEVICE="cpu"
    ;;
  "")
    DEVICE="cuda"
    ;;
  *)
    echo "ERROR: Unknown device: $device_mode (expected cpu|gpu)" >&2
    usage
    exit 2
    ;;
esac

VIZ="umap"  # tsne | umap | none

EPOCHS_EASY=10
EPOCHS_MEDIUM=10
EPOCHS_HARD=10

CLUSTERS_EASY=2
CLUSTERS=6

case "$task" in
  easy)
    exec "$PY" run_all_tasks.py \
      --tasks easy \
      --device "$DEVICE" \
      --viz "$VIZ" \
      --epochs-easy "$EPOCHS_EASY" \
      --clusters-easy "$CLUSTERS_EASY" \
      --clusters "$CLUSTERS"
    ;;
  medium)
    exec "$PY" run_all_tasks.py \
      --tasks medium \
      --device "$DEVICE" \
      --viz "$VIZ" \
      --epochs-medium "$EPOCHS_MEDIUM" \
      --clusters "$CLUSTERS"
    ;;
  hard)
    exec "$PY" run_all_tasks.py \
      --tasks hard \
      --device "$DEVICE" \
      --viz "$VIZ" \
      --epochs-hard "$EPOCHS_HARD" \
      --clusters "$CLUSTERS"
    ;;
  all)
    exec "$PY" run_all_tasks.py \
      --device "$DEVICE" \
      --viz "$VIZ" \
      --epochs-easy "$EPOCHS_EASY" \
      --epochs-medium "$EPOCHS_MEDIUM" \
      --epochs-hard "$EPOCHS_HARD" \
      --clusters-easy "$CLUSTERS_EASY" \
      --clusters "$CLUSTERS"
    ;;
  *)
    echo "ERROR: Unknown task: $task" >&2
    usage
    exit 2
    ;;
esac
