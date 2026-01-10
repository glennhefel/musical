#!/usr/bin/env bash
# Usage (must be sourced):
#   source scripts/activate_env.sh            # create .venv if needed + activate
#   INSTALL_DEPS=1 source scripts/activate_env.sh  # also pip install -r requirements.txt

set -euo pipefail

# This script must be sourced to modify the current shell environment.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Error: this script must be sourced, not executed." >&2
  echo "Run: source scripts/activate_env.sh" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
REQ_FILE="${REPO_ROOT}/requirements.txt"

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found on PATH." >&2
  return 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating virtualenv at ${VENV_DIR}"
  python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
  if [[ -f "${REQ_FILE}" ]]; then
    echo "Installing dependencies from ${REQ_FILE}"
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r "${REQ_FILE}"
  else
    echo "Warning: ${REQ_FILE} not found; skipping dependency install." >&2
  fi
fi

echo "Activated: $(python -c 'import sys; print(sys.executable)')"