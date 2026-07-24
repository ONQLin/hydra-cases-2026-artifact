#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/myenv/bin/python"
ESTIMATOR="${1:-markov}"
OUTPUT_DIR="${2:-$ROOT_DIR/Fast_Estimate/predicted_configs}"

cd "$ROOT_DIR"
"$PYTHON_BIN" "$ROOT_DIR/Fast_Estimate/fast_estimate_models.py"   --estimator "$ESTIMATOR"   --output-dir "$OUTPUT_DIR"
