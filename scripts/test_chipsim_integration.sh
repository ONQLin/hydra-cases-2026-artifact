#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -x ".chipsim-venv/bin/python" ]] || \
   [[ ! -x "third_party/CHIPSIM/integrations/gem5/build/Garnet_standalone/gem5.opt" ]]; then
    echo "CHIPSIM is not built. Run: bash scripts/setup_chipsim.sh"
    exit 1
fi

echo "Running HYDRA-to-CHIPSIM adapter tests..."
python -m unittest tests.test_chipsim_backend tests.test_execution_backend \
    tests.test_execution_cleanup \
    tests.test_chipsim_contention.CoSimulationTest tests.test_serving_recorder \
    tests.test_chipsim_checkpoints tests.test_chipsim_validation

echo "Running real Garnet contention and HBM DMA pacing checks..."
RUN_CHIPSIM_GARNET_TESTS=1 .chipsim-venv/bin/python -m unittest \
    tests.test_chipsim_runtime_service tests.test_chipsim_contention.PacketContentionTest \
    tests.test_chipsim_dma_pacing

echo "Running one HYDRA transformer block through main.py -> CHIPSIM -> Garnet..."
python main.py \
    --simulator-backend chipsim \
    --cluster-config.batch-size 1 \
    --metrics-config.output-dir output_sanity_checks/chipsim \
    --metrics-config.label-name transformer-block-demo

echo "CHIPSIM integration smoke test passed."
