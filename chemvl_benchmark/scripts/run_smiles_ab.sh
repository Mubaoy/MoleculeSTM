#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BENCHMARK_ROOT}/.." && pwd)"

: "${CHEMVL_DATA_ROOT:?Set CHEMVL_DATA_ROOT to the ChemVL chemvl-data directory.}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/chemvl_benchmark_smiles_runs}"

cd "${REPO_ROOT}"
python -u "${BENCHMARK_ROOT}/scripts/run_batch.py" \
  --tables A B \
  --runseeds 1 2 3 \
  --chemvl_data_root "${CHEMVL_DATA_ROOT}" \
  --molecule_type SMILES \
  --batch_size 8 \
  --output_root "${OUTPUT_ROOT}" \
  --continue_on_error \
  --retries 1 \
  "$@"
