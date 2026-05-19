# MoleculeSTM-ChemVL Benchmark

This repository is a fork of [chao1224/MoleculeSTM](https://github.com/chao1224/MoleculeSTM) with an added ChemVL-style benchmark wrapper for fair molecular property prediction evaluation.

The original MoleculeSTM project studies multi-modal molecule structure-text representation learning. This fork keeps the original MoleculeSTM code and adds a reproducible evaluation package under `chemvl_benchmark/`, so that MoleculeSTM can be evaluated on the same ChemVL data splits and metrics used by the ChemVL/MolMCL comparison workflow.

## Project Goal

The goal is to compare MoleculeSTM and related molecular representation models under a consistent evaluation protocol:

- same MoleculeNet datasets;
- same ChemVL scaffold and random-scaffold split indices;
- same runseed protocol (`runseed 1, 2, 3`);
- same metric definitions;
- compact, traceable result summaries committed to the fork.

This avoids comparing results produced with different dataset splits or hidden preprocessing choices.

## What This Fork Adds

```text
chemvl_benchmark/
  README.md                 Detailed benchmark documentation
  DECISIONS.md              Design decisions and deviations from upstream
  SUBMIT_FILES.md           Submission/checklist notes
  configs/                  ChemVL-style benchmark configs
  splits/                   Fixed train/validation/test split indices
  scripts/                  Runners and summarization scripts
  results/                  Compact CSV/Markdown result summaries
```

The benchmark module supports two MoleculeSTM molecule-input branches:

- `Graph`: MoleculeSTM graph encoder.
- `SMILES`: MoleculeSTM MegaMolBART/SMILES encoder.

The committed split files are intentionally included because the split choice has a major effect on performance and reproducibility.

## Benchmarks

### Table A: MoleculeNet scaffold

Ten MoleculeNet datasets using ChemVL scaffold split:

```text
bbbp, bace, clintox, tox21, sider, hiv, esol, freesolv, lipo, qm7
```

Classification tasks use ROC-AUC. Regression tasks use RMSE, except QM7 uses MAE.

### Table B: MoleculeNet random_scaffold

The same ten MoleculeNet datasets using ChemVL random-scaffold split.

### Table C: MoleculeACE MolMCL protocol

The fixed configs and splits for 30 MoleculeACE datasets are included. The current committed Table C summary contains MolMCL baselines only; MoleculeSTM Table C runs were not completed in this fork.

## Current Results

Compact result files are stored under:

```text
chemvl_benchmark/results/moleculestm_graph/
chemvl_benchmark/results/moleculestm_smiles/
```

Important result tables:

```text
chemvl_benchmark/results/moleculestm_graph/table_A_moleculenet_scaffold.csv
chemvl_benchmark/results/moleculestm_graph/table_B_moleculenet_random_scaffold.csv
chemvl_benchmark/results/moleculestm_smiles/table_A_moleculenet_scaffold.csv
chemvl_benchmark/results/moleculestm_smiles/table_B_moleculenet_random_scaffold.csv
```

The SMILES results use the RNG fix documented in `chemvl_benchmark/DECISIONS.md`.

## Quick Start

Use [RUNNING.md](RUNNING.md) for environment setup and run commands.

Minimal smoke test:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data

python -u chemvl_benchmark/scripts/run_batch.py   --tables A   --tasks bbbp   --runseeds 1   --molecule_type Graph   --epochs 1   --output_root outputs/chemvl_benchmark_smoke
```

Full A/B runs:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
bash chemvl_benchmark/scripts/run_graph_ab.sh
bash chemvl_benchmark/scripts/run_smiles_ab.sh
```

## Reproducibility Notes

- The fork does not vendor ChemVL as a git submodule.
- ChemVL processed data is expected to exist locally and is referenced via `CHEMVL_DATA_ROOT`.
- Fixed split indices are committed under `chemvl_benchmark/splits/`.
- Large local artifacts are not committed: datasets, checkpoints, full run directories, logs, and prediction dumps.
- `.gitignore` is configured to prevent accidental submission of local data and model weights.

## Relationship to Upstream MoleculeSTM

This fork is based on the official MoleculeSTM repository:

```text
https://github.com/chao1224/MoleculeSTM
```

Original paper:

```text
MoleculeSTM: Multi-modal Molecule Structure-text Model for Text-based Editing and Retrieval
Nature Machine Intelligence, 2023
```

This fork only adds the ChemVL benchmark wrapper and result package. The original MoleculeSTM model code remains the base implementation.
