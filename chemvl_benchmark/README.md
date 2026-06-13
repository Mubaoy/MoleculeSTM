# ChemVL-Split MoleculeSTM Benchmark

This folder contains a self-contained MoleculeSTM-side benchmark wrapper for running MoleculeSTM on fixed ChemVL train/validation/test splits.

The goal is reproducible comparison under the ChemVL evaluation protocol without vendoring ChemVL as a git submodule. ChemVL is treated as an external data provider: this repository stores the benchmark configs, fixed split indices, MoleculeSTM runners, and compact result summaries.

## What Is Included

- `configs/`: ChemVL-style benchmark configs for Table A/B/C.
- `splits/`: fixed train/validation/test indices exported from ChemVL.
- `scripts/run_on_chemvl_split.py`: single-task MoleculeSTM runner using a ChemVL config and a fixed split file.
- `scripts/run_batch.py`: batch runner for Table A/B/C and runseed 1-3.
- `scripts/summarize.py`: result aggregator producing `mean +/- std` tables.
- `results/moleculestm_graph/`: compact Graph-branch summaries and optional MolMCL baselines.
- `results/moleculestm_smiles/`: compact SMILES-branch summaries after the seed fix.

Large local files are intentionally not included: processed datasets, pretrained checkpoints, full per-run outputs, logs, and prediction dumps.

## Benchmarks

Table A is MoleculeNet with ChemVL `scaffold` split:

- Classification: `bbbp`, `bace`, `clintox`, `tox21`, `sider`, `hiv`, metric ROC-AUC.
- Regression: `esol`, `freesolv`, `lipo`, metric RMSE.
- Regression: `qm7`, metric MAE.

Table B is the same 10 MoleculeNet datasets with ChemVL `random_scaffold` split.

Table C is MoleculeACE with the MolMCL protocol. The committed Table C result summary currently contains MolMCL baselines only; MoleculeSTM Table C runs were not part of the completed run set.

## Environment

Use the upstream MoleculeSTM environment as the base. The completed runs used a Python 3.7 environment with RDKit available. RDKit is the main version-sensitive dependency because scaffold generation and molecule parsing can change across versions.

Minimum additional Python packages used by these scripts:

```bash
pip install pandas scikit-learn
```

For Graph runs, install the same PyTorch Geometric stack required by upstream MoleculeSTM. For SMILES runs, make sure MegaMolBART assets and MoleculeSTM SMILES pretrained weights are available under the upstream expected paths or pass explicit paths via environment variables.

Optional environment variables:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
export MOLECULESTM_GRAPH_CKPT=/path/to/graph/molecule_model.pth
export MOLECULESTM_SMILES_CKPT=/path/to/smiles/molecule_model.pth
```

`CHEMVL_DATA_ROOT` must contain the processed CSV files under the same layout used by ChemVL, for example:

```text
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/bace/processed/bace_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/regression/esol/processed/esol_processed_ac.csv
```

## Quick Smoke Test

From the MoleculeSTM repository root:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data

python -u chemvl_benchmark/scripts/run_batch.py \
  --tables A \
  --tasks bbbp \
  --runseeds 1 \
  --molecule_type Graph \
  --epochs 1 \
  --output_root outputs/chemvl_benchmark_smoke
```

For the SMILES branch:

```bash
python -u chemvl_benchmark/scripts/run_batch.py \
  --tables A \
  --tasks bbbp \
  --runseeds 1 \
  --molecule_type SMILES \
  --epochs 1 \
  --batch_size 8 \
  --output_root outputs/chemvl_benchmark_smiles_smoke
```

## Full A/B Runs

Graph branch:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
bash chemvl_benchmark/scripts/run_graph_ab.sh
```

SMILES branch:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
bash chemvl_benchmark/scripts/run_smiles_ab.sh
```

Both scripts run Table A and Table B with `runseed 1 2 3`. Override defaults by appending extra flags:

```bash
OUTPUT_ROOT=outputs/my_smiles_run bash chemvl_benchmark/scripts/run_smiles_ab.sh --tasks bace bbbp
```

## Summarize Results

```bash
python chemvl_benchmark/scripts/summarize.py \
  --moleculestm_output outputs/chemvl_benchmark_graph_runs \
  --output_dir chemvl_benchmark/results/generated_graph
```

To include external MolMCL baseline result JSON files, pass:

```bash
python chemvl_benchmark/scripts/summarize.py \
  --moleculestm_output outputs/chemvl_benchmark_graph_runs \
  --chemvl_results /path/to/chemvl-data/results \
  --include_molmcl \
  --output_dir chemvl_benchmark/results/generated_graph_with_molmcl
```

## Committed Result Summaries

Graph branch summaries:

- `results/moleculestm_graph/table_A_moleculenet_scaffold.csv`
- `results/moleculestm_graph/table_B_moleculenet_random_scaffold.csv`

SMILES branch summaries:

- `results/moleculestm_smiles/table_A_moleculenet_scaffold.csv`
- `results/moleculestm_smiles/table_B_moleculenet_random_scaffold.csv`

The SMILES results use the seed fix documented in `DECISIONS.md`.

## Parameter Audit

From the repository root:

```bash
conda activate molstm
bash parameter_audit.sh
```

The script writes `parameter_summary.csv` in the repository root by default. It audits only the MoleculeSTM branches in this fork (`Graph` and `SMILES`). GEM and MoLFormer have their own independent `parameter_audit.sh` scripts in their respective repositories.

The MoleculeSTM count is based on `model.parameters()` plus the ChemVL prediction head, after setting `requires_grad` according to the finetune strategy:

- `fine_tuning`: encoder and prediction head are trainable.
- `linear_probing`: encoder is frozen and only the prediction head is trainable.

The default config is Table A `bbbp`, a representative single-task setup. For multi-task heads such as `tox21` or `sider`, pass a different config:

```bash
bash parameter_audit.sh \
  --config chemvl_benchmark/configs/table_A_moleculenet_scaffold/tox21.json \
  --output outputs/parameter_summary_tox21.csv
```
