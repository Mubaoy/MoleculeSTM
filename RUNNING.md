# Running The ChemVL Benchmark

This document describes how to run the MoleculeSTM-ChemVL benchmark in this fork.

## 1. Environment

Use the original MoleculeSTM environment as the base. The completed runs used a Python 3.7 environment with RDKit and PyTorch Geometric available.

Core requirements:

```bash
conda create -n molstm python=3.7
conda activate molstm

conda install -y -c rdkit rdkit=2020.09.1.0
conda install -y -c conda-forge -c pytorch pytorch=1.9.1
conda install -y -c pyg -c conda-forge pyg==2.0.3

pip install pandas scikit-learn tqdm transformers
```

For SMILES/MegaMolBART runs, also install the dependencies required by the original MoleculeSTM SMILES branch, including MegaMolBART/Megatron and Apex.

If you already have a working MoleculeSTM environment, use that environment directly.

## 2. Required Local Files

This repository does not commit datasets or pretrained checkpoints.

Set the path to ChemVL processed data:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
```

Expected data layout:

```text
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/bace/processed/bace_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/bbbp/processed/bbbp_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/regression/esol/processed/esol_processed_ac.csv
```

The MoleculeSTM checkpoints are expected at the original MoleculeSTM paths under `data/pretrained_MoleculeSTM/`. You can override them:

```bash
export MOLECULESTM_GRAPH_CKPT=/path/to/graph/molecule_model.pth
export MOLECULESTM_SMILES_CKPT=/path/to/smiles/molecule_model.pth
```

## 3. Smoke Test

Run one small Graph job:

```bash
python -u chemvl_benchmark/scripts/run_batch.py   --tables A   --tasks bbbp   --runseeds 1   --molecule_type Graph   --epochs 1   --output_root outputs/chemvl_benchmark_smoke
```

Run one small SMILES job:

```bash
python -u chemvl_benchmark/scripts/run_batch.py   --tables A   --tasks bbbp   --runseeds 1   --molecule_type SMILES   --epochs 1   --batch_size 8   --output_root outputs/chemvl_benchmark_smiles_smoke
```

Use `--dry_run` first if you only want to check command generation.

## 4. Full Table A/B Runs

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

Both scripts run:

```text
tables: A B
runseeds: 1 2 3
```

To restrict datasets:

```bash
bash chemvl_benchmark/scripts/run_graph_ab.sh --tasks bace bbbp
```

To customize output:

```bash
OUTPUT_ROOT=outputs/my_smiles_run bash chemvl_benchmark/scripts/run_smiles_ab.sh
```

## 5. Summarize

Graph result summary:

```bash
python chemvl_benchmark/scripts/summarize.py   --moleculestm_output outputs/chemvl_benchmark_graph_runs   --output_dir outputs/chemvl_benchmark_graph_tables
```

SMILES result summary:

```bash
python chemvl_benchmark/scripts/summarize.py   --moleculestm_output outputs/chemvl_benchmark_smiles_runs   --output_dir outputs/chemvl_benchmark_smiles_tables
```

The script writes:

```text
raw_results.csv
summary_long.csv
table_A_moleculenet_scaffold.csv
table_B_moleculenet_random_scaffold.csv
```

## 6. Existing Committed Results

Committed result summaries are under:

```text
chemvl_benchmark/results/moleculestm_graph/
chemvl_benchmark/results/moleculestm_smiles/
```

These are compact CSV/Markdown summaries only. Full training outputs are intentionally excluded.

## 7. Notes

- Do not commit `data/`, `outputs/`, checkpoints, logs, or cache files.
- Use the fixed splits in `chemvl_benchmark/splits/` for fair comparison.
- Details about design choices are documented in `chemvl_benchmark/DECISIONS.md`.
