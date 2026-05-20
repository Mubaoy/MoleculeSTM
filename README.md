# MoleculeSTM-ChemVL Benchmark

This repository is a fork of [chao1224/MoleculeSTM](https://github.com/chao1224/MoleculeSTM). It keeps the original MoleculeSTM implementation and adds a ChemVL-style benchmark package under `chemvl_benchmark/` for fair molecular property prediction evaluation.

The purpose of this fork is not to reimplement MoleculeSTM. Instead, it connects MoleculeSTM to the ChemVL evaluation protocol so that MoleculeSTM can be compared with ChemVL/MolMCL-style baselines under the same datasets, splits, run seeds, and metrics.

## 1. What This Project Adds

The added benchmark package is organized as follows:

```text
chemvl_benchmark/
  README.md                 Detailed ChemVL benchmark documentation
  DECISIONS.md              Design decisions and implementation notes
  SUBMIT_FILES.md           Submission checklist
  configs/                  ChemVL-style benchmark configs
  splits/                   Fixed train/validation/test split indices
  scripts/                  Run and summarization scripts
  results/                  Compact CSV/Markdown result summaries
```

Supported MoleculeSTM molecule-input branches:

- `Graph`: MoleculeSTM graph encoder.
- `SMILES`: MoleculeSTM MegaMolBART/SMILES encoder.

Main goals:

- use the same ChemVL MoleculeNet scaffold and random-scaffold splits;
- use the same `runseed = 1, 2, 3` protocol;
- use consistent classification and regression metrics;
- commit small, traceable split/config/result files;
- avoid committing datasets, checkpoints, full training outputs, and logs.

## 2. Benchmark Scope

### Table A: MoleculeNet scaffold

Datasets:

```text
bbbp, bace, clintox, tox21, sider, hiv, esol, freesolv, lipo, qm7
```

Metrics:

- Classification tasks: ROC-AUC.
- Regression tasks: RMSE.
- QM7: MAE.

### Table B: MoleculeNet random_scaffold

The same ten MoleculeNet datasets are evaluated with the ChemVL random-scaffold split.

### Table C: MoleculeACE MolMCL protocol

Configs and fixed splits for 30 MoleculeACE datasets are included. The committed Table C result summary currently contains MolMCL baselines only. MoleculeSTM Table C runs were not completed in this fork.

## 3. Environment Setup

If you already have a working MoleculeSTM conda environment, you can reuse it. Otherwise, create one with:

```bash
conda create -n molstm python=3.7
conda activate molstm

conda install -y -c rdkit rdkit=2020.09.1.0
conda install -y -c conda-forge -c pytorch pytorch=1.9.1
conda install -y -c pyg -c conda-forge pyg==2.0.3

pip install pandas scikit-learn tqdm transformers requests matplotlib spacy Levenshtein ogb==1.2.0
```

For the SMILES branch, install the original MoleculeSTM SMILES dependencies as well, including MegaMolBART/Megatron and Apex. Reusing an already working MoleculeSTM environment is recommended.

## 4. Data And Checkpoint Preparation

This repository does not include datasets or pretrained checkpoints.

Set the ChemVL processed-data directory:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
```

Expected layout examples:

```text
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/bace/processed/bace_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/bbbp/processed/bbbp_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/regression/esol/processed/esol_processed_ac.csv
```

MoleculeSTM checkpoints are expected under the original MoleculeSTM-style paths by default:

```text
data/pretrained_MoleculeSTM/
data/pretrained_MegaMolBART/checkpoints/
MoleculeSTM/bart_vocab.txt
```

You can override checkpoint paths with:

```bash
export MOLECULESTM_GRAPH_CKPT=/path/to/graph/molecule_model.pth
export MOLECULESTM_SMILES_CKPT=/path/to/smiles/molecule_model.pth
```

## 5. Quick Smoke Tests

Graph branch smoke test:

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

SMILES branch smoke test:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data

python -u chemvl_benchmark/scripts/run_batch.py \
  --tables A \
  --tasks bbbp \
  --runseeds 1 \
  --molecule_type SMILES \
  --epochs 1 \
  --batch_size 8 \
  --output_root outputs/chemvl_benchmark_smiles_smoke
```

To only check command generation without training:

```bash
python chemvl_benchmark/scripts/run_batch.py \
  --tables A \
  --tasks bbbp \
  --runseeds 1 \
  --molecule_type Graph \
  --dry_run
```

## 6. Full Table A/B Runs

Run the Graph branch on Table A/B:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
bash chemvl_benchmark/scripts/run_graph_ab.sh
```

Run the SMILES branch on Table A/B:

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
bash chemvl_benchmark/scripts/run_smiles_ab.sh
```

Both scripts run:

```text
Table A + Table B
runseed 1, 2, 3
```

Run only selected datasets:

```bash
bash chemvl_benchmark/scripts/run_graph_ab.sh --tasks bace bbbp
bash chemvl_benchmark/scripts/run_smiles_ab.sh --tasks bace bbbp
```

Use a custom output directory:

```bash
OUTPUT_ROOT=outputs/my_graph_run bash chemvl_benchmark/scripts/run_graph_ab.sh
OUTPUT_ROOT=outputs/my_smiles_run bash chemvl_benchmark/scripts/run_smiles_ab.sh
```

## 7. Summarize Results

Summarize Graph results:

```bash
python chemvl_benchmark/scripts/summarize.py \
  --moleculestm_output outputs/chemvl_benchmark_graph_runs \
  --output_dir outputs/chemvl_benchmark_graph_tables
```

Summarize SMILES results:

```bash
python chemvl_benchmark/scripts/summarize.py \
  --moleculestm_output outputs/chemvl_benchmark_smiles_runs \
  --output_dir outputs/chemvl_benchmark_smiles_tables
```

The summarization script writes:

```text
raw_results.csv
summary_long.csv
table_A_moleculenet_scaffold.csv
table_B_moleculenet_random_scaffold.csv
```

## 8. Committed Results

Completed compact summaries are stored in:

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

## 9. Notes

Do not commit the following local artifacts:

```text
data/
outputs/
checkpoints/
*.pth
*.pt
*.ckpt
*.log
__pycache__/
```

These are excluded by `.gitignore`. Fixed split files under `chemvl_benchmark/splits/**/*.npz` are intentionally committed because dataset splits directly affect model performance and fairness.

## 10. Relationship To Upstream MoleculeSTM

This repository is based on the official MoleculeSTM project:

```text
https://github.com/chao1224/MoleculeSTM
```

Original paper:

```text
MoleculeSTM: Multi-modal Molecule Structure-text Model for Text-based Editing and Retrieval
Nature Machine Intelligence, 2023
```

This fork only adds the ChemVL benchmark wrapper, fixed split/config files, and compact experiment summaries. The original MoleculeSTM model implementation remains the base code.
