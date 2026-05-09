# Submission Checklist

Use this checklist before pushing the MoleculeSTM fork.

## Add These Files

```bash
git add .gitignore
git add chemvl_benchmark/README.md
git add chemvl_benchmark/DECISIONS.md
git add chemvl_benchmark/SUBMIT_FILES.md
git add chemvl_benchmark/scripts
git add chemvl_benchmark/configs
git add chemvl_benchmark/splits
git add chemvl_benchmark/results/moleculestm_graph
git add chemvl_benchmark/results/moleculestm_smiles
```

## Do Not Add These Files

```text
data/
outputs/
checkpoints/
apex/
MolBART/
llama3-8B/
__pycache__/
*.pyc
*.log
*.pth
*.pt
*.ckpt
best_predictions.npz
train_val_test_history.csv
```

## Sanity Checks

```bash
python -m py_compile chemvl_benchmark/scripts/run_on_chemvl_split.py
python -m py_compile chemvl_benchmark/scripts/run_batch.py
python -m py_compile chemvl_benchmark/scripts/summarize.py

python chemvl_benchmark/scripts/run_batch.py \
  --tables A \
  --tasks bbbp \
  --runseeds 1 \
  --molecule_type Graph \
  --epochs 1 \
  --dry_run
```

If `CHEMVL_DATA_ROOT` is available, run the same command without `--dry_run` as a smoke test.

## Expected Small Artifact Counts

```text
configs: 50 json files
splits: 50 npz files + 50 metadata json files
results: compact csv/md summaries only
```
