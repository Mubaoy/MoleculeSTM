# ChemVL Benchmark Decisions

This document records the implementation decisions made while adding ChemVL-style evaluation to a MoleculeSTM fork.

## No ChemVL Submodule

ChemVL is not added as a git submodule. The benchmark only needs fixed processed-data paths, ChemVL-style configs, and exact train/validation/test indices. Keeping those inside the MoleculeSTM fork avoids maintaining a nested fork and avoids coupling MoleculeSTM reproduction to ChemVL development dependencies.

The required external input is a local `chemvl-data` directory with the processed CSV layout used by ChemVL.

## Fixed Splits Are Committed

The split files in `splits/` are committed because split choice materially affects performance. They are small and make the comparison reproducible.

Each split has:

- `.npz`: `train_idx`, `val_idx`, `test_idx`.
- `.json`: metadata with task name, split type, and subset sizes.

## Configs Use Path Placeholders

Configs contain `${CHEMVL_DATA_ROOT}` and `${MOLECULESTM_ROOT}` placeholders instead of machine-specific absolute paths. `scripts/run_on_chemvl_split.py` expands these at runtime.

This keeps the configs traceable while avoiding machine-specific absolute paths in the fork.

## MoleculeSTM Branches

Two MoleculeSTM molecule-input branches are supported:

- `Graph`: MoleculeSTM graph encoder.
- `SMILES`: MoleculeSTM MegaMolBART/SMILES encoder.

Both branches share the same ChemVL configs and split indices.

## SMILES RNG Fix

MegaMolBART/Megatron checkpoint loading can restore checkpoint RNG state. In the original wrapper, this made `runseed=1`, `runseed=2`, and `runseed=3` produce identical training trajectories for SMILES runs.

The fix is implemented in `scripts/run_on_chemvl_split.py`:

- re-apply `set_seed(runseed)` after loading the molecule encoder checkpoint and before creating the prediction head;
- pass an explicit `torch.Generator` seeded by `runseed` into the training DataLoader.

The committed SMILES summaries were generated after this fix.

## Metrics

The metric choices follow the ChemVL benchmark setup:

- MoleculeNet classification: ROC-AUC.
- MoleculeNet regression: RMSE, except QM7 uses MAE.
- MoleculeACE MolMCL protocol: R2 in the MolMCL baseline summary.

## Result Scope

Completed MoleculeSTM runs:

- Table A MoleculeNet scaffold, Graph branch, runseed 1-3.
- Table B MoleculeNet random_scaffold, Graph branch, runseed 1-3.
- Table A MoleculeNet scaffold, SMILES branch, runseed 1-3.
- Table B MoleculeNet random_scaffold, SMILES branch, runseed 1-3.

Table C currently includes existing MolMCL baseline summaries only. MoleculeSTM Table C was not run to completion as part of this package.

## Files Intentionally Excluded

The fork should not include:

- processed ChemVL datasets;
- pretrained MoleculeSTM or MegaMolBART checkpoints;
- full per-run training outputs;
- prediction dumps such as `best_predictions.npz`;
- logs;
- Python caches.

Only compact CSV/Markdown summaries and fixed split indices are committed.
