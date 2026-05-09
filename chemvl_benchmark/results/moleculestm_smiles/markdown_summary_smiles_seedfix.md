# MoleculeSTM-SMILES Seedfix Results

Completed runs: 60/60 result.json

Failed runs: 0

Runs: runseed 1-3

Molecule input branch: SMILES

Result root: `outputs/chemvl_benchmark_smiles_seedfix_runs`

Summary root: `outputs/chemvl_benchmark_smiles_seedfix_tables`

## Table A: MoleculeNet scaffold

| dataset | metric | mean +/- std |
|---|---|---|
| bbbp | ROC-AUC | 0.7001 +/- 0.0312 |
| bace | ROC-AUC | 0.7685 +/- 0.0250 |
| clintox | ROC-AUC | 0.9933 +/- 0.0052 |
| tox21 | ROC-AUC | 0.7427 +/- 0.0098 |
| sider | ROC-AUC | 0.5994 +/- 0.0098 |
| hiv | ROC-AUC | 0.6982 +/- 0.0059 |
| esol | RMSE | 0.9313 +/- 0.0169 |
| freesolv | RMSE | 2.6557 +/- 0.0924 |
| lipo | RMSE | 0.8208 +/- 0.0201 |
| qm7 | MAE | 73.8773 +/- 3.1784 |

## Table B: MoleculeNet random_scaffold

| dataset | metric | mean +/- std |
|---|---|---|
| bbbp | ROC-AUC | 0.9320 +/- 0.0024 |
| bace | ROC-AUC | 0.7962 +/- 0.0265 |
| clintox | ROC-AUC | 0.9929 +/- 0.0023 |
| tox21 | ROC-AUC | 0.7937 +/- 0.0104 |
| sider | ROC-AUC | 0.5948 +/- 0.0115 |
| hiv | ROC-AUC | 0.7094 +/- 0.0145 |
| esol | RMSE | 0.9226 +/- 0.0171 |
| freesolv | RMSE | 2.1922 +/- 0.1799 |
| lipo | RMSE | 0.8205 +/- 0.0326 |
| qm7 | MAE | 68.2455 +/- 1.8080 |
