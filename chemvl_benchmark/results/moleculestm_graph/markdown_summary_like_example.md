## Table A: scaffold (MoleculeSTM-Graph)

| dataset | metric | mean ± std |
|---|---:|---:|
| bbbp | ROC-AUC | 0.6867 ± 0.0210 |
| bace | ROC-AUC | 0.8003 ± 0.0234 |
| clintox | ROC-AUC | 0.9104 ± 0.0265 |
| tox21 | ROC-AUC | 0.7658 ± 0.0051 |
| sider | ROC-AUC | 0.5866 ± 0.0265 |
| hiv | ROC-AUC | 0.7740 ± 0.0172 |
| esol | RMSE | 1.1594 ± 0.0577 |
| freesolv | RMSE | 2.5813 ± 0.2445 |
| lipo | RMSE | 0.7256 ± 0.0153 |
| qm7 | MAE | 139.8482 ± 3.3376 |

## Table B: random_scaffold (MoleculeSTM-Graph)

| dataset | metric | mean ± std |
|---|---:|---:|
| bbbp | ROC-AUC | 0.8386 ± 0.0228 |
| bace | ROC-AUC | 0.7934 ± 0.0283 |
| clintox | ROC-AUC | 0.7912 ± 0.0526 |
| tox21 | ROC-AUC | 0.7960 ± 0.0079 |
| sider | ROC-AUC | 0.6111 ± 0.0234 |
| hiv | ROC-AUC | 0.8115 ± 0.0017 |
| esol | RMSE | 1.2707 ± 0.1196 |
| freesolv | RMSE | 1.7069 ± 0.2555 |
| lipo | RMSE | 0.6976 ± 0.0010 |
| qm7 | MAE | 115.5080 ± 6.6214 |

## Table C: MoleculeACE MolMCL protocol (MolMCL-GIN, existing baseline)

| dataset | metric | mean ± std |
|---|---:|---:|
| CHEMBL1862_Ki | R2 | 0.7702 ± 0.0070 |
| CHEMBL1871_Ki | R2 | 0.6072 ± 0.0013 |
| CHEMBL2034_Ki | R2 | 0.4671 ± 0.0249 |
| CHEMBL2047_EC50 | R2 | 0.2544 ± 0.1281 |
| CHEMBL204_Ki | R2 | 0.7810 ± 0.0130 |
| CHEMBL2147_Ki | R2 | 0.8436 ± 0.0157 |
| CHEMBL214_Ki | R2 | 0.6362 ± 0.0096 |
| CHEMBL218_EC50 | R2 | 0.5493 ± 0.0180 |
| CHEMBL219_Ki | R2 | 0.2760 ± 0.0209 |
| CHEMBL228_Ki | R2 | 0.6780 ± 0.0165 |
| CHEMBL231_Ki | R2 | 0.6807 ± 0.0027 |
| CHEMBL233_Ki | R2 | 0.6914 ± 0.0104 |
| CHEMBL234_Ki | R2 | 0.6139 ± 0.0091 |
| CHEMBL235_EC50 | R2 | 0.6659 ± 0.0008 |
| CHEMBL236_Ki | R2 | 0.6973 ± 0.0180 |
| CHEMBL237_EC50 | R2 | 0.6376 ± 0.0415 |
| CHEMBL237_Ki | R2 | 0.7033 ± 0.0119 |
| CHEMBL238_Ki | R2 | 0.6467 ± 0.0267 |
| CHEMBL239_EC50 | R2 | 0.5084 ± 0.0133 |
| CHEMBL244_Ki | R2 | 0.7980 ± 0.0013 |
| CHEMBL262_Ki | R2 | 0.5948 ± 0.0038 |
| CHEMBL264_Ki | R2 | 0.6815 ± 0.0049 |
| CHEMBL2835_Ki | R2 | 0.7760 ± 0.0200 |
| CHEMBL287_Ki | R2 | 0.4971 ± 0.0283 |
| CHEMBL2971_Ki | R2 | 0.8555 ± 0.0124 |
| CHEMBL3979_EC50 | R2 | 0.4990 ± 0.0474 |
| CHEMBL4005_Ki | R2 | 0.5419 ± 0.0166 |
| CHEMBL4203_Ki | R2 | 0.3337 ± 0.0536 |
| CHEMBL4616_EC50 | R2 | 0.5337 ± 0.1222 |
| CHEMBL4792_Ki | R2 | 0.4215 ± 0.0410 |

## Table C: MoleculeACE MolMCL protocol (MolMCL-GPS, existing baseline)

| dataset | metric | mean ± std |
|---|---:|---:|
| CHEMBL1862_Ki | R2 | 0.8162 ± 0.0099 |
| CHEMBL1871_Ki | R2 | 0.5363 ± 0.0702 |
| CHEMBL2034_Ki | R2 | 0.5439 ± 0.0140 |
| CHEMBL2047_EC50 | R2 | 0.3168 ± 0.0944 |
| CHEMBL204_Ki | R2 | 0.8197 ± 0.0124 |
| CHEMBL2147_Ki | R2 | 0.8834 ± 0.0100 |
| CHEMBL214_Ki | R2 | 0.6522 ± 0.0061 |
| CHEMBL218_EC50 | R2 | 0.5243 ± 0.0269 |
| CHEMBL219_Ki | R2 | 0.3767 ± 0.0238 |
| CHEMBL228_Ki | R2 | 0.6371 ± 0.0390 |
| CHEMBL231_Ki | R2 | 0.7670 ± 0.0110 |
| CHEMBL233_Ki | R2 | 0.6900 ± 0.0175 |
| CHEMBL234_Ki | R2 | 0.7046 ± 0.0052 |
| CHEMBL235_EC50 | R2 | 0.7017 ± 0.0296 |
| CHEMBL236_Ki | R2 | 0.7254 ± 0.0058 |
| CHEMBL237_EC50 | R2 | 0.6951 ± 0.0388 |
| CHEMBL237_Ki | R2 | 0.7161 ± 0.0082 |
| CHEMBL238_Ki | R2 | 0.6428 ± 0.0234 |
| CHEMBL239_EC50 | R2 | 0.5358 ± 0.0202 |
| CHEMBL244_Ki | R2 | 0.8104 ± 0.0027 |
| CHEMBL262_Ki | R2 | 0.6256 ± 0.0125 |
| CHEMBL264_Ki | R2 | 0.7400 ± 0.0056 |
| CHEMBL2835_Ki | R2 | 0.8329 ± 0.0194 |
| CHEMBL287_Ki | R2 | 0.5602 ± 0.0220 |
| CHEMBL2971_Ki | R2 | 0.8315 ± 0.0090 |
| CHEMBL3979_EC50 | R2 | 0.5729 ± 0.0316 |
| CHEMBL4005_Ki | R2 | 0.5376 ± 0.0668 |
| CHEMBL4203_Ki | R2 | 0.3630 ± 0.0351 |
| CHEMBL4616_EC50 | R2 | 0.4713 ± 0.1061 |
| CHEMBL4792_Ki | R2 | 0.5023 ± 0.0207 |
