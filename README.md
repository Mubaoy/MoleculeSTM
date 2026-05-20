# MoleculeSTM-ChemVL Benchmark

本仓库 fork 自 [chao1224/MoleculeSTM](https://github.com/chao1224/MoleculeSTM)，在原始 MoleculeSTM 代码基础上加入了 `chemvl_benchmark/` 评估模块，用于在 ChemVL 的固定数据划分和指标设置下公平评估 MoleculeSTM。

原始 MoleculeSTM 是一个分子结构-文本多模态表征模型。本 fork 的重点不是重新实现 MoleculeSTM，而是把 MoleculeSTM 接入 ChemVL evaluation protocol，便于和 ChemVL/MolMCL 等方法在相同数据划分下比较。

## 1. 本项目做了什么

新增内容集中在：

```text
chemvl_benchmark/
  README.md                 ChemVL benchmark 详细说明
  DECISIONS.md              关键修改和决策记录
  SUBMIT_FILES.md           提交文件清单
  configs/                  ChemVL-style benchmark 配置
  splits/                   固定 train/val/test 划分索引
  scripts/                  运行与汇总脚本
  results/                  已完成实验的 CSV/Markdown 汇总结果
```

支持两种 MoleculeSTM 分子输入分支：

- `Graph`：使用 MoleculeSTM graph encoder。
- `SMILES`：使用 MoleculeSTM MegaMolBART/SMILES encoder。

核心目标：

- 统一使用 ChemVL 的 MoleculeNet scaffold / random_scaffold 划分。
- 固定 `runseed = 1, 2, 3`。
- 统一分类/回归指标。
- 提交小型、可追踪的 split/config/result 文件。
- 不提交数据集、checkpoint、完整训练输出和日志。

## 2. Benchmark 范围

### Table A: MoleculeNet scaffold

数据集：

```text
bbbp, bace, clintox, tox21, sider, hiv, esol, freesolv, lipo, qm7
```

指标：

- 分类任务：ROC-AUC。
- 回归任务：RMSE。
- QM7：MAE。

### Table B: MoleculeNet random_scaffold

同样的 10 个 MoleculeNet 数据集，使用 ChemVL random_scaffold 划分。

### Table C: MoleculeACE MolMCL protocol

仓库中保留了 30 个 MoleculeACE 数据集的 config 和 split。当前已提交的 Table C 结果是 MolMCL baseline 汇总；MoleculeSTM 的 Table C 还没有完整跑完。

## 3. 环境构建

如果你已经有可以跑 MoleculeSTM 的 conda 环境，可以直接复用。否则可以按下面方式创建环境。

```bash
conda create -n molstm python=3.7
conda activate molstm

conda install -y -c rdkit rdkit=2020.09.1.0
conda install -y -c conda-forge -c pytorch pytorch=1.9.1
conda install -y -c pyg -c conda-forge pyg==2.0.3

pip install pandas scikit-learn tqdm transformers requests matplotlib spacy Levenshtein ogb==1.2.0
```

如果要运行 SMILES 分支，还需要原始 MoleculeSTM 的 MegaMolBART/Megatron/Apex 依赖。可以参考官方 MoleculeSTM 的安装方式，也可以复用已经配置好的 MoleculeSTM 环境。

## 4. 数据和 checkpoint 准备

本仓库不提交数据集和预训练权重，需要本地准备。

设置 ChemVL 数据目录：

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
```

该目录需要包含 ChemVL processed CSV，例如：

```text
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/bace/processed/bace_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/classification/bbbp/processed/bbbp_processed_ac.csv
${CHEMVL_DATA_ROOT}/finetuning_datasets/MPP/regression/esol/processed/esol_processed_ac.csv
```

MoleculeSTM checkpoint 默认使用官方目录结构：

```text
data/pretrained_MoleculeSTM/
data/pretrained_MegaMolBART/checkpoints/
MoleculeSTM/bart_vocab.txt
```

也可以用环境变量指定：

```bash
export MOLECULESTM_GRAPH_CKPT=/path/to/graph/molecule_model.pth
export MOLECULESTM_SMILES_CKPT=/path/to/smiles/molecule_model.pth
```

## 5. 快速测试

Graph 分支 smoke test：

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

SMILES 分支 smoke test：

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

只检查命令是否能生成，不真正训练：

```bash
python chemvl_benchmark/scripts/run_batch.py \
  --tables A \
  --tasks bbbp \
  --runseeds 1 \
  --molecule_type Graph \
  --dry_run
```

## 6. 完整运行方法

Graph 分支运行 Table A/B：

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
bash chemvl_benchmark/scripts/run_graph_ab.sh
```

SMILES 分支运行 Table A/B：

```bash
export CHEMVL_DATA_ROOT=/path/to/chemvl-data
bash chemvl_benchmark/scripts/run_smiles_ab.sh
```

两个脚本默认运行：

```text
Table A + Table B
runseed 1, 2, 3
```

只跑指定数据集：

```bash
bash chemvl_benchmark/scripts/run_graph_ab.sh --tasks bace bbbp
bash chemvl_benchmark/scripts/run_smiles_ab.sh --tasks bace bbbp
```

自定义输出目录：

```bash
OUTPUT_ROOT=outputs/my_graph_run bash chemvl_benchmark/scripts/run_graph_ab.sh
OUTPUT_ROOT=outputs/my_smiles_run bash chemvl_benchmark/scripts/run_smiles_ab.sh
```

## 7. 结果汇总

Graph 结果汇总：

```bash
python chemvl_benchmark/scripts/summarize.py \
  --moleculestm_output outputs/chemvl_benchmark_graph_runs \
  --output_dir outputs/chemvl_benchmark_graph_tables
```

SMILES 结果汇总：

```bash
python chemvl_benchmark/scripts/summarize.py \
  --moleculestm_output outputs/chemvl_benchmark_smiles_runs \
  --output_dir outputs/chemvl_benchmark_smiles_tables
```

汇总脚本会生成：

```text
raw_results.csv
summary_long.csv
table_A_moleculenet_scaffold.csv
table_B_moleculenet_random_scaffold.csv
```

## 8. 已提交结果

已完成实验的汇总结果位于：

```text
chemvl_benchmark/results/moleculestm_graph/
chemvl_benchmark/results/moleculestm_smiles/
```

其中：

```text
chemvl_benchmark/results/moleculestm_graph/table_A_moleculenet_scaffold.csv
chemvl_benchmark/results/moleculestm_graph/table_B_moleculenet_random_scaffold.csv
chemvl_benchmark/results/moleculestm_smiles/table_A_moleculenet_scaffold.csv
chemvl_benchmark/results/moleculestm_smiles/table_B_moleculenet_random_scaffold.csv
```

SMILES 结果使用了 `chemvl_benchmark/DECISIONS.md` 中记录的 RNG 修复。

## 9. 注意事项

不要提交以下内容：

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

这些已经在 `.gitignore` 中排除。固定 split 文件 `chemvl_benchmark/splits/**/*.npz` 是例外，需要提交，因为数据划分直接影响性能比较。

## 10. 与官方 MoleculeSTM 的关系

本仓库保留官方 MoleculeSTM 作为基础实现：

```text
https://github.com/chao1224/MoleculeSTM
```

本 fork 只新增 ChemVL benchmark wrapper、固定 split/config 和实验汇总结果。原始 MoleculeSTM 论文和模型代码归属于官方项目。
