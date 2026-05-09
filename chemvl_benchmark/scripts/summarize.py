import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOLECULESTM_OUTPUT = PROJECT_ROOT / "outputs" / "chemvl_benchmark_runs"
DEFAULT_CHEMVL_RESULTS = os.environ.get("CHEMVL_RESULTS")
if DEFAULT_CHEMVL_RESULTS is None and os.environ.get("CHEMVL_DATA_ROOT"):
    DEFAULT_CHEMVL_RESULTS = str(Path(os.environ["CHEMVL_DATA_ROOT"]) / "results")
DEFAULT_OUTPUT_DIR = BENCHMARK_ROOT / "results" / "generated_summary"

MOLECULENET_ORDER = ["bbbp", "bace", "clintox", "tox21", "sider", "hiv", "esol", "freesolv", "lipo", "qm7"]
TABLE_FILES = {
    "A": "table_A_moleculenet_scaffold.csv",
    "B": "table_B_moleculenet_random_scaffold.csv",
    "C": "table_C_moleculeace_molmcl.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize ChemVL split benchmark results.")
    parser.add_argument("--moleculestm_output", default=str(DEFAULT_MOLECULESTM_OUTPUT))
    parser.add_argument("--chemvl_results", default=DEFAULT_CHEMVL_RESULTS)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--include_molmcl", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Warning: failed to read {path}: {exc}")
        return None


def split_from_table(table: str) -> str:
    if table == "A":
        return "scaffold"
    if table == "B":
        return "random_scaffold"
    return "MolMCL"


def collect_moleculestm(root: Path) -> List[Dict]:
    rows = []
    for result_path in sorted(root.rglob("result.json")):
        result = read_json(result_path)
        cfg = read_json(result_path.parent / "config_used.json") or {}
        if not result:
            continue
        bench = cfg.get("benchmark") or {}
        dataset = cfg.get("dataset") or {}
        table = bench.get("table")
        if table is None:
            for part in result_path.parts:
                if part.startswith("table_A"):
                    table = "A"
                elif part.startswith("table_B"):
                    table = "B"
                elif part.startswith("table_C"):
                    table = "C"
        if table is None:
            continue
        value = result.get("best_valid_on_test")
        if value is None:
            continue
        rows.append(
            {
                "table": table,
                "benchmark": bench.get("benchmark") or dataset.get("benchmark") or ("moleculeace" if table == "C" else "moleculenet"),
                "split": bench.get("split") or split_from_table(table),
                "task": bench.get("task_alias") or dataset.get("dataset") or result_path.parents[1].name,
                "method": f"MoleculeSTM-{cfg.get('molecule_type', 'Graph')}",
                "metric": str(result.get("metric") or bench.get("metric") or "").lower(),
                "value": float(value),
                "best_valid": result.get("best_valid"),
                "best_epoch": result.get("best_valid_epoch"),
                "run_id": result_path.parent.name,
                "source": str(result_path),
            }
        )
    return rows


def method_from_version(version: str) -> str:
    if version.startswith("molmcl_gps"):
        return "MolMCL-GPS"
    if version.startswith("molmcl_gin"):
        return "MolMCL-GIN"
    return version


def table_from_molmcl(benchmark: str, version: str) -> str:
    if benchmark == "moleculeace":
        return "C"
    if "random-scaffold" in version:
        return "B"
    return "A"


def metric_from_molmcl(benchmark: str, task: str, version: str) -> str:
    if benchmark == "moleculeace":
        return "r2"
    if "cls" in version:
        return "rocauc"
    return "mae" if task == "qm7" else "rmse"


def collect_molmcl(root: Path) -> List[Dict]:
    rows = []
    for benchmark in ["moleculenet", "moleculeace"]:
        base = root / benchmark / "molmcl_under_chemvl"
        if not base.is_dir():
            continue
        for result_path in sorted(base.glob("*/*/*/result.json")):
            result = read_json(result_path)
            if not result or result.get("best_valid_on_test") is None:
                continue
            version = result_path.parents[2].name
            task = result_path.parents[1].name
            table = table_from_molmcl(benchmark, version)
            rows.append(
                {
                    "table": table,
                    "benchmark": benchmark,
                    "split": split_from_table(table),
                    "task": task,
                    "method": method_from_version(version),
                    "metric": metric_from_molmcl(benchmark, task, version),
                    "value": float(result["best_valid_on_test"]),
                    "best_valid": result.get("best_valid"),
                    "best_epoch": result.get("best_valid_epoch"),
                    "run_id": result_path.parent.name,
                    "source": str(result_path),
                }
            )
    return rows


def format_value(mean: float, std: float, n: int) -> str:
    if n <= 1 or np.isnan(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} +/- {std:.4f}"


def ordered_tasks(table: str, tasks: List[str]) -> List[str]:
    if table in ("A", "B"):
        return [task for task in MOLECULENET_ORDER if task in tasks] + [task for task in sorted(tasks) if task not in MOLECULENET_ORDER]
    return sorted(tasks)


def write_table(summary: pd.DataFrame, table: str, output_dir: Path):
    cur = summary[summary["table"] == table].copy()
    if cur.empty:
        return
    cur["formatted"] = [format_value(row.mean, row.std, int(row.n)) for row in cur.itertuples(index=False)]
    pivot = cur.pivot_table(index=["task", "metric"], columns="method", values="formatted", aggfunc="first", sort=False).reset_index()
    order = ordered_tasks(table, pivot["task"].tolist())
    pivot["task"] = pd.Categorical(pivot["task"], categories=order, ordered=True)
    pivot = pivot.sort_values(["task", "metric"]).astype({"task": str})
    pivot.to_csv(output_dir / TABLE_FILES[table], index=False)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_moleculestm(Path(args.moleculestm_output))
    if args.include_molmcl:
        if not args.chemvl_results:
            raise SystemExit("Pass --chemvl_results or set CHEMVL_RESULTS/CHEMVL_DATA_ROOT when using --include_molmcl.")
        rows.extend(collect_molmcl(Path(args.chemvl_results)))
    if not rows:
        raise SystemExit("No result rows found.")

    raw = pd.DataFrame(rows)
    raw.to_csv(output_dir / "raw_results.csv", index=False)
    summary = (
        raw.groupby(["table", "benchmark", "split", "task", "method", "metric"], dropna=False)
        .agg(mean=("value", "mean"), std=("value", "std"), n=("value", "count"))
        .reset_index()
    )
    summary.to_csv(output_dir / "summary_long.csv", index=False)
    for table in ["A", "B", "C"]:
        write_table(summary, table, output_dir)
    print(f"Wrote summary files to {output_dir}")


if __name__ == "__main__":
    main()
