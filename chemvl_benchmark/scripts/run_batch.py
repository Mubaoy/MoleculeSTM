import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHEMVL_ROOT = Path(os.environ.get("CHEMVL_ROOT", str(PROJECT_ROOT.parent / "ChemVL-private")))
DEFAULT_CHEMVL_DATA_ROOT = os.environ.get("CHEMVL_DATA_ROOT")
TABLE_DIRS = {
    "A": "table_A_moleculenet_scaffold",
    "B": "table_B_moleculenet_random_scaffold",
    "C": "table_C_moleculeace_molmcl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run MoleculeSTM on generated ChemVL benchmark configs/splits.")
    parser.add_argument("--tables", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--runseeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--chemvl_root", default=str(DEFAULT_CHEMVL_ROOT))
    parser.add_argument(
        "--chemvl_data_root",
        default=DEFAULT_CHEMVL_DATA_ROOT,
        help="Path to ChemVL chemvl-data. Required when configs contain ${CHEMVL_DATA_ROOT}.",
    )
    parser.add_argument("--config_root", default=str(BENCHMARK_ROOT / "configs"))
    parser.add_argument("--split_root", default=str(BENCHMARK_ROOT / "splits"))
    parser.add_argument("--output_root", default=str(PROJECT_ROOT / "outputs" / "chemvl_benchmark_runs"))
    parser.add_argument("--molecule_type", default="Graph", choices=["Graph", "SMILES"])
    parser.add_argument("--training_mode", default="fine_tuning", choices=["fine_tuning", "linear_probing"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", choices=["Adam", "AdamW"], default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--input_model_path", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--max_runs", type=int, default=None)
    return parser.parse_args()


def read_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jobs(args) -> Iterable[Dict]:
    config_root = Path(args.config_root)
    split_root = Path(args.split_root)
    task_filter = set(args.tasks or [])
    for table in args.tables:
        table_dir = TABLE_DIRS[table]
        cfg_dir = config_root / table_dir
        split_dir = split_root / table_dir
        if not cfg_dir.is_dir():
            raise FileNotFoundError(f"Missing config directory: {cfg_dir}")
        for cfg_path in sorted(cfg_dir.glob("*.json")):
            cfg = read_config(cfg_path)
            task = cfg.get("benchmark", {}).get("task_alias") or cfg_path.stem
            if task_filter and task not in task_filter and cfg_path.stem not in task_filter:
                continue
            split_path = split_dir / f"{cfg_path.stem}.npz"
            if not split_path.is_file():
                raise FileNotFoundError(f"Missing split file: {split_path}")
            for runseed in args.runseeds:
                out_dir = Path(args.output_root) / table_dir / task / f"runseed_{runseed}"
                yield {
                    "table": table,
                    "task": task,
                    "config": cfg_path,
                    "split": split_path,
                    "runseed": runseed,
                    "output_dir": out_dir,
                }


def build_command(args, job: Dict) -> List[str]:
    cmd = [
        sys.executable,
        str(BENCHMARK_ROOT / "scripts" / "run_on_chemvl_split.py"),
        "--chemvl_root",
        str(Path(args.chemvl_root).resolve()),
        "--config",
        str(job["config"]),
        "--split_indices_path",
        str(job["split"]),
        "--molecule_type",
        args.molecule_type,
        "--training_mode",
        args.training_mode,
        "--device",
        str(args.device),
        "--runseed",
        str(job["runseed"]),
        "--output_dir",
        str(job["output_dir"]),
    ]
    if args.chemvl_data_root:
        cmd.extend(["--chemvl_data_root", str(Path(args.chemvl_data_root).resolve())])
    optional = {
        "--epochs": args.epochs,
        "--batch_size": args.batch_size,
        "--lr": args.lr,
        "--weight_decay": args.weight_decay,
        "--optimizer": args.optimizer,
        "--num_workers": args.num_workers,
        "--input_model_path": args.input_model_path,
    }
    for flag, value in optional.items():
        if value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def main():
    args = parse_args()
    jobs = list(iter_jobs(args))
    if args.max_runs is not None:
        jobs = jobs[: args.max_runs]
    if not jobs:
        print("No jobs selected.", flush=True)
        return

    for idx, job in enumerate(jobs, start=1):
        result_path = Path(job["output_dir"]) / "result.json"
        if result_path.is_file() and not args.overwrite:
            print(f"[{idx}/{len(jobs)}] skip existing {result_path}", flush=True)
            continue
        cmd = build_command(args, job)
        print(f"[{idx}/{len(jobs)}] {job['table']} {job['task']} runseed={job['runseed']}", flush=True)
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)

        failed = None
        for attempt in range(args.retries + 1):
            try:
                subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
                failed = None
                break
            except subprocess.CalledProcessError as exc:
                failed = exc
                print(f"Job failed on attempt {attempt + 1}/{args.retries + 1}: {exc}", flush=True)
                if attempt < args.retries:
                    time.sleep(10)
        if failed is not None:
            fail_path = Path(job["output_dir"]) / "failed.json"
            with fail_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "table": job["table"],
                        "task": job["task"],
                        "runseed": job["runseed"],
                        "returncode": failed.returncode,
                        "command": cmd,
                    },
                    f,
                    indent=2,
                )
            if not args.continue_on_error:
                raise failed
            print(f"Recorded failed job and continuing: {fail_path}", flush=True)


if __name__ == "__main__":
    main()
