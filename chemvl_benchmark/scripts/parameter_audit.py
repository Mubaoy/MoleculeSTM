#!/usr/bin/env python3
"""Count MoleculeSTM trainable/total parameters for the ChemVL protocol."""

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_on_chemvl_split import build_model, load_json, set_seed  # noqa: E402


DEFAULT_CONFIG = PROJECT_ROOT / "chemvl_benchmark" / "configs" / "table_A_moleculenet_scaffold" / "bbbp.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "parameter_summary.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit MoleculeSTM parameters under the local ChemVL fine-tuning setup."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="CSV output path.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="ChemVL config used to infer the prediction-head size. Default: Table A BBBP, num_tasks=1.",
    )
    parser.add_argument(
        "--molecule_types",
        nargs="+",
        default=["Graph", "SMILES"],
        choices=["Graph", "SMILES"],
        help="MoleculeSTM branches to audit.",
    )
    parser.add_argument(
        "--training_modes",
        nargs="+",
        default=["fine_tuning"],
        choices=["fine_tuning", "linear_probing"],
        help="Finetune strategies to audit. Default matches the reported ChemVL A/B runs.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--runseed", type=int, default=1)
    parser.add_argument("--strict", action="store_true", help="Fail on the first model-construction error.")
    parser.add_argument("--graph_ckpt", default=os.environ.get("MOLECULESTM_GRAPH_CKPT"))
    parser.add_argument("--smiles_ckpt", default=os.environ.get("MOLECULESTM_SMILES_CKPT"))
    parser.add_argument(
        "--megamolbart_input_dir",
        default=os.environ.get("MEGAMOLBART_INPUT_DIR"),
        help="MegaMolBART checkpoint directory used by the SMILES branch.",
    )
    parser.add_argument(
        "--vocab_path",
        default=os.environ.get("MEGAMOLBART_VOCAB_PATH", str(PROJECT_ROOT / "MoleculeSTM" / "bart_vocab.txt")),
    )
    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--JK", default="last")
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", default="gin")
    parser.add_argument("--graph_pooling", default="mean")
    return parser.parse_args()


def format_count(value):
    if value == "" or value is None:
        return ""
    if value >= 1_000_000:
        return "{:.2f}M".format(value / 1_000_000)
    if value >= 1_000:
        return "{:.2f}K".format(value / 1_000)
    return str(value)


def make_row(paper, variant, strategy, scope, task_alias, num_tasks, trainable, total, status, notes):
    return {
        "paper": paper,
        "variant": variant,
        "finetune_strategy": strategy,
        "scope": scope,
        "task_alias": task_alias,
        "num_tasks": num_tasks,
        "trainable": trainable,
        "total": total,
        "trainable_formatted": format_count(trainable),
        "total_formatted": format_count(total),
        "trainable_over_total": "{} / {}".format(format_count(trainable), format_count(total)),
        "status": status,
        "notes": notes,
    }


def make_error_row(cfg, molecule_type, training_mode, args, exc):
    return {
        "paper": "MoleculeSTM",
        "variant": molecule_type,
        "finetune_strategy": "FT" if training_mode == "fine_tuning" else "LP",
        "scope": "ChemVL representative task from {}".format(args.config),
        "task_alias": cfg.get("benchmark", {}).get("task_alias", ""),
        "num_tasks": cfg.get("dataset", {}).get("num_tasks", ""),
        "trainable": "",
        "total": "",
        "trainable_formatted": "",
        "total_formatted": "",
        "trainable_over_total": "ERROR",
        "status": "ERROR",
        "notes": "{}: {}\n{}".format(type(exc).__name__, exc, traceback.format_exc(limit=1).strip()),
    }


def count_parameters(modules):
    params = []
    for module in modules:
        params.extend(list(module.parameters()))
    total = sum(param.numel() for param in params)
    trainable = sum(param.numel() for param in params if param.requires_grad)
    return trainable, total


def apply_training_mode(model, head, training_mode):
    train_encoder = training_mode == "fine_tuning"
    for param in model.parameters():
        param.requires_grad = train_encoder
    for param in head.parameters():
        param.requires_grad = True


def resolve_existing_path(path):
    if not path:
        return None
    path = Path(path).expanduser()
    if path.exists():
        return str(path)
    return None


def default_checkpoint(molecule_type):
    if molecule_type == "Graph":
        return str(
            PROJECT_ROOT
            / "data"
            / "pretrained_MoleculeSTM"
            / "SciBERT-Graph-3e-5-1-1e-4-1-InfoNCE-0.1-32-32"
            / "molecule_model.pth"
        )
    return str(
        PROJECT_ROOT
        / "data"
        / "pretrained_MoleculeSTM"
        / "SciBERT-MegaMolBART-1e-5-1-1e-4-1-EBM_NCE-0.1-32-32"
        / "molecule_model.pth"
    )


def make_runner_args(args, molecule_type, training_mode):
    if molecule_type == "Graph":
        input_model_path = resolve_existing_path(args.graph_ckpt) or resolve_existing_path(default_checkpoint("Graph"))
    else:
        input_model_path = resolve_existing_path(args.smiles_ckpt) or resolve_existing_path(default_checkpoint("SMILES"))

    return SimpleNamespace(
        molecule_type=molecule_type,
        training_mode=training_mode,
        device=args.device,
        input_model_path=input_model_path,
        megamolbart_input_dir=resolve_existing_path(args.megamolbart_input_dir),
        vocab_path=args.vocab_path,
        gnn_emb_dim=args.gnn_emb_dim,
        num_layer=args.num_layer,
        JK=args.JK,
        dropout_ratio=args.dropout_ratio,
        gnn_type=args.gnn_type,
        graph_pooling=args.graph_pooling,
    )


def audit_moleculestm(args, cfg, molecule_type, training_mode):
    num_tasks = int(cfg["dataset"]["num_tasks"])
    task_alias = cfg["benchmark"]["task_alias"]
    split_name = cfg["benchmark"]["split"]
    table_name = cfg["benchmark"]["table"]

    runner_args = make_runner_args(args, molecule_type, training_mode)
    if molecule_type == "SMILES" and not torch.cuda.is_available():
        raise RuntimeError("MegaMolBART construction requires CUDA in the upstream MoleculeSTM wrapper.")

    device = torch.device(
        "cuda:{}".format(args.device)
        if molecule_type == "SMILES" and torch.cuda.is_available()
        else "cpu"
    )
    set_seed(args.runseed)
    model, head, megamolbart_wrapper, input_model_path = build_model(
        runner_args, num_tasks=num_tasks, device=device, seed=args.runseed
    )
    apply_training_mode(model, head, training_mode)
    trainable, total = count_parameters([model, head])
    checkpoint_status = "loaded" if input_model_path and Path(input_model_path).exists() else "not_loaded"

    del head
    del model
    del megamolbart_wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return make_row(
        "MoleculeSTM",
        molecule_type,
        "FT" if training_mode == "fine_tuning" else "LP",
        "ChemVL Table {} {}, representative task {}".format(table_name, split_name, task_alias),
        task_alias,
        num_tasks,
        trainable,
        total,
        "OK",
        "Counted encoder plus ChemVL prediction head after applying {} requires_grad logic. "
        "Checkpoint loading does not change parameter count. checkpoint_status={}".format(
            training_mode, checkpoint_status
        ),
    )


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper",
        "variant",
        "finetune_strategy",
        "scope",
        "task_alias",
        "num_tasks",
        "trainable",
        "total",
        "trainable_formatted",
        "total_formatted",
        "trainable_over_total",
        "status",
        "notes",
    ]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main():
    args = parse_args()
    cfg = load_json(args.config)
    rows = []
    for molecule_type in args.molecule_types:
        for training_mode in args.training_modes:
            try:
                rows.append(audit_moleculestm(args, cfg, molecule_type, training_mode))
            except Exception as exc:
                if args.strict:
                    raise
                rows.append(make_error_row(cfg, molecule_type, training_mode, args, exc))

    output_path = write_csv(rows, args.output)
    print("Wrote {}".format(output_path))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
