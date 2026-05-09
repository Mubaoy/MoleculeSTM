import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader as PyGDataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MoleculeSTM.datasets.utils import mol_to_graph_data_obj_simple
from MoleculeSTM.models import GNN, GNN_graphpred
from MoleculeSTM.utils import get_molecule_repr_MoleculeSTM

DEFAULT_CHEMVL_ROOT = os.environ.get("CHEMVL_ROOT", str(PROJECT_ROOT.parent / "ChemVL-private"))
DEFAULT_GRAPH_CKPT = os.environ.get(
    "MOLECULESTM_GRAPH_CKPT",
    str(
        PROJECT_ROOT
        / "data"
        / "pretrained_MoleculeSTM"
        / "SciBERT-Graph-3e-5-1-1e-4-1-InfoNCE-0.1-32-32"
        / "molecule_model.pth"
    ),
)
DEFAULT_SMILES_CKPT = os.environ.get(
    "MOLECULESTM_SMILES_CKPT",
    str(
        PROJECT_ROOT
        / "data"
        / "pretrained_MoleculeSTM"
        / "SciBERT-MegaMolBART-1e-5-1-1e-4-1-EBM_NCE-0.1-32-32"
        / "molecule_model.pth"
    ),
)
DEFAULT_MEGAMOLBART_INPUT_DIR = str(PROJECT_ROOT / "data" / "pretrained_MegaMolBART" / "checkpoints")
DEFAULT_VOCAB_PATH = str(PROJECT_ROOT / "MoleculeSTM" / "bart_vocab.txt")


class ChemVLSMILESDataset(Dataset):
    def __init__(self, smiles, labels):
        self.smiles = list(smiles)
        self.labels = np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, index):
        return self.smiles[index], torch.tensor(self.labels[index], dtype=torch.float32)


class ChemVLGraphDataset(Dataset):
    def __init__(self, smiles, labels):
        labels = np.asarray(labels, dtype=np.float32)
        self.data_list = []
        for index, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("Invalid SMILES at index {}: {}".format(index, smi))
            data = mol_to_graph_data_obj_simple(mol)
            data.id = torch.tensor([index])
            data.y = torch.tensor(labels[index], dtype=torch.float32)
            data.smiles = smi
            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MoleculeSTM property prediction on ChemVL data with ChemVL split indices."
    )
    parser.add_argument("--chemvl_root", type=str, default=DEFAULT_CHEMVL_ROOT)
    parser.add_argument(
        "--chemvl_data_root",
        type=str,
        default=os.environ.get("CHEMVL_DATA_ROOT"),
        help="Path to ChemVL chemvl-data. Used to expand ${CHEMVL_DATA_ROOT} in benchmark configs.",
    )
    parser.add_argument("--config", type=str, required=True, help="ChemVL JSON config to mirror.")
    parser.add_argument("--molecule_type", type=str, default="Graph", choices=["Graph", "SMILES"])
    parser.add_argument("--training_mode", type=str, default="fine_tuning", choices=["fine_tuning", "linear_probing"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=[None, "Adam", "AdamW"])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--runseed", type=int, default=None, help="Override cfg['training']['runseed'].")
    parser.add_argument("--eval_train", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument(
        "--split_indices_path",
        type=str,
        default=None,
        help="Optional .npz containing train_idx/val_idx/test_idx exported by ChemVL.",
    )

    parser.add_argument("--input_model_path", type=str, default=None)
    parser.add_argument("--megamolbart_input_dir", type=str, default=DEFAULT_MEGAMOLBART_INPUT_DIR)
    parser.add_argument("--vocab_path", type=str, default=DEFAULT_VOCAB_PATH)

    parser.add_argument("--gnn_emb_dim", type=int, default=300)
    parser.add_argument("--num_layer", type=int, default=5)
    parser.add_argument("--JK", type=str, default="last")
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
    parser.add_argument("--gnn_type", type=str, default="gin")
    parser.add_argument("--graph_pooling", type=str, default="mean")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def expand_placeholders(value, replacements):
    if isinstance(value, str):
        for key, replacement in replacements.items():
            token = "${" + key + "}"
            if token in value:
                if not replacement:
                    raise ValueError(f"Config requires {token}; pass --{key.lower()} or set {key}.")
                value = value.replace(token, replacement)
        return value
    if isinstance(value, list):
        return [expand_placeholders(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: expand_placeholders(item, replacements) for key, item in value.items()}
    return value


def expand_config(cfg, args):
    replacements = {
        "CHEMVL_DATA_ROOT": str(Path(args.chemvl_data_root).resolve()) if args.chemvl_data_root else None,
        "MOLECULESTM_ROOT": str(PROJECT_ROOT),
    }
    return expand_placeholders(cfg, replacements)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def add_chemvl_to_path(chemvl_root):
    chemvl_root = os.path.abspath(chemvl_root)
    if chemvl_root not in sys.path:
        sys.path.insert(0, chemvl_root)


def get_chemvl_csv_path(cfg):
    dataset_cfg = cfg["dataset"]
    dataset = dataset_cfg["dataset"]
    dataroot = dataset_cfg["dataroot"]
    csv_path = os.path.join(dataroot, dataset, "processed", "{}_processed_ac.csv".format(dataset))
    if not os.path.exists(csv_path):
        raise FileNotFoundError("ChemVL processed CSV not found: {}".format(csv_path))
    return csv_path


def load_chemvl_table(cfg):
    csv_path = get_chemvl_csv_path(cfg)
    df = pd.read_csv(csv_path)
    if "smiles" not in df.columns:
        raise ValueError("Expected a 'smiles' column in {}".format(csv_path))
    label_cols = [c for c in df.columns if c not in ("index", "smiles")]
    if not label_cols:
        raise ValueError("No label columns found in {}".format(csv_path))
    smiles = df["smiles"].astype(str).tolist()
    if len(label_cols) == 1:
        labels = np.array(df[label_cols[0]].apply(lambda x: str(x).split(" ")).tolist(), dtype=np.float32)
    else:
        labels = df[label_cols].values.astype(np.float32)
    return df, smiles, labels, csv_path, label_cols


def compute_chemvl_split(cfg, smiles, labels, chemvl_root):
    add_chemvl_to_path(chemvl_root)
    split_name = cfg["dataset"].get("split")
    indices = list(range(len(smiles)))

    if cfg["dataset"].get("benchmark") == "moleculeace" or split_name == "molmcl":
        from utils.moleculeace_molmcl import moleculeace_split

        split_cfg = cfg["dataset"].get("moleculeace_split") or {}
        train_idx, val_idx, test_idx = moleculeace_split(
            list(smiles),
            np.asarray(labels, dtype=float).reshape(-1).tolist(),
            in_log10=bool(cfg["dataset"].get("moleculeace_in_log10", True)),
            n_clusters=int(split_cfg.get("n_clusters", 5)),
            val_size=float(split_cfg.get("val_size", 0.1)),
            test_size=float(split_cfg.get("test_size", 0.1)),
            similarity=float(split_cfg.get("similarity", 0.9)),
            potency_fold=int(split_cfg.get("potency_fold", 10)),
            remove_stereo=bool(split_cfg.get("remove_stereo", False)),
        )
        return np.array(train_idx), np.array(val_idx), np.array(test_idx), "moleculeace_molmcl"

    from utils.splitter import (
        random_scaffold_split_train_val_test,
        scaffold_split_balanced_train_val_test,
        scaffold_split_train_val_test,
        split_train_val_test_idx,
        split_train_val_test_idx_stratified_v2,
    )

    seed = int(cfg["training"].get("seed", 42))
    chirality = bool(cfg["dataset"].get("chirality", True))
    if split_name == "random":
        train_idx, val_idx, test_idx = split_train_val_test_idx(
            indices, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
        )
    elif split_name == "stratified":
        train_idx, val_idx, test_idx = split_train_val_test_idx_stratified_v2(
            indices, labels, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed
        )
    elif split_name == "scaffold":
        train_idx, val_idx, test_idx = scaffold_split_train_val_test(
            indices, smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1, include_chirality=chirality
        )
    elif split_name == "random_scaffold":
        train_idx, val_idx, test_idx = random_scaffold_split_train_val_test(
            indices, smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=seed, include_chirality=chirality
        )
    elif split_name == "scaffold_balanced":
        train_idx, val_idx, test_idx = scaffold_split_balanced_train_val_test(
            indices,
            smiles,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=seed,
            balanced=True,
            include_chirality=chirality,
        )
    else:
        raise ValueError("Unsupported ChemVL split: {}".format(split_name))

    return np.array(train_idx), np.array(val_idx), np.array(test_idx), split_name


def load_split_indices(path, total):
    data = np.load(path)
    train_idx = data["train_idx"].astype(int)
    val_idx = data["val_idx"].astype(int)
    test_idx = data["test_idx"].astype(int)
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    if len(np.unique(all_idx)) != len(all_idx):
        raise ValueError("Split indices overlap in {}".format(path))
    if np.min(all_idx) < 0 or np.max(all_idx) >= total:
        raise ValueError("Split indices out of range for total={} in {}".format(total, path))
    return train_idx, val_idx, test_idx


def infer_metric(cfg):
    task_type = cfg["dataset"]["task_type"]
    if task_type == "classification":
        return "rocauc", "max"
    metric = str(cfg.get("training", {}).get("eval_metric") or "").lower()
    if metric == "r2":
        return "r2", "max"
    dataset = cfg["dataset"]["dataset"]
    if dataset in ["qm7", "qm8", "qm9"]:
        return "mae", "min"
    return "rmse", "min"


def build_dataset(args, smiles, labels):
    if args.molecule_type == "Graph":
        return ChemVLGraphDataset(smiles, labels)
    return ChemVLSMILESDataset(smiles, labels)


def build_loaders(args, dataset, train_idx, val_idx, test_idx, batch_size, num_workers, seed):
    loader_cls = PyGDataLoader if args.molecule_type == "Graph" else TorchDataLoader
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = loader_cls(
        Subset(dataset, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        generator=generator,
    )
    val_loader = loader_cls(Subset(dataset, val_idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = loader_cls(Subset(dataset, test_idx.tolist()), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def resolve_input_model_path(args):
    if args.input_model_path:
        return args.input_model_path
    return DEFAULT_GRAPH_CKPT if args.molecule_type == "Graph" else DEFAULT_SMILES_CKPT


def load_shape_compatible_state_dict(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in checkpoint.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        print("Skipped shape-incompatible checkpoint keys: {}".format(", ".join(skipped)))
    if unexpected:
        print("Unexpected checkpoint keys after filtering: {}".format(", ".join(unexpected)))
    if missing:
        print("Missing model keys after partial load: {}".format(", ".join(missing)))


def build_model(args, num_tasks, device, seed=None):
    input_model_path = resolve_input_model_path(args)
    megamolbart_wrapper = None

    if args.molecule_type == "SMILES":
        from MoleculeSTM.models.mega_molbart.mega_mol_bart import MegaMolBART

        megamolbart_wrapper = MegaMolBART(
            vocab_path=args.vocab_path,
            input_dir=args.megamolbart_input_dir,
            output_dir=None,
        )
        model = megamolbart_wrapper.model
        molecule_dim = 256
        if input_model_path and os.path.exists(input_model_path):
            print("Loading MoleculeSTM SMILES molecule model from {}.".format(input_model_path))
            model.load_state_dict(torch.load(input_model_path, map_location="cpu"))
    else:
        molecule_node_model = GNN(
            num_layer=args.num_layer,
            emb_dim=args.gnn_emb_dim,
            JK=args.JK,
            drop_ratio=args.dropout_ratio,
            gnn_type=args.gnn_type,
        )
        model = GNN_graphpred(
            num_layer=args.num_layer,
            emb_dim=args.gnn_emb_dim,
            JK=args.JK,
            graph_pooling=args.graph_pooling,
            num_tasks=num_tasks,
            molecule_node_model=molecule_node_model,
        )
        molecule_dim = args.gnn_emb_dim
        if input_model_path and os.path.exists(input_model_path):
            print("Loading MoleculeSTM Graph molecule model from {}.".format(input_model_path))
            if "GraphMVP" in input_model_path:
                model.from_pretrained(input_model_path)
            else:
                load_shape_compatible_state_dict(model, input_model_path)
        else:
            print("No graph checkpoint found; using randomly initialized GNN.")

    model = model.to(device)
    # MegaMolBART/Megatron checkpoint loading can restore checkpoint RNG state.
    # Re-apply the run seed before creating the prediction head.
    if seed is not None:
        set_seed(seed)
    head = nn.Linear(molecule_dim, num_tasks).to(device)
    return model, head, megamolbart_wrapper, input_model_path


def forward_batch(args, model, head, megamolbart_wrapper, batch, device, train_encoder, num_tasks):
    if args.molecule_type == "SMILES":
        smiles_list, y = batch
        smiles_list = list(smiles_list)
        y = y.to(device).float()
        with torch.set_grad_enabled(train_encoder):
            molecule_repr = get_molecule_repr_MoleculeSTM(
                smiles_list,
                mol2latent=None,
                molecule_type="SMILES",
                MegaMolBART_wrapper=megamolbart_wrapper,
            )
    else:
        batch = batch.to(device)
        y = batch.y.view(-1, num_tasks).to(device).float()
        with torch.set_grad_enabled(train_encoder):
            molecule_repr = get_molecule_repr_MoleculeSTM(
                batch,
                mol2latent=None,
                molecule_type="Graph",
                molecule_model=model,
            )
    if not train_encoder:
        molecule_repr = molecule_repr.detach()
    return head(molecule_repr).float(), y


def train_one_epoch(args, model, head, megamolbart_wrapper, loader, optimizer, criterion, task_type, device, num_tasks):
    train_encoder = args.training_mode == "fine_tuning"
    model.train() if train_encoder else model.eval()
    head.train()
    total_loss = 0.0

    for batch in loader:
        pred, y = forward_batch(args, model, head, megamolbart_wrapper, batch, device, train_encoder, num_tasks)
        if task_type == "classification":
            valid = y >= 0
            loss_mat = criterion(pred, y)
            loss = loss_mat[valid].mean()
        else:
            loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu().item())

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(args, model, head, megamolbart_wrapper, loader, task_type, device, num_tasks):
    model.eval()
    head.eval()
    y_true_list, y_pred_list = [], []
    for batch in loader:
        pred, y = forward_batch(args, model, head, megamolbart_wrapper, batch, device, False, num_tasks)
        y_true_list.append(y.detach().cpu())
        y_pred_list.append(pred.detach().cpu())

    y_true = torch.cat(y_true_list, dim=0).numpy()
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    if task_type == "classification":
        roc_list = []
        acc_list = []
        for task_idx in range(y_true.shape[1]):
            valid = y_true[:, task_idx] >= 0
            if np.sum(valid) == 0:
                continue
            task_y = y_true[valid, task_idx]
            task_pred = y_pred[valid, task_idx]
            if np.sum(task_y == 1) > 0 and np.sum(task_y == 0) > 0:
                roc_list.append(roc_auc_score(task_y, task_pred))
            acc_list.append(accuracy_score(task_y, (task_pred >= 0).astype(np.float32)))
        return {
            "rocauc": float(np.mean(roc_list)) if roc_list else float("nan"),
            "acc": float(np.mean(acc_list)) if acc_list else float("nan"),
        }, y_true, y_pred

    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }, y_true, y_pred


def is_better(left, right, mode):
    if np.isnan(left):
        return False
    return left > right if mode == "max" else left < right


def make_output_dir(args, cfg):
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        config_stem = Path(args.config).stem
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
        out = Path("outputs") / "moleculestm_on_chemvl_split" / config_stem / timestamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def main(args):
    cfg = expand_config(load_json(args.config), args)
    df, smiles, labels, csv_path, label_cols = load_chemvl_table(cfg)

    task_type = cfg["dataset"]["task_type"]
    num_tasks = int(labels.shape[1])
    metric_name, metric_mode = infer_metric(cfg)

    runseed = int(args.runseed if args.runseed is not None else cfg["training"].get("runseed", cfg["training"].get("seed", 42)))
    set_seed(runseed)

    if args.split_indices_path:
        train_idx, val_idx, test_idx = load_split_indices(args.split_indices_path, len(smiles))
        split_source = "external_npz:{}".format(os.path.abspath(args.split_indices_path))
    else:
        train_idx, val_idx, test_idx, split_source = compute_chemvl_split(cfg, smiles, labels, args.chemvl_root)

    epochs = int(args.epochs if args.epochs is not None else cfg["training"]["epochs"])
    batch_size = int(args.batch_size if args.batch_size is not None else cfg["training"]["batch_size"])
    lr = float(args.lr if args.lr is not None else cfg["training"]["lr"])
    weight_decay = float(args.weight_decay if args.weight_decay is not None else cfg["training"].get("weight_decay", 0.0))
    optimizer_name = args.optimizer or cfg["training"].get("optimizer") or "Adam"
    num_workers = int(args.num_workers if args.num_workers is not None else cfg["basic"].get("num_workers", 0))

    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args, smiles, labels)
    model, head, megamolbart_wrapper, input_model_path = build_model(args, num_tasks, device, seed=runseed)
    train_loader, val_loader, test_loader = build_loaders(
        args, dataset, train_idx, val_idx, test_idx, batch_size, num_workers, runseed
    )

    if args.training_mode == "fine_tuning":
        params = list(model.parameters()) + list(head.parameters())
    else:
        params = list(head.parameters())
    optimizer_cls = optim.AdamW if optimizer_name.lower() == "adamw" else optim.Adam
    optimizer = optimizer_cls(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none") if task_type == "classification" else nn.MSELoss()

    output_dir = make_output_dir(args, cfg)
    np.savez(output_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    run_config = {
        "chemvl_config": os.path.abspath(args.config),
        "chemvl_csv": csv_path,
        "label_cols": label_cols,
        "split_source": split_source,
        "molecule_type": args.molecule_type,
        "training_mode": args.training_mode,
        "input_model_path": input_model_path,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "num_workers": num_workers,
        "metric_name": metric_name,
        "metric_mode": metric_mode,
        "runseed": runseed,
        "benchmark": cfg.get("benchmark", {}),
        "dataset": cfg.get("dataset", {}),
        "sizes": {
            "total": len(df),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
    }
    with open(output_dir / "config_used.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(json.dumps(run_config, indent=2))
    best_value = -np.inf if metric_mode == "max" else np.inf
    best_epoch = 0
    best_test_value = None
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            args, model, head, megamolbart_wrapper, train_loader, optimizer, criterion, task_type, device, num_tasks
        )
        train_metrics = evaluate(args, model, head, megamolbart_wrapper, train_loader, task_type, device, num_tasks)[0] if args.eval_train else {}
        val_metrics, val_target, val_pred = evaluate(args, model, head, megamolbart_wrapper, val_loader, task_type, device, num_tasks)
        test_metrics, test_target, test_pred = evaluate(args, model, head, megamolbart_wrapper, test_loader, task_type, device, num_tasks)

        row = {"epoch": epoch, "train_loss": train_loss}
        for key, value in train_metrics.items():
            row["train_{}".format(key)] = value
        for key, value in val_metrics.items():
            row["val_{}".format(key)] = value
        for key, value in test_metrics.items():
            row["test_{}".format(key)] = value
        history.append(row)
        print(row)

        val_value = val_metrics[metric_name]
        if is_better(val_value, best_value, metric_mode):
            best_value = val_value
            best_test_value = test_metrics[metric_name]
            best_epoch = epoch
            np.savez(output_dir / "best_predictions.npz", val_target=val_target, val_pred=val_pred, test_target=test_target, test_pred=test_pred)
            if args.save_model:
                torch.save({"model": model.state_dict(), "head": head.state_dict()}, output_dir / "best_model.pth")

    pd.DataFrame(history).to_csv(output_dir / "train_val_test_history.csv", index=False)
    result = {
        "best_valid": float(best_value),
        "best_valid_on_test": float(best_test_value) if best_test_value is not None else None,
        "best_valid_epoch": int(best_epoch),
        "metric": metric_name,
        "metric_mode": metric_mode,
        "output_dir": str(output_dir),
    }
    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    args_global = parse_args()
    main(args_global)
