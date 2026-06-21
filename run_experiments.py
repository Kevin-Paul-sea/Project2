"""
Run the comparison experiments required by the CIFAR-10 task.
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from train_cifar import run_training
from utils.nn import save_csv, save_json


def experiment_specs():
    return [
        {
            "run_name": "filters_small",
            "model": "cifar_conv",
            "width_multiplier": 0.75,
            "activation": "relu",
            "dropout": 0.2,
            "loss": "cross_entropy",
            "optimizer": "adamw",
        },
        {
            "run_name": "filters_base",
            "model": "cifar_conv",
            "width_multiplier": 1.0,
            "activation": "relu",
            "dropout": 0.2,
            "loss": "cross_entropy",
            "optimizer": "adamw",
        },
        {
            "run_name": "filters_wide",
            "model": "cifar_conv",
            "width_multiplier": 1.5,
            "activation": "relu",
            "dropout": 0.2,
            "loss": "cross_entropy",
            "optimizer": "adamw",
        },
        {
            "run_name": "activation_leaky_relu",
            "model": "cifar_conv",
            "width_multiplier": 1.0,
            "activation": "leaky_relu",
            "dropout": 0.2,
            "loss": "cross_entropy",
            "optimizer": "adamw",
        },
        {
            "run_name": "activation_gelu",
            "model": "cifar_conv",
            "width_multiplier": 1.0,
            "activation": "gelu",
            "dropout": 0.2,
            "loss": "cross_entropy",
            "optimizer": "adamw",
        },
        {
            "run_name": "loss_label_smoothing",
            "model": "cifar_conv",
            "width_multiplier": 1.0,
            "activation": "relu",
            "dropout": 0.2,
            "loss": "label_smoothing",
            "optimizer": "adamw",
        },
        {
            "run_name": "loss_focal",
            "model": "cifar_conv",
            "width_multiplier": 1.0,
            "activation": "relu",
            "dropout": 0.2,
            "loss": "focal",
            "optimizer": "adamw",
        },
        {
            "run_name": "optimizer_sgd",
            "model": "cifar_conv",
            "width_multiplier": 1.0,
            "activation": "relu",
            "dropout": 0.2,
            "loss": "cross_entropy",
            "optimizer": "sgd",
            "lr": 0.05,
        },
        {
            "run_name": "optimizer_adam",
            "model": "cifar_conv",
            "width_multiplier": 1.0,
            "activation": "relu",
            "dropout": 0.2,
            "loss": "cross_entropy",
            "optimizer": "adam",
        },
        {
            "run_name": "regularization_dropout_high",
            "model": "cifar_conv",
            "width_multiplier": 1.0,
            "activation": "relu",
            "dropout": 0.5,
            "loss": "label_smoothing",
            "optimizer": "adamw",
        },
        {
            "run_name": "residual_connection",
            "model": "cifar_residual",
            "width_multiplier": 1.0,
            "activation": "relu",
            "dropout": 0.1,
            "loss": "cross_entropy",
            "optimizer": "adamw",
        },
    ]


def base_namespace(args):
    return Namespace(
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_name="",
        model="cifar_conv",
        activation="relu",
        width_multiplier=1.0,
        dropout=0.2,
        batch_norm=True,
        loss="cross_entropy",
        label_smoothing=0.1,
        focal_gamma=2.0,
        optimizer="adamw",
        scheduler=args.scheduler,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        n_train=args.n_train,
        n_test=args.n_test,
        augment=args.augment,
        seed=args.seed,
        device=args.device,
    )


def run_all(args):
    rows = []
    for index, spec in enumerate(experiment_specs(), start=1):
        run_args = base_namespace(args)
        for key, value in spec.items():
            setattr(run_args, key, value)
        if "lr" not in spec:
            run_args.lr = args.lr
        run_args.seed = args.seed + index - 1
        result = run_training(run_args)
        row = spec.copy()
        row.update(result)
        rows.append(row)
    rows = sorted(rows, key=lambda item: item["best_test_accuracy"], reverse=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(rows, args.output_dir / "experiment_summary.csv")
    save_json(rows, args.output_dir / "experiment_summary.json")
    best = rows[0]
    print(f"best={best['run_name']} acc={best['best_test_accuracy']:.4f}")
    return rows


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports" / "cifar_experiments")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=-1)
    parser.add_argument("--n-test", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", type=str, choices=("none", "cosine", "step"), default="cosine")
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
