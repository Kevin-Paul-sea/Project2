"""
Compare VGG-A and VGG-A with Batch Normalization on CIFAR-10.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data.loaders import get_cifar_loaders
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters
from train_cifar import build_optimizer, evaluate
from utils.nn import (
    get_device,
    gradient_norm,
    min_max_curves,
    save_csv,
    save_json,
    save_torch_checkpoint,
    set_random_seeds,
    smooth_curve,
)


def parse_learning_rates(text):
    if isinstance(text, (list, tuple)):
        return [float(item) for item in text]
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def format_lr(lr):
    return f"{lr:.0e}".replace("-", "m").replace("+", "")


def classifier_gradient_vector(model):
    last_layer = model.classifier[-1]
    if last_layer.weight.grad is None:
        return None
    return last_layer.weight.grad.detach().flatten().cpu()


def train_variant(variant, lr, args, device, train_loader, test_loader):
    if variant == "vgg_a":
        model = VGG_A().to(device)
    elif variant == "vgg_a_bn":
        model = VGG_A_BatchNorm().to(device)
    else:
        raise ValueError(f"unknown variant: {variant}")
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        args.optimizer,
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    step_losses = []
    gradient_norms = []
    gradient_cosines = []
    gradient_differences = []
    epoch_rows = []
    best_accuracy = 0.0
    previous_gradient = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            current_gradient = classifier_gradient_vector(model)
            if current_gradient is None or previous_gradient is None:
                gradient_cosines.append(None)
                gradient_differences.append(None)
            else:
                cosine = F.cosine_similarity(current_gradient, previous_gradient, dim=0).item()
                difference = torch.norm(current_gradient - previous_gradient, p=2).item()
                gradient_cosines.append(float(cosine))
                gradient_differences.append(float(difference))
            if current_gradient is not None:
                previous_gradient = current_gradient
            gradient_norms.append(gradient_norm(model))
            step_losses.append(float(loss.item()))
            optimizer.step()
        test_stats = evaluate(model, test_loader, criterion, device)
        best_accuracy = max(best_accuracy, test_stats["accuracy"])
        row = {
            "variant": variant,
            "learning_rate": lr,
            "epoch": epoch,
            "test_loss": test_stats["loss"],
            "test_accuracy": test_stats["accuracy"],
            "test_error": 1.0 - test_stats["accuracy"],
            "seconds": time.time() - start_time,
        }
        epoch_rows.append(row)
        print(f"{variant} lr={lr:g} epoch={epoch} test_acc={test_stats['accuracy']:.4f}")
    checkpoint_path = args.output_dir / f"{variant}_lr_{format_lr(lr)}.pt"
    save_torch_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "variant": variant,
            "learning_rate": lr,
            "epochs": args.epochs,
            "best_accuracy": best_accuracy,
            "parameters": get_number_of_parameters(model),
        },
        checkpoint_path,
    )
    return {
        "variant": variant,
        "learning_rate": lr,
        "parameters": get_number_of_parameters(model),
        "best_accuracy": best_accuracy,
        "best_error": 1.0 - best_accuracy,
        "step_losses": step_losses,
        "gradient_norms": gradient_norms,
        "gradient_cosines": gradient_cosines,
        "gradient_differences": gradient_differences,
        "epoch_rows": epoch_rows,
        "checkpoint": str(checkpoint_path),
    }


def group_histories(histories, variant):
    return [history for history in histories if history["variant"] == variant]


def plot_loss_landscape(histories, output_path, smooth_window=1):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    settings = [
        ("vgg_a", "VGG-A", "tab:red", 0.22),
        ("vgg_a_bn", "VGG-A+BN", "tab:blue", 0.22),
    ]
    for variant, label, color, alpha in settings:
        variant_histories = group_histories(histories, variant)
        min_curve, max_curve = min_max_curves([history["step_losses"] for history in variant_histories])
        min_curve = smooth_curve(min_curve, smooth_window)
        max_curve = smooth_curve(max_curve, smooth_window)
        if not min_curve or not max_curve:
            continue
        x_values = np.arange(len(min_curve))
        axis.plot(x_values, min_curve, color=color, linewidth=1.2, label=f"{label} min")
        axis.plot(x_values, max_curve, color=color, linewidth=1.2, linestyle="--", label=f"{label} max")
        axis.fill_between(x_values, min_curve, max_curve, color=color, alpha=alpha)
    axis.set_xlabel("training step")
    axis.set_ylabel("training loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_gradient_statistics(histories, output_path, smooth_window=1):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    settings = [
        ("vgg_a", "VGG-A", "tab:red"),
        ("vgg_a_bn", "VGG-A+BN", "tab:blue"),
    ]
    for variant, label, color in settings:
        variant_histories = group_histories(histories, variant)
        min_norm, max_norm = min_max_curves([history["gradient_norms"] for history in variant_histories])
        min_diff, max_diff = min_max_curves(
            [
                [value for value in history["gradient_differences"] if value is not None]
                for history in variant_histories
            ]
        )
        min_norm = smooth_curve(min_norm, smooth_window)
        max_norm = smooth_curve(max_norm, smooth_window)
        min_diff = smooth_curve(min_diff, smooth_window)
        max_diff = smooth_curve(max_diff, smooth_window)
        if min_norm and max_norm:
            x_values = np.arange(len(min_norm))
            axes[0].plot(x_values, max_norm, color=color, label=f"{label} max")
            axes[0].fill_between(x_values, min_norm, max_norm, color=color, alpha=0.2)
        if min_diff and max_diff:
            x_values = np.arange(len(min_diff))
            axes[1].plot(x_values, max_diff, color=color, label=f"{label} max")
            axes[1].fill_between(x_values, min_diff, max_diff, color=color, alpha=0.2)
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("gradient norm")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel("successive gradient difference")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_accuracy(histories, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 5))
    for history in histories:
        rows = history["epoch_rows"]
        epochs = [row["epoch"] for row in rows]
        accuracies = [row["test_accuracy"] for row in rows]
        label = f"{history['variant']} lr={history['learning_rate']:g}"
        axis.plot(epochs, accuracies, marker="o", linewidth=1.2, label=label)
    axis.set_xlabel("epoch")
    axis.set_ylabel("test accuracy")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_bn_experiment(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    set_random_seeds(args.seed, device=device)
    pin_memory = device.type == "cuda"
    train_loader, test_loader = get_cifar_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.workers,
        n_train=args.n_train,
        n_test=args.n_test,
        augment=args.augment,
        download=True,
        pin_memory=pin_memory,
    )
    learning_rates = parse_learning_rates(args.learning_rates)
    histories = []
    summary_rows = []
    for variant in ("vgg_a", "vgg_a_bn"):
        for lr in learning_rates:
            history = train_variant(variant, lr, args, device, train_loader, test_loader)
            histories.append(history)
            history_path = args.output_dir / f"{variant}_lr_{format_lr(lr)}_history.json"
            save_json(history, history_path)
            summary_rows.append(
                {
                    "variant": variant,
                    "learning_rate": lr,
                    "parameters": history["parameters"],
                    "best_accuracy": history["best_accuracy"],
                    "best_error": history["best_error"],
                    "checkpoint": history["checkpoint"],
                }
            )
    save_json(
        {
            "learning_rates": learning_rates,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "histories": histories,
        },
        args.output_dir / "bn_histories.json",
    )
    save_csv(summary_rows, args.output_dir / "bn_summary.csv")
    epoch_rows = []
    for history in histories:
        epoch_rows.extend(history["epoch_rows"])
    save_csv(epoch_rows, args.output_dir / "bn_epoch_metrics.csv")
    plot_loss_landscape(histories, args.output_dir / "loss_landscape_compare.png", args.smooth_window)
    plot_gradient_statistics(histories, args.output_dir / "gradient_statistics.png", args.smooth_window)
    plot_accuracy(histories, args.output_dir / "bn_accuracy.png")
    best = max(summary_rows, key=lambda row: row["best_accuracy"])
    print(f"best={best['variant']} lr={best['learning_rate']:g} acc={best['best_accuracy']:.4f}")
    return histories


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports" / "bn")
    parser.add_argument("--learning-rates", type=str, default="0.001,0.002,0.0001,0.0005")
    parser.add_argument("--optimizer", type=str, choices=("sgd", "adam", "adamw", "rmsprop"), default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=-1)
    parser.add_argument("--n-test", type=int, default=-1)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smooth-window", type=int, default=1)
    return parser


def main():
    args = build_arg_parser().parse_args()
    run_bn_experiment(args)


if __name__ == "__main__":
    main()
