"""
Train CIFAR-10 classifiers and save reproducible experiment artifacts.
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import torch

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data.loaders import get_cifar_loaders
from models.vgg import MODEL_NAMES, build_model
from utils.losses import build_criterion
from utils.nn import (
    AverageMeter,
    count_parameters,
    ensure_dir,
    get_device,
    save_csv,
    save_json,
    save_torch_checkpoint,
    set_random_seeds,
    top1_accuracy,
)


def build_optimizer(name, parameters, lr, weight_decay=0.0, momentum=0.9):
    name = str(name).lower()
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(name, optimizer, epochs):
    name = str(name).lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.2)
    raise ValueError(f"unknown scheduler: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    step_losses = []
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1_accuracy(logits.detach(), targets), batch_size)
        step_losses.append(float(loss.item()))
    return {
        "loss": loss_meter.average,
        "accuracy": acc_meter.average,
        "step_losses": step_losses,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(top1_accuracy(logits, targets), batch_size)
    return {
        "loss": loss_meter.average,
        "accuracy": acc_meter.average,
    }


def plot_history(rows, output_path):
    if not rows:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in rows], label="train")
    axes[0].plot(epochs, [row["test_loss"] for row in rows], label="test")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [row["train_accuracy"] for row in rows], label="train")
    axes[1].plot(epochs, [row["test_accuracy"] for row in rows], label="test")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def serializable_config(args):
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def make_run_name(args):
    if args.run_name:
        return args.run_name
    width = str(args.width_multiplier).replace(".", "p")
    drop = str(args.dropout).replace(".", "p")
    return f"{args.model}_{args.optimizer}_{args.loss}_{args.activation}_w{width}_d{drop}"


def save_checkpoint(path, model, optimizer, epoch, best_accuracy, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_torch_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "config": config,
        },
        path,
    )


def run_training(args):
    device = get_device(args.device)
    set_random_seeds(args.seed, device=device)
    pin_memory = device.type == "cuda"
    output_root = ensure_dir(args.output_dir)
    run_dir = ensure_dir(output_root / make_run_name(args))
    config = serializable_config(args)
    config["run_dir"] = str(run_dir)
    save_json(config, run_dir / "config.json")
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
    model = build_model(
        name=args.model,
        activation=args.activation,
        width_multiplier=args.width_multiplier,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
    ).to(device)
    criterion = build_criterion(
        name=args.loss,
        label_smoothing=args.label_smoothing,
        focal_gamma=args.focal_gamma,
    )
    optimizer = build_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)
    rows = []
    all_step_losses = []
    best_accuracy = 0.0
    best_epoch = 0
    print(f"device={device} parameters={count_parameters(model)}")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_stats = evaluate(model, test_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()
        elapsed = time.time() - start_time
        all_step_losses.extend(train_stats["step_losses"])
        row = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "test_loss": test_stats["loss"],
            "test_accuracy": test_stats["accuracy"],
            "test_error": 1.0 - test_stats["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        rows.append(row)
        if test_stats["accuracy"] > best_accuracy:
            best_accuracy = test_stats["accuracy"]
            best_epoch = epoch
            save_checkpoint(run_dir / "best_model.pt", model, optimizer, epoch, best_accuracy, config)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} test_acc={row['test_accuracy']:.4f}"
        )
    save_checkpoint(run_dir / "last_model.pt", model, optimizer, args.epochs, best_accuracy, config)
    save_csv(rows, run_dir / "metrics.csv")
    save_json(
        {
            "best_epoch": best_epoch,
            "best_test_accuracy": best_accuracy,
            "best_test_error": 1.0 - best_accuracy,
            "parameters": count_parameters(model),
            "step_losses": all_step_losses,
        },
        run_dir / "summary.json",
    )
    plot_history(rows, run_dir / "training_curves.png")
    return {
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_test_accuracy": best_accuracy,
        "best_test_error": 1.0 - best_accuracy,
        "parameters": count_parameters(model),
    }


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports" / "cifar")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--model", type=str, choices=MODEL_NAMES, default="cifar_conv")
    parser.add_argument("--activation", type=str, choices=("relu", "leaky_relu", "elu", "gelu", "silu"), default="relu")
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-norm", action="store_true", default=True)
    parser.add_argument("--no-batch-norm", dest="batch_norm", action="store_false")
    parser.add_argument("--loss", type=str, choices=("cross_entropy", "label_smoothing", "focal"), default="cross_entropy")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--optimizer", type=str, choices=("sgd", "adam", "adamw", "rmsprop"), default="adamw")
    parser.add_argument("--scheduler", type=str, choices=("none", "cosine", "step"), default="cosine")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n-train", type=int, default=-1)
    parser.add_argument("--n-test", type=int, default=-1)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    result = run_training(args)
    print(f"best_test_acc={result['best_test_accuracy']:.4f} run_dir={result['run_dir']}")


if __name__ == "__main__":
    main()
