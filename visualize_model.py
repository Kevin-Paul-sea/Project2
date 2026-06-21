"""
Create visualizations for trained CIFAR-10 models.
"""

import argparse
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from data.loaders import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, get_cifar_loader
from models.vgg import build_model
from utils.nn import get_device


def first_conv_layer(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    raise ValueError("model has no Conv2d layer")


def normalize_image(image):
    image = image - image.min()
    maximum = image.max()
    if maximum > 0:
        image = image / maximum
    return image


def save_filter_grid(model, output_path, max_filters=64):
    conv = first_conv_layer(model)
    weights = conv.weight.detach().cpu().numpy()
    n_filters = min(max_filters, weights.shape[0])
    columns = int(np.ceil(np.sqrt(n_filters)))
    rows = int(np.ceil(n_filters / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.3, rows * 1.3))
    axes = np.array(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for index in range(n_filters):
        filt = weights[index]
        if filt.shape[0] == 3:
            image = filt.transpose(1, 2, 0)
        else:
            image = filt.mean(axis=0)
        axes[index].imshow(normalize_image(image))
    fig.tight_layout(pad=0.1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def denormalize_batch(images):
    mean = torch.tensor(CIFAR10_MEAN, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)
    return torch.clamp(images * std + mean, 0.0, 1.0)


@torch.no_grad()
def save_prediction_grid(model, loader, device, output_path, n_images=16):
    model.eval()
    images, labels = next(iter(loader))
    images = images[:n_images].to(device)
    labels = labels[:n_images].to(device)
    logits = model(images)
    predictions = logits.argmax(dim=1)
    images = denormalize_batch(images).cpu().numpy()
    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    columns = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for index in range(min(n_images, len(images))):
        image = images[index].transpose(1, 2, 0)
        truth = CIFAR10_CLASSES[int(labels[index])]
        pred = CIFAR10_CLASSES[int(predictions[index])]
        axes[index].imshow(image)
        axes[index].set_title(f"p:{pred}\nt:{truth}", fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_model_from_checkpoint(checkpoint_path, args, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model_name = args.model or config.get("model", "cifar_conv")
    activation = args.activation or config.get("activation", "relu")
    width_multiplier = args.width_multiplier or config.get("width_multiplier", 1.0)
    dropout = args.dropout if args.dropout is not None else config.get("dropout", 0.2)
    batch_norm = config.get("batch_norm", True)
    model = build_model(
        model_name,
        activation=activation,
        width_multiplier=width_multiplier,
        dropout=dropout,
        batch_norm=batch_norm,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports" / "visualizations")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--activation", type=str, default="")
    parser.add_argument("--width-multiplier", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main():
    args = build_arg_parser().parse_args()
    device = get_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_from_checkpoint(args.checkpoint, args, device)
    loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=True,
        num_workers=args.workers,
        download=True,
    )
    save_filter_grid(model, args.output_dir / "first_layer_filters.png")
    save_prediction_grid(model, loader, device, args.output_dir / "sample_predictions.png", args.n_images)
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
