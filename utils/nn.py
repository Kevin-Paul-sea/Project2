"""
Shared neural-network utilities.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def init_weights_(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def set_random_seeds(seed_value=0, device=None):
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_name="auto"):
    if device_name is None or device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def count_parameters(model, trainable_only=False):
    total = 0
    for parameter in model.parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        total += int(np.prod(parameter.shape))
    return total


def top1_accuracy(logits, targets):
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def move_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_cpu(item) for item in value)
    return value


def save_torch_checkpoint(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    payload = move_to_cpu(data)
    try:
        torch.save(payload, str(tmp_path))
    except RuntimeError:
        torch.save(payload, str(tmp_path), _use_new_zipfile_serialization=False)
    tmp_path.replace(path)


def save_csv(rows, path, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def average(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count


def gradient_norm(model):
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        total += parameter.grad.detach().norm(2).item() ** 2
    return float(total ** 0.5)


def min_max_curves(curves):
    arrays = [np.asarray(curve, dtype=np.float64) for curve in curves if len(curve) > 0]
    if not arrays:
        return [], []
    min_length = min(len(array) for array in arrays)
    aligned = np.stack([array[:min_length] for array in arrays], axis=0)
    return aligned.min(axis=0).tolist(), aligned.max(axis=0).tolist()


def smooth_curve(values, window=1):
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(values) < window:
        return values.tolist()
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid").tolist()
