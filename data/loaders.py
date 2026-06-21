"""
CIFAR-10 data loading utilities.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items):
        self.dataset = dataset
        self.n_items = min(int(n_items), len(dataset))

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return self.n_items


def get_transforms(train=True, augment=False):
    transform_steps = []
    if train and augment:
        transform_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return transforms.Compose(transform_steps)


def get_cifar_dataset(root="../data", train=True, augment=False, n_items=-1, download=True):
    root = Path(root).expanduser()
    dataset = datasets.CIFAR10(
        root=str(root),
        train=train,
        download=download,
        transform=get_transforms(train=train, augment=augment),
    )
    if n_items is not None and int(n_items) > 0:
        dataset = PartialDataset(dataset, n_items)
    return dataset


def get_cifar_loader(
    root="../data",
    batch_size=128,
    train=True,
    shuffle=None,
    num_workers=4,
    n_items=-1,
    augment=False,
    download=True,
    pin_memory=False,
):
    if shuffle is None:
        shuffle = train
    dataset = get_cifar_dataset(
        root=root,
        train=train,
        augment=augment,
        n_items=n_items,
        download=download,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_cifar_loaders(
    root="../data",
    batch_size=128,
    num_workers=4,
    n_train=-1,
    n_test=-1,
    augment=True,
    download=True,
    pin_memory=False,
):
    train_loader = get_cifar_loader(
        root=root,
        batch_size=batch_size,
        train=True,
        shuffle=True,
        num_workers=num_workers,
        n_items=n_train,
        augment=augment,
        download=download,
        pin_memory=pin_memory,
    )
    test_loader = get_cifar_loader(
        root=root,
        batch_size=batch_size,
        train=False,
        shuffle=False,
        num_workers=num_workers,
        n_items=n_test,
        augment=False,
        download=download,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def save_sample_grid(root="../data", output_path="sample.png", n_images=16):
    loader = get_cifar_loader(
        root=root,
        batch_size=n_images,
        train=True,
        shuffle=True,
        num_workers=0,
        augment=False,
    )
    images, labels = next(iter(loader))
    columns = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2, rows * 2))
    axes = np.array(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for index, axis in enumerate(axes[:n_images]):
        image = images[index].numpy().transpose(1, 2, 0)
        image = image * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
        image = np.clip(image, 0.0, 1.0)
        axis.imshow(image)
        axis.set_title(CIFAR10_CLASSES[int(labels[index])], fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


if __name__ == "__main__":
    loader = get_cifar_loader(num_workers=0, batch_size=8)
    images, labels = next(iter(loader))
    print(f"batch={tuple(images.shape)} labels={labels.tolist()}")
