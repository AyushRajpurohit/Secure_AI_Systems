from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "./data"
    batch_size: int = 64
    split_seed: int = 42
    train_size: int = 55_000
    val_size: int = 5_000
    num_workers: int = 0
    pin_memory: bool = False


def get_transform():
    """
    Keep model inputs directly in [0, 1].

    This is intentional: adversarial perturbation bounds such as epsilon are
    therefore expressed directly in pixel-value space and clipping to [0, 1]
    is unambiguous.
    """
    return transforms.ToTensor()


def get_datasets(config: DataConfig = DataConfig()):
    transform = get_transform()

    full_train = datasets.MNIST(
        root=config.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=config.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    if config.train_size + config.val_size != len(full_train):
        raise ValueError(
            f"train_size + val_size must equal {len(full_train)}, "
            f"got {config.train_size + config.val_size}"
        )

    generator = torch.Generator().manual_seed(config.split_seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [config.train_size, config.val_size],
        generator=generator,
    )

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(config: DataConfig = DataConfig()):
    train_dataset, val_dataset, test_dataset = get_datasets(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return train_loader, val_loader, test_loader
