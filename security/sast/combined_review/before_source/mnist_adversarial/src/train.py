from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from data import DataConfig, get_dataloaders
from evaluate import evaluate_model, print_evaluation
from model import MNISTCNN
from runtime import choose_device, set_global_seed


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 0.001
    epochs: int = 5
    random_seed: int = 42
    checkpoint_path: str = "./checkpoints/clean_model.pt"
    metadata_path: str = "./results/clean_run_metadata.json"


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size

    return running_loss / total_examples


def build_hardware_metadata(device: torch.device):
    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        metadata["cuda_version"] = torch.version.cuda
        metadata["gpu_name"] = torch.cuda.get_device_name(0)

    return metadata


def main():
    data_config = DataConfig(
        batch_size=64,
        split_seed=42,
        train_size=55_000,
        val_size=5_000,
    )
    train_config = TrainConfig(
        learning_rate=0.001,
        epochs=5,
        random_seed=42,
    )

    set_global_seed(train_config.random_seed)
    device = choose_device()
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(data_config)

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config.learning_rate,
    )

    history = []

    for epoch in range(1, train_config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_result = evaluate_model(
            model,
            val_loader,
            device,
            criterion,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_result.loss,
                "val_accuracy": val_result.accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{train_config.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_result.loss:.6f} | "
            f"val_acc={val_result.accuracy * 100:.2f}%"
        )

    checkpoint_path = Path(train_config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "MNISTCNN",
            "data_config": asdict(data_config),
            "train_config": asdict(train_config),
        },
        checkpoint_path,
    )

    test_result = evaluate_model(
        model,
        test_loader,
        device,
        criterion,
    )
    print_evaluation("Official MNIST test set", test_result)

    metadata = {
        "hardware": build_hardware_metadata(device),
        "data_config": asdict(data_config),
        "train_config": asdict(train_config),
        "history": history,
        "test": {
            "loss": test_result.loss,
            "accuracy": test_result.accuracy,
            "total_examples": test_result.total_examples,
            "total_inference_seconds": test_result.total_inference_seconds,
            "avg_inference_ms_per_sample": test_result.avg_inference_ms_per_sample,
            "confusion_matrix": test_result.confusion_matrix.tolist(),
        },
        "timing_method": (
            "Forward-pass timing only using time.perf_counter(); "
            "CUDA/MPS synchronized before and after forward pass when applicable."
        ),
    }

    metadata_path = Path(train_config.metadata_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved checkpoint to: {checkpoint_path}")
    print(f"Saved run metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
