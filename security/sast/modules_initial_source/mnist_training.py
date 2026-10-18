"""Shared MNIST model, training, and evaluation utilities.

Before calling train_experiment, explicitly assign:
    CONFIG
    device
    validation_loader

Protected training callers must verify data before training.
"""

import random
import time
import numpy as np
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 × 28 → 14 × 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 × 14 → 7 × 7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def make_dataset(images, labels):
    image_tensor = (
        torch.from_numpy(np.ascontiguousarray(images))
        .unsqueeze(1)
        .to(torch.float32)
        .div(255.0)
    )

    label_tensor = torch.from_numpy(
        np.ascontiguousarray(labels)
    ).to(torch.long)

    return TensorDataset(image_tensor, label_tensor)


def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def evaluate_model(model, loader, device):
    model.eval()

    loss_function = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    # Rows = true classes; columns = predicted classes.
    confusion = torch.zeros((10, 10), dtype=torch.long)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        total_loss += loss_function(logits, labels).item()
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

        pair_indices = (
            10 * labels.cpu() + predictions.cpu()
        )
        confusion += torch.bincount(
            pair_indices,
            minlength=100,
        ).reshape(10, 10)

    source_count = confusion[1].sum().item()

    return {
        "accuracy": total_correct / total_examples,
        "loss": total_loss / total_examples,
        "class_1_recall": (
            confusion[1, 1].item() / source_count
            if source_count else None
        ),
        "targeted_1_to_7_error_rate": (
            confusion[1, 7].item() / source_count
            if source_count else None
        ),
        "confusion_matrix": confusion.tolist(),
        "examples": total_examples,
    }


def train_experiment(training_dataset, seed):
    set_training_seed(seed)

    # Fresh initialization for each experiment.
    model = MNISTCNN().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
    )
    loss_function = nn.CrossEntropyLoss()

    shuffle_generator = torch.Generator().manual_seed(seed)

    training_loader = DataLoader(
        training_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        generator=shuffle_generator,
        num_workers=0,
    )

    best_validation_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []

    started = time.perf_counter()

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()

        training_loss_sum = 0.0
        training_correct = 0
        training_count = 0

        for images, labels in training_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()

            training_loss_sum += loss.item() * labels.size(0)
            training_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            training_count += labels.size(0)

        validation_metrics = evaluate_model(
            model, validation_loader, device
        )

        history.append({
            "epoch": epoch,
            "training_loss": training_loss_sum / training_count,
            "training_accuracy": training_correct / training_count,
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        })

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch

            # Keep the selected weights independent of future updates.
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        print(
            f"Epoch {epoch}/{CONFIG['epochs']} | "
            f"train accuracy: {training_correct / training_count:.4f} | "
            f"validation accuracy: {validation_metrics['accuracy']:.4f} | "
            f"validation loss: {validation_metrics['loss']:.4f}",
            flush=True,
        )

    elapsed_seconds = time.perf_counter() - started

    model.load_state_dict(best_state)
    model.eval()

    return model, history, best_epoch, elapsed_seconds

@torch.inference_mode()
def benchmark_inference(
    model,
    dataset,
    device,
    batch_size=256,
    warmup_runs=10,
    measured_runs=50,
):
    model.eval()

    images = dataset.tensors[0][:batch_size].to(device)
    actual_batch_size = len(images)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup_runs):
        model(images)

    synchronize()
    timings_ms = []

    for _ in range(measured_runs):
        synchronize()
        started = time.perf_counter()

        model(images)

        synchronize()
        timings_ms.append(
            (time.perf_counter() - started) * 1000
        )

    median_batch_ms = float(np.median(timings_ms))

    return {
        "scope": "forward pass; inputs already on device",
        "batch_size": actual_batch_size,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "median_batch_ms": median_batch_ms,
        "p95_batch_ms": float(np.percentile(timings_ms, 95)),
        "amortized_ms_per_image": (
            median_batch_ms / actual_batch_size
        ),
        "device": str(device),
    }
