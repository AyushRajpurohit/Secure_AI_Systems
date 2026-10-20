from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from runtime import synchronize_device


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    confusion_matrix: np.ndarray
    total_examples: int
    total_inference_seconds: float
    avg_inference_ms_per_sample: float


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> EvaluationResult:
    """
    Evaluate a model consistently across clean, poisoned, and defended models.

    Timing notes:
    - Measures model forward-pass time only.
    - Excludes dataloader / disk-loading time.
    - Synchronizes CUDA/MPS before/after the forward pass when using a GPU.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_inference_seconds = 0.0

    all_predictions = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        synchronize_device(device)

        start = time.perf_counter()
        logits = model(images)

        synchronize_device(device)

        elapsed = time.perf_counter() - start
        total_inference_seconds += elapsed

        loss = criterion(logits, labels)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += batch_size

        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    if total_examples == 0:
        raise ValueError("Evaluation loader is empty")

    cm = confusion_matrix(
        all_targets,
        all_predictions,
        labels=list(range(10)),
    )

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    avg_ms = (total_inference_seconds / total_examples) * 1000.0

    return EvaluationResult(
        loss=avg_loss,
        accuracy=accuracy,
        confusion_matrix=cm,
        total_examples=total_examples,
        total_inference_seconds=total_inference_seconds,
        avg_inference_ms_per_sample=avg_ms,
    )


def print_evaluation(name: str, result: EvaluationResult) -> None:
    print(f"\n=== {name} ===")
    print(f"Loss: {result.loss:.6f}")
    print(f"Accuracy: {result.accuracy * 100:.2f}%")
    print(f"Examples: {result.total_examples}")
    print(f"Forward-pass time: {result.total_inference_seconds:.6f} s")
    print(
        "Average inference time: " f"{result.avg_inference_ms_per_sample:.6f} ms/sample"
    )
    print("Confusion matrix:")
    print(result.confusion_matrix)
