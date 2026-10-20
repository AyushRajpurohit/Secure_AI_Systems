from pathlib import Path
import json
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "attacks"))

from data import DataConfig, get_dataloaders
from evaluate import evaluate_model, print_evaluation
from fgsm import fgsm_attack
from model import MNISTCNN
from runtime import choose_device, set_global_seed


def main():
    set_global_seed(42)
    device = choose_device()

    train_loader, val_loader, test_loader = get_dataloaders(
        DataConfig(batch_size=64, split_seed=42)
    )

    model = MNISTCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    epsilon = 0.3

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Generate adversarial samples against the current model.
            adversarial = fgsm_attack(
                model,
                images,
                labels,
                criterion,
                epsilon=epsilon,
            )

            # Combine clean and adversarial samples 50/50 per source batch.
            combined_images = torch.cat([images, adversarial.detach()], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            model.train()
            logits = model(combined_images)
            loss = criterion(logits, combined_labels)
            loss.backward()
            optimizer.step()

            batch_size = combined_labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        val_result = evaluate_model(
            model,
            val_loader,
            device,
            criterion,
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={total_loss/total_examples:.6f} | "
            f"val_acc={val_result.accuracy*100:.2f}%"
        )

    checkpoint_path = ROOT / "checkpoints" / "adversarially_trained_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "defense": "FGSM adversarial training",
            "epsilon": epsilon,
            "epochs": epochs,
        },
        checkpoint_path,
    )

    test_result = evaluate_model(model, test_loader, device, criterion)
    print_evaluation(
        "Adversarially trained model on clean MNIST test set",
        test_result,
    )

    result_path = ROOT / "results" / "adversarial_training_clean_test.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "loss": test_result.loss,
                "accuracy": test_result.accuracy,
                "avg_inference_ms_per_sample":
                    test_result.avg_inference_ms_per_sample,
                "confusion_matrix":
                    test_result.confusion_matrix.tolist(),
            },
            indent=2,
        )
    )

    print(f"\nSaved defended checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
