from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "attacks"))

from data import DataConfig, get_dataloaders
from evaluate import evaluate_model, print_evaluation
from model import MNISTCNN
from pgd import pgd_attack
from runtime import choose_device, set_global_seed


def main():
    set_global_seed(42)
    device = choose_device()
    print(f"Using device: {device}")
    train_loader, val_loader, test_loader = get_dataloaders(
        DataConfig(batch_size=64, split_seed=42)
    )
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    epsilon = 0.3
    alpha = 0.01
    pgd_steps = 10

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_index, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            adversarial_images = pgd_attack(
                model, images, labels, criterion,
                epsilon=epsilon, alpha=alpha, steps=pgd_steps, random_start=True,
            )
            combined_images = torch.cat([images, adversarial_images.detach()], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            model.train()
            logits = model(combined_images)
            loss = criterion(logits, combined_labels)
            loss.backward()
            optimizer.step()

            batch_size = combined_labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            if (batch_index + 1) % 200 == 0:
                print(
                    f"Epoch {epoch}/{epochs} "
                    f"| Batch {batch_index + 1}/{len(train_loader)} "
                    f"| Loss {loss.item():.4f}"
                )

        validation_result = evaluate_model(model, val_loader, device, criterion)
        print()
        print(f"Epoch {epoch:02d}/{epochs}")
        print(f"Train loss: {total_loss / total_examples:.6f}")
        print(f"Validation loss: {validation_result.loss:.6f}")
        print(f"Validation accuracy: {validation_result.accuracy * 100:.2f}%")
        print("-" * 50)

    checkpoint_path = ROOT / "checkpoints" / "pgd_adversarially_trained_model.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "defense": "PGD adversarial training",
            "epsilon": epsilon,
            "alpha": alpha,
            "pgd_steps": pgd_steps,
            "epochs": epochs,
            "learning_rate": 0.001,
            "batch_size": 64,
        },
        checkpoint_path,
    )
    print()
    print("Saved PGD defended model to:")
    print(checkpoint_path)
    test_result = evaluate_model(model, test_loader, device, criterion)
    print_evaluation(
        "PGD adversarially trained model on clean MNIST test set", test_result,
    )


if __name__ == "__main__":
    main()
