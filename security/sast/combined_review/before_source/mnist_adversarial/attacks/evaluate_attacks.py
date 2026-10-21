from pathlib import Path
import argparse
import sys
import time

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "attacks"))

from data import DataConfig, get_dataloaders
from model import MNISTCNN
from fgsm import fgsm_attack
from pgd import pgd_attack
from runtime import choose_device, set_global_seed, synchronize_device
from sanity import check_adversarial

FGSM_EPSILONS = (0.1, 0.2, 0.3)
PGD_EPSILONS = (0.05, 0.10, 0.20, 0.30)
PGD_ALPHA = 0.01
PGD_STEPS = 40


def load_model(checkpoint_path, device):
    model = MNISTCNN().to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_attack(
    model,
    dataloader,
    device,
    attack_name,
    attack_fn,
    *,
    epsilon=None,
    print_sanity=False,
    defended=False,
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    clean_correct = 0
    adv_correct = 0
    originally_correct = 0
    successful_attacks = 0
    total_adv_loss = 0.0

    all_adv_predictions = []
    all_targets = []
    max_linf, pixel_min, pixel_max = 0.0, 1.0, 0.0

    synchronize_device(device)
    start = time.perf_counter()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_logits = model(images)
            clean_pred = clean_logits.argmax(dim=1)

        adversarial = attack_fn(
            model,
            images,
            labels,
            criterion,
        )
        if epsilon is not None:
            batch_linf, batch_min, batch_max = check_adversarial(
                images, adversarial, epsilon
            )
            max_linf = max(max_linf, batch_linf)
            pixel_min = min(pixel_min, batch_min)
            pixel_max = max(pixel_max, batch_max)

        with torch.no_grad():
            adv_logits = model(adversarial)
            adv_pred = adv_logits.argmax(dim=1)
            adv_loss = criterion(adv_logits, labels)

        batch_size = labels.size(0)
        total += batch_size
        clean_correct += (clean_pred == labels).sum().item()
        adv_correct += (adv_pred == labels).sum().item()
        total_adv_loss += adv_loss.item() * batch_size

        clean_correct_mask = clean_pred == labels
        originally_correct += clean_correct_mask.sum().item()
        successful_attacks += (
            clean_correct_mask & (adv_pred != labels)
        ).sum().item()

        all_adv_predictions.extend(adv_pred.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    synchronize_device(device)
    elapsed = time.perf_counter() - start

    attack_success_rate = (
        successful_attacks / originally_correct
        if originally_correct > 0
        else 0.0
    )

    if total == 0:
        raise ValueError("Evaluation loader is empty")

    cm = confusion_matrix(
        all_targets,
        all_adv_predictions,
        labels=list(range(10)),
    )

    print(f"\n {attack_name} " if defended else f"\n=== {attack_name} ===")
    print(f"Clean accuracy: {clean_correct / total * 100:.2f}%")
    print(f"Adversarial accuracy: {adv_correct / total * 100:.2f}%")
    print(f"Adversarial loss: {total_adv_loss / total:.6f}")
    print(f"Attack success rate: {attack_success_rate * 100:.2f}%")
    print(f"Total attack+evaluation time: {elapsed:.3f} s")
    print("Adversarial confusion matrix:")
    print(cm)
    if print_sanity and epsilon is not None:
        print(
            f"Attack sanity: max_linf={max_linf:.8f}, "
            f"pixel_min={pixel_min:.8f}, pixel_max={pixel_max:.8f}"
        )
    return {
        "clean_accuracy": clean_correct / total,
        "adversarial_accuracy": adv_correct / total,
        "attack_success_rate": attack_success_rate,
        "adversarial_loss": total_adv_loss / total,
        "total_examples": total,
        "confusion_matrix": cm,
        "max_linf": max_linf if epsilon is not None else None,
    }


def evaluate_suite(model, test_loader, device, *, defended=False, print_sanity=False):
    prefix = "Defended model - " if defended else ""
    results = {}
    for epsilon in FGSM_EPSILONS:
        set_global_seed(42)
        name = f"{prefix}FGSM epsilon={epsilon}"
        results[name] = evaluate_attack(
            model,
            test_loader,
            device,
            attack_name=name,
            attack_fn=lambda m, x, y, c, eps=epsilon: fgsm_attack(
                m, x, y, c, epsilon=eps
            ),
            epsilon=epsilon,
            print_sanity=print_sanity,
            defended=defended,
        )

    for epsilon in ((0.3,) if defended else PGD_EPSILONS):
        # Reset before each attack for comparable random starts across models.
        set_global_seed(42)
        name = f"{prefix}PGD epsilon={epsilon} alpha={PGD_ALPHA} steps={PGD_STEPS}"
        results[name] = evaluate_attack(
            model,
            test_loader,
            device,
            attack_name=name,
            attack_fn=lambda m, x, y, c, eps=epsilon: pgd_attack(
                m, x, y, c, epsilon=eps, alpha=PGD_ALPHA,
                steps=PGD_STEPS, random_start=True,
            ),
            epsilon=epsilon,
            print_sanity=print_sanity,
            defended=defended,
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline MNIST attacks")
    parser.add_argument("--print-sanity", action="store_true")
    args = parser.parse_args()
    set_global_seed(42)
    device = choose_device()
    _, _, test_loader = get_dataloaders(DataConfig(batch_size=64, split_seed=42))
    model = load_model(ROOT / "checkpoints" / "clean_model.pt", device)
    evaluate_suite(model, test_loader, device, print_sanity=args.print_sanity)


if __name__ == "__main__":
    main()
