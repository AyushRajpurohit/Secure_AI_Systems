from pathlib import Path
import argparse
import sys

import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "attacks"))

from data import DataConfig, get_dataloaders
from evaluate import evaluate_model
from evaluate_attacks import evaluate_suite, load_model
from runtime import choose_device, set_global_seed


def evaluate_clean(model, dataloader, device):
    result = evaluate_model(model, dataloader, device, nn.CrossEntropyLoss())
    print("\n Defended model: Clean test set ")
    print(f"Clean accuracy: {result.accuracy * 100:.2f}%")
    print(f"Clean loss: {result.loss:.6f}")
    return result


def main(argv=None, *, default_checkpoint=None):
    if default_checkpoint is None:
        default_checkpoint = ROOT / "checkpoints" / "adversarially_trained_model.pt"
    parser = argparse.ArgumentParser(description="Evaluate a defended MNIST checkpoint")
    parser.add_argument(
        "checkpoint", type=Path, nargs="?",
        default=default_checkpoint,
    )
    parser.add_argument("--print-sanity", action="store_true")
    args = parser.parse_args(argv)
    set_global_seed(42)
    device = choose_device()
    print(f"Using device: {device}")
    _, _, test_loader = get_dataloaders(DataConfig(batch_size=64, split_seed=42))
    model = load_model(args.checkpoint, device)
    evaluate_clean(model, test_loader, device)
    evaluate_suite(
        model, test_loader, device, defended=True, print_sanity=args.print_sanity,
    )


if __name__ == "__main__":
    main()
