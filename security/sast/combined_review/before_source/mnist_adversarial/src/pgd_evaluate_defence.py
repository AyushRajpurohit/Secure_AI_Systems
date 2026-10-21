"""Evaluate the PGD-trained checkpoint, or an explicitly supplied checkpoint."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_defended import main


if __name__ == "__main__":
    main(default_checkpoint=ROOT / "checkpoints" / "pgd_adversarially_trained_model.pt")
