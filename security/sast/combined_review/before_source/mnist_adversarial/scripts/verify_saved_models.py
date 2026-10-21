"""Test and evaluate the saved checkpoints without retraining them."""

import argparse
import contextlib
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (
    "clean_model.pt",
    "adversarially_trained_model.pt",
    "pgd_adversarially_trained_model.pt",
)


class Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_report(path, report):
    path.write_text(
        json.dumps(report, indent=2, default=lambda value: value.tolist(), allow_nan=False)
        + "\n"
    )


def verify(report, output):
    import torch

    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "attacks")]
    from data import DataConfig, get_dataloaders
    from evaluate import evaluate_model, print_evaluation
    from evaluate_attacks import (
        FGSM_EPSILONS, PGD_ALPHA, PGD_EPSILONS, PGD_STEPS, evaluate_suite, load_model,
    )
    from runtime import choose_device, set_global_seed
    from train import build_hardware_metadata

    device = choose_device()
    report["runtime"] = {
        **build_hardware_metadata(device),
        "torchvision": version("torchvision"),
        "numpy": version("numpy"),
        "scikit_learn": version("scikit-learn"),
        "torch_threads": torch.get_num_threads(),
    }
    paths = [ROOT / "checkpoints" / name for name in CHECKPOINTS]
    report["checkpoint_sha256"] = {path.name: sha256(path) for path in paths}
    source_paths = sorted(
        path for folder in ("src", "attacks", "defenses", "tests", "scripts")
        for path in (ROOT / folder).rglob("*.py")
    ) + [ROOT / "evaluate_defended.py", ROOT / "requirements.txt"]
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path) for path in source_paths
    }
    save_report(output, report)

    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    tests = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    print(tests.stdout, end="")
    print(tests.stderr, end="")
    report["tests"] = {"command": command, "returncode": tests.returncode,
                       "output": tests.stdout + tests.stderr}
    save_report(output, report)
    if tests.returncode:
        raise RuntimeError("Correctness tests failed; evaluation was not started")

    set_global_seed(42)
    config = DataConfig(data_dir=str(ROOT / "data"), batch_size=64, split_seed=42)
    report["data_config"] = asdict(config)
    report["evaluation_config"] = {
        "seed": 42,
        "fgsm_epsilon": list(FGSM_EPSILONS),
        "baseline_pgd_epsilon": list(PGD_EPSILONS),
        "defense_pgd_epsilon": [0.3],
        "pgd_alpha": PGD_ALPHA,
        "pgd_steps": PGD_STEPS,
        "pgd_random_start": True,
        "pgd_trajectories": 1,
    }
    _, _, test_loader = get_dataloaders(config)
    if len(test_loader.dataset) != 10_000:
        raise ValueError("Expected the official 10,000-image test set")

    report["models"] = {}
    for path in paths:
        print(f"\nCheckpoint: {path.name}", flush=True)
        model = load_model(path, device)
        clean = evaluate_model(model, test_loader, device)
        print_evaluation("Clean official MNIST test set", clean)
        model_result = {"clean": asdict(clean)}
        report["models"][path.name] = model_result
        save_report(output, report)
        model_result["attacks"] = evaluate_suite(
            model, test_loader, device,
            defended=path.name != "clean_model.pt", print_sanity=True,
        )
        save_report(output, report)

    after = {path.name: sha256(path) for path in paths}
    report["checkpoint_files_unchanged"] = after == report["checkpoint_sha256"]
    report["source_files_unchanged"] = all(
        sha256(ROOT / path) == digest for path, digest in report["source_sha256"].items()
    )
    if not report["checkpoint_files_unchanged"] or not report["source_files_unchanged"]:
        raise RuntimeError("Checkpoint or source files changed during evaluation")
    report["status"] = "complete"
    save_report(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "verified_run")
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error("Run without -O so the attack assertions stay enabled")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "metrics.json"
    report = {"status": "incomplete", "python": platform.python_version(),
              "python_executable": sys.executable}
    save_report(output, report)
    with (args.output_dir / "evaluation.txt").open("w", buffering=1) as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                verify(report, output)
            except Exception as exc:
                report["error"] = f"{type(exc).__name__}: {exc}"
                save_report(output, report)
                traceback.print_exc()
                raise SystemExit(1)
    print(f"Saved verified results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
