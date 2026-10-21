#!/usr/bin/env bash
# Train all models and save the full run without deleting earlier results.
set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$PROJECT_ROOT"

if [[ "${1:-}" == "--worker" ]]; then
    RUN_DIR="$2"
    PYTHON_EXE="$3"

    run_step() {
        local label="$1"
        shift
        printf '\n=== %s ===\n' "$label"
        "$@" 2>&1 | tee "$RUN_DIR/$label.txt"
    }

    for required_file in scripts/verify_saved_models.py src/train.py defenses/adversarial_training.py defenses/pgd_adversarial_training.py; do
        if [[ ! -f "$required_file" ]]; then
            printf 'Missing project file: %s\n' "$required_file" >&2
            printf 'Copy the updated project files, including the scripts/ directory, to this machine and rerun.\n' >&2
            exit 1
        fi
    done

    run_step preflight "$PYTHON_EXE" -u - "$RUN_DIR" <<'PY'
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from dataclasses import asdict
import numpy as np
import sklearn
import torch
import torchvision

root = Path.cwd()
output = Path(sys.argv[1])
sys.path[:0] = [str(root / 'src'), str(root / 'attacks')]
from data import DataConfig, get_datasets
from train import TrainConfig, build_hardware_metadata
from runtime import choose_device
from evaluate_attacks import FGSM_EPSILONS, PGD_EPSILONS, PGD_ALPHA, PGD_STEPS
from defenses import adversarial_training, pgd_adversarial_training

if sys.flags.optimize:
    raise RuntimeError('Run without Python optimization so assertions stay enabled')
config = DataConfig()
record = {
    'python_executable': sys.executable,
    'hardware': build_hardware_metadata(choose_device()),
    'torchvision': torchvision.__version__,
    'numpy': np.__version__,
    'scikit_learn': sklearn.__version__,
    'data_config': asdict(config),
    'baseline_training': asdict(TrainConfig()),
    'evaluation': {'fgsm_epsilon': list(FGSM_EPSILONS),
                   'baseline_pgd_epsilon': list(PGD_EPSILONS),
                   'defense_pgd_epsilon': [0.3], 'pgd_alpha': PGD_ALPHA,
                   'pgd_steps': PGD_STEPS, 'random_start': True, 'trajectories': 1},
}
(output / 'run_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
(output / 'environment.txt').write_text('\n'.join(packages) + '\n')
print(json.dumps(record, indent=2), flush=True)
train, val, test = get_datasets(config)
np.savez_compressed(output / 'split_indices.npz',
                    train_indices=np.asarray(train.indices),
                    val_indices=np.asarray(val.indices),
                    test_indices=np.arange(len(test)))
print('Recorded actual generated split:', len(train), len(val), len(test))
PY

    # Preflight must succeed before any model or result is overwritten.
    mkdir -p "$RUN_DIR/before"
    if [[ -d checkpoints ]]; then
        cp -a checkpoints "$RUN_DIR/before/"
    fi
    for file in results/clean_run_metadata.json results/adversarial_training_clean_test.json; do
        if [[ -f "$file" ]]; then
            cp "$file" "$RUN_DIR/before/"
        fi
    done

    source_files=(src/*.py attacks/*.py defenses/*.py tests/*.py scripts/*.py run_all.sh evaluate_defended.py requirements.txt)
    # Standalone and merged repository layouts keep ignore rules in different places.
    if [[ -f .gitignore ]]; then
        ignore_file=.gitignore
    elif [[ -f ../.gitignore ]]; then
        ignore_file=../.gitignore
    else
        printf 'Missing repository .gitignore\n' >&2
        exit 1
    fi
    source_files+=("$ignore_file")
    sha256sum "${source_files[@]}" > "$RUN_DIR/source_sha256.txt"
    for file in "${source_files[@]}"; do
        snapshot_file="$file"
        if [[ "$file" == ../.gitignore ]]; then
            snapshot_file="repository_root/.gitignore"
        fi
        mkdir -p "$RUN_DIR/source/$(dirname "$snapshot_file")"
        cp "$file" "$RUN_DIR/source/$snapshot_file"
    done
    sha256sum data/MNIST/raw/* > "$RUN_DIR/dataset_sha256.txt"
    git status --short > "$RUN_DIR/git_status_before.txt" 2>&1 || true

    run_step baseline_training "$PYTHON_EXE" -u src/train.py
    cp results/clean_run_metadata.json "$RUN_DIR/clean_run_metadata.json"

    run_step fgsm_training "$PYTHON_EXE" -u defenses/adversarial_training.py
    cp results/adversarial_training_clean_test.json "$RUN_DIR/adversarial_training_clean_test.json"

    run_step pgd_training "$PYTHON_EXE" -u defenses/pgd_adversarial_training.py

    # The verifier runs all 14 tests, then all clean and attack evaluations.
    run_step verification "$PYTHON_EXE" -u scripts/verify_saved_models.py --output-dir "$RUN_DIR/evaluation"
    run_step source_integrity sha256sum -c "$RUN_DIR/source_sha256.txt"
    run_step dataset_integrity sha256sum -c "$RUN_DIR/dataset_sha256.txt"

    sha256sum checkpoints/*.pt > checkpoints/SHA256SUMS
    cp checkpoints/SHA256SUMS "$RUN_DIR/checkpoints.sha256"
    cp -a checkpoints "$RUN_DIR/checkpoints"
    "$PYTHON_EXE" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "complete"' "$RUN_DIR/evaluation/metrics.json"
    printf '\nTraining and evaluation finished successfully.\n'
    exit 0
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    printf 'Usage: bash run_all.sh [python_executable]\n'
    printf 'Activate your working PyTorch environment first. Existing models are backed up, then retrained.\n'
    exit 0
fi
if (( $# > 1 )); then
    printf 'Usage: bash run_all.sh [python_executable]\n' >&2
    exit 2
fi
if (( $# == 1 )); then
    PYTHON_EXE="$1"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXE="$(command -v python)"
else
    PYTHON_EXE="$(command -v python3)"
fi

mkdir -p "$PROJECT_ROOT/results"
RUN_DIR="$(mktemp -d "$PROJECT_ROOT/results/full_run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
printf '{"status":"running"}\n' > "$RUN_DIR/run_status.json"
printf 'Python: %s\nResults: %s\n' "$PYTHON_EXE" "$RUN_DIR"

# A separate worker lets tee close fully before the archive is created.
set +e
bash "$PROJECT_ROOT/run_all.sh" --worker "$RUN_DIR" "$PYTHON_EXE" 2>&1 | tee "$RUN_DIR/terminal.txt"
pipeline_status=("${PIPESTATUS[@]}")
set -e

if [[ "${pipeline_status[0]}" != 0 || "${pipeline_status[1]}" != 0 ]]; then
    printf '{"status":"failed","worker_exit_code":%s,"logging_exit_code":%s}\n' "${pipeline_status[0]}" "${pipeline_status[1]}" > "$RUN_DIR/run_status.json"
    printf '\nRun stopped. Your logs are saved in:\n%s\n' "$RUN_DIR" >&2
    printf 'Check terminal.txt for the exact error. Missing project files must be copied; only import failures indicate an environment problem.\n' >&2
    exit 1
fi

printf '{"status":"complete"}\n' > "$RUN_DIR/run_status.json"
tar -czf "${RUN_DIR}.tar.gz" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
printf '\nDONE. Share this archive:\n%s.tar.gz\n' "$RUN_DIR"
