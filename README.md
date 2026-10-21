# Secure AI Systems: MNIST red and blue teaming

Repository: https://github.com/AyushRajpurohit/Secure_AI_Systems

This project evaluates a PyTorch MNIST CNN against inference-time adversarial examples (FGSM and PGD, OWASP ML01) and training-label poisoning (OWASP ML02). Defenses include FGSM/PGD adversarial training and trusted-snapshot dataset verification with clean recovery.

Start with [the combined report](report/REPORT.md) and [the integration review](report/INTEGRATION.md). Existing result files are the experimental record, not outputs to overwrite during setup.

## Layout

- `mnist_adversarial/`: executed adversarial experiment implementation, three checkpoints, raw MNIST, accepted run and its source snapshots.
- `src/`: reusable poisoning/integrity and MNIST training utilities.
- `config/`: agreed configuration and saved 55k/5k split.
- `notebooks/poisoning_experiments.ipynb`: executed Colab poisoning record, including original Drive paths and inline functions.
- `results/poisoning/`: three-seed metrics, confusion matrices, gate and recovery evidence.
- `security/sast/`: historical and combined scans with source hashes and manual reviews.
- `scripts/scan_security.py`: reproducible combined first-party scan.

The two executed tracks share architecture, pixel scaling and exact saved split. They use different trained weights, shuffle implementations and checkpoint-selection rules. Their results remain separate matched comparisons; this repository does not claim that both executed tracks already use a single imported trainer. Historical copies under result/security folders are evidence, not runnable application modules.

## Setup

Use Python 3.10 or newer. From the repository root on Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` is a convenience dependency list, not a complete lockfile. The accepted GPU adversarial run used Python 3.10.22, torch 2.7.0+cu126, torchvision 0.22.0+cu126, numpy 1.26.4 and scikit-learn 1.7.0. Consult `mnist_adversarial/results/full_run_20261019_212556_9hj6oN/environment.txt` for that run. The poisoning run used Python 3.13.16, torch 2.11.0+cpu, torchvision 0.26.0+cpu and numpy 2.1.3. There is no complete original poisoning lockfile. Do not assume different environments reproduce identical floating-point results or timings.

For a CPU-only installation matching the reviewed adversarial major versions:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
```

Choose an appropriate CUDA build if using a GPU; the CPU command does not reproduce GPU timing.

## Fast checks: no full training

```bash
cd mnist_adversarial
python -m unittest discover -s tests -v
bash -n run_all.sh
cd ..
python scripts/scan_security.py
```

Tests include real defense-loop smoke tests on tiny synthetic batches; they do not retrain the saved MNIST models. The scanner creates a new timestamped evidence directory, scans active first-party modules/tests/helpers and exports the current notebook without executing it. It excludes historical snapshots, third-party packages, datasets, weights and result data. Raw LOW warnings still require review. IPython magic/shell bodies and Bash scripts are not directly audited by Bandit.

## Evaluate saved adversarial models without retraining

This is computationally substantial: it evaluates all three models on 10,000 images with FGSM and iterative PGD.

```bash
cd mnist_adversarial
python scripts/verify_saved_models.py --output-dir results/local_verification_01
```

Use a new output directory on each run. Existing checkpoints remain unchanged. Avoid `python -O` with the full verifier, which intentionally rejects it; active attack-bound checks now also use explicit exceptions and remain enabled under optimization.

The loader expects a dictionary containing `model_state_dict`. Poisoning checkpoints are plain state dictionaries and cannot be passed directly to this loader without adapting the format.

## Reproduce poisoning work in Colab

Open `notebooks/poisoning_experiments.ipynb` in Colab. Its execution record preserves inline definitions and Drive paths; it is not a turn-key local notebook. Review the `PROJECT_DIR` assignment first. To make new runs, use a NEW Drive directory, such as `/content/drive/MyDrive/Secure_MNIST_Reproduction_20261021`, so original evidence is preserved.

Run setup/data/reference cells first, then nested label selection, gate/recovery, function export/import, baseline training, attack runs, recovered training and seed repeats. Use notebook section headings rather than relying on mutable cell numbers. The old standalone 30% attack is exploratory; reported experiments use the nested permutation selection. Package-upload/export-history/SAST cells are workflow history and are not needed to generate the scientific results; use the root scan utility for the final source scan. Later package smoke cells require the separately supplied package ZIP and are not prerequisites for the original experiments.

The notebook writes a new trusted reference before poisoning. This reference is an experimental trust assumption, not proof of authentic source data. Never regenerate expected hashes from attacked candidate data to pass verification. Actual original poisoning checkpoints, attack NPZs, trusted reference and recovery snapshot are not in this repository; retain or share the originals from Drive as needed. Original instructions used `Secure_MNIST_Person2`; inspect the current notebook's PROJECT_DIR rather than assuming a folder name.

The reusable trainer accepts:

```python
model, history, best_epoch, seconds = train_experiment(
    dataset, seed=42, config=config, device=device,
    validation_loader=validation_loader,
)
```

Protected training must verify raw data before constructing/using the training dataset. The low-level trainer intentionally also supports unprotected attack experiments. The exported standalone label-flip helper uses `choice`; it is not the final nested `permutation` selector.

## Full adversarial retraining

`bash mnist_adversarial/run_all.sh` creates a new run directory, backs up current checkpoints/results, then overwrites the working checkpoint files with newly trained models. It is NOT required to read or validate the submitted metrics. The merged-layout ignore-file path is fixed. Bash syntax and snapshot-path discovery were checked, but the full expensive training runner was not rerun during integration review. It uses GNU `sha256sum`; on macOS install/provide that utility or use Linux/Colab.

## Main observed results

- Original clean adversarial baseline: 98.84% accuracy; FGSM epsilon .3: 10.99%; PGD epsilon .3/40 steps: 0%.
- FGSM-trained model: 98.54% clean; 99.56% against FGSM .3, but 0% against PGD .3. Single-step accuracy is not evidence of broad robustness.
- PGD-trained model: 98.99% clean; 77.75% FGSM .3; 55.08% PGD .3/40 steps, one start.
- At 50% of class-1 labels poisoned (5.63% of all training labels), mean clean-test accuracy fell from 98.9133% to 96.0067%; mean 1→7 error rose from .0294% to 24.8752% across three training seeds.
- The integrity gate blocked all tested altered candidates; verified clean recovery restored the seed-42 baseline.

These are matched within-track comparisons. Evasion ASR and poisoning targeted error use different denominators. CPU batch-median timings and GPU accumulated-forward timings must not be directly compared.
