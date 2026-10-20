# Secure AI Systems — MNIST Red/Blue Teaming

PyTorch CNN baseline, untargeted white-box FGSM/PGD attacks, and FGSM/PGD adversarial training on MNIST. Source, checkpoints, and existing result files are included.

## Completed results

The final run is `results/full_run_20261019_212556_9hj6oN/`. It completed training and evaluation, passed all 14 correctness tests, and preserved source, dataset, and checkpoint integrity during the recorded checks.

| Model | Clean accuracy | FGSM epsilon 0.3 | PGD epsilon 0.3 |
| --- | ---: | ---: | ---: |
| Baseline | 98.84% | 10.99% | 0.00% |
| FGSM-trained | 98.54% | 99.56% | 0.00% |
| PGD-trained | 98.99% | 77.75% | 55.08% |

All figures use the full 10,000-image test set. PGD uses alpha 0.01, 40 steps, and one random start. Read [PROJECT_REPORT.md](PROJECT_REPORT.md) for the report, [results.md](results.md) for the short summary, and [CONFUSION_MATRICES.md](CONFUSION_MATRICES.md) for all 18 clean/attacked matrices. Full-precision values are in [evaluation/metrics.json](results/full_run_20261019_212556_9hj6oN/evaluation/metrics.json). [PUBLICATION_REVIEW.md](PUBLICATION_REVIEW.md) records the final verification and publication checks. The earlier audit files are historical and do not supply the final result figures.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from the repository root. MNIST downloads to `./data` if needed. Virtual environments are machine-specific; recreate `.venv` if its original Python interpreter is unavailable.

The completed GPU run used Python 3.10.22, PyTorch 2.7.0+cu126, torchvision 0.22.0+cu126, NumPy 1.26.4, and scikit-learn 1.7.0 on an NVIDIA RTX 6000 Ada Generation GPU. The exact environment is recorded with the run. `requirements.txt` remains the existing unpinned dependency list; installing it on another machine does not guarantee identical versions or timings.

To run the entire pipeline in your working PyTorch environment:

```bash
bash run_all.sh
```

The script backs up existing checkpoints/metric files, retrains all three models, runs the tests and full evaluation, and writes a new `results/full_run_*` directory and archive. It preserves previous run folders and the MNIST data. To evaluate existing checkpoints without retraining:

```bash
python scripts/verify_saved_models.py --output-dir results/verified_run
```

The output directory must be new. Both scripts record failures instead of presenting incomplete output as a finished run. Copy the `scripts/` directory along with `run_all.sh` when moving the project to another machine.

## Fixed experiment settings

- Inputs: `ToTensor()` in `[0,1]`, without `Normalize()`.
- Split: 55,000 training / 5,000 validation examples from the official training set, seed 42. The official 10,000-example test set is used only for final evaluation.
- CNN: two convolution/ReLU/pooling blocks, then a 128-unit hidden layer and 10 logits. All experiments use the same architecture.
- Baseline and both defenses: Adam, learning rate 0.001, source batch size 64, 5 epochs, seed 42.
- FGSM training: epsilon 0.3; concatenate each clean batch with its adversarial batch (50/50, up to 128 training examples per update).
- PGD training: epsilon 0.3, alpha 0.01, 10 steps, random start; the same 50/50 combination.
- Main PGD evaluation: epsilon 0.3, alpha 0.01, 40 steps, random start.
- Baseline PGD sweep: epsilon 0.05, 0.10, 0.20, 0.30, each with alpha 0.01 and 40 steps.
- FGSM evaluation: epsilon 0.1, 0.2, 0.3.

## Evaluate saved checkpoints

```bash
python attacks/evaluate_attacks.py --print-sanity
python evaluate_defended.py checkpoints/adversarially_trained_model.pt --print-sanity
python evaluate_defended.py checkpoints/pgd_adversarially_trained_model.pt --print-sanity
```

`evaluate_defended.py` defaults to the FGSM-trained checkpoint when no path is provided. `src/pgd_evaluate_defence.py` accepts the same arguments and defaults to the PGD-trained checkpoint.

Baseline and defenses use the same official test loader configuration (batch size 64, no shuffling), attack implementations, FGSM settings, and main PGD settings. The additional PGD sweep runs on the baseline. Each attack resets seed 42, so the main PGD comparison starts from the same random perturbations on the same device/runtime.

Evaluation prints clean accuracy, adversarial accuracy/loss, attack success rate, elapsed attack/evaluation time, and adversarial confusion matrices. Accuracy uses all test examples. Attack success rate uses only examples classified correctly before the attack; it reports 0 if that set is empty. Confusion matrix rows are true labels and columns are predictions, in digit order 0–9.

Attack bounds are asserted automatically, including the random start and every PGD step. `--print-sanity` also prints the maximum L-infinity perturbation and minimum/maximum adversarial pixel values for each attack over the test set. Attack functions accept `verbose=True` to print these statistics for an individual batch. Checks use a perturbation tolerance of `1e-6` and require pixels in `[0,1]`. Run Python normally; `python -O` disables assertions.

FGSM/PGD generate input gradients in evaluation mode, return detached examples, preserve parameter gradients, and restore the caller's model mode. Training clears optimizer gradients before generating adversarial examples using the current model, then trains on the combined batch.

## Reproduce training

```bash
python src/train.py
python defenses/adversarial_training.py
python defenses/pgd_adversarial_training.py
```

These commands overwrite their corresponding checkpoint files:

```text
checkpoints/clean_model.pt
checkpoints/adversarially_trained_model.pt
checkpoints/pgd_adversarially_trained_model.pt
```

Baseline training saves `results/clean_run_metadata.json`. FGSM training saves `results/adversarial_training_clean_test.json`. The existing output schemas are retained. The PGD training script itself prints clean test metrics without a results JSON; the full-run verifier saves clean metrics and all attack results for all three models in the run's `evaluation/metrics.json`.

The original dataset and all three checkpoints are included. See [data/README.md](data/README.md) for attribution and integrity information, and [checkpoints/README.md](checkpoints/README.md) for the model files. Archives, old backups, virtual environments, and failed setup attempts are ignored by Git without being deleted. Dataset files, `.pt` checkpoints, and the successful run evidence are not ignored. Use command-line Git for the dataset, since its largest file exceeds the browser-upload limit.

Training and evaluation seed Python, NumPy, and PyTorch with 42. cuDNN benchmarking is disabled and deterministic cuDNN kernels are requested. This improves repeatability on a fixed environment; it does not promise bitwise equality across hardware or PyTorch versions. The original baseline metadata records PyTorch 2.7.0+cu126 and Python 3.10.22. `requirements.txt` specifies the existing dependencies without pinning a new environment over the old experiments.

## Correctness checks

```bash
python -m unittest discover -s tests -v
```

Tests cover input gradients, fixed-origin PGD projection, random initialization, zero epsilon, detached outputs, parameter gradient isolation, model modes, both training scripts, checkpoint selection, matching evaluation attacks, metric denominators, confusion matrices, empty loaders, and the official dataset split. MNIST must be available locally or downloadable for the split check.

See `AUDIT.md` for findings and validation results. A finite PGD run is an empirical attack, not a certificate of robustness. The sweep is diagnostic: it reports observed accuracy without forcing monotonic results or selecting settings to improve reported robustness.

`results/audit_validation.json` and `results/audit_validation.txt` record this review's runtime checks: full clean-test evaluation for all three checkpoints, baseline attacks on the first 1,024 test examples, and defense attack smoke tests on the first 64. Their subset attack results must not be reported as full-test robustness results. The evaluation commands above use all 10,000 test examples.
