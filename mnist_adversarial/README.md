# Secure AI Systems — MNIST Red/Blue Teaming

A PyTorch CNN for MNIST, tested with untargeted white-box FGSM and PGD attacks and defended with adversarial training.

## Completed results

Results from `results/full_run_20261019_212556_9hj6oN/`:

| Model | Clean accuracy | FGSM epsilon 0.3 | PGD epsilon 0.3 |
| --- | ---: | ---: | ---: |
| Baseline | 98.84% | 10.99% | 0.00% |
| FGSM-trained | 98.54% | 99.56% | 0.00% |
| PGD-trained | 98.99% | 77.75% | 55.08% |

All figures use the full 10,000-image test set. PGD uses alpha 0.01, 40 steps, and one random start. Read [PROJECT_REPORT.md](PROJECT_REPORT.md) for the report, [results.md](results.md) for the short summary, and [CONFUSION_MATRICES.md](CONFUSION_MATRICES.md) for all 18 clean/attacked matrices. Full-precision values are in [evaluation/metrics.json](results/full_run_20261019_212556_9hj6oN/evaluation/metrics.json).

## Setup

Run the commands below from `mnist_adversarial/`. MNIST is included in `data/` and is downloaded if missing.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The completed GPU run used Python 3.10.22, PyTorch 2.7.0+cu126, torchvision 0.22.0+cu126, NumPy 1.26.4, and scikit-learn 1.7.0 on an NVIDIA RTX 6000 Ada Generation GPU. Package versions are recorded in the run’s `environment.txt`. Dependencies in `requirements.txt` are unpinned, so versions and timings may differ on another machine.

To run the entire pipeline in your working PyTorch environment:

```bash
bash run_all.sh
```

The script backs up existing checkpoints/metric files, retrains all three models, runs the tests and full evaluation, and writes a new `results/full_run_*` directory and archive. It preserves previous run folders and the MNIST data. To evaluate existing checkpoints without retraining:

```bash
python scripts/verify_saved_models.py --output-dir results/verified_run
```

The output directory must be new.

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

Baseline training saves `results/clean_run_metadata.json`. FGSM training saves `results/adversarial_training_clean_test.json`. The PGD training script itself prints clean test metrics without a results JSON; the full-run verifier saves clean metrics and all attack results for all three models in the run's `evaluation/metrics.json`.

The dataset hashes are in [data/SHA256SUMS](data/SHA256SUMS). See [checkpoints/README.md](checkpoints/README.md) for model details and hashes.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover attack gradients and bounds, model modes, training updates, evaluation metrics, checkpoint selection, and the dataset split.
