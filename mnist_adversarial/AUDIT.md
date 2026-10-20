# Correctness audit

> Historical audit of an earlier snapshot. The completed final run and its verified results are documented in `PROJECT_REPORT.md` and `PUBLICATION_REVIEW.md`. Do not substitute this audit's old subset results for the final full-test metrics.

Scope: current working tree, compared with the staged source where useful. Several fixes and refactors were already unstaged when this audit began; they are not attributed to this review. No checkpoint or existing result JSON was overwritten.

## Findings

No numerical error was found in the FGSM/PGD update rules, training updates, metric denominators, or dataset separation.

- FGSM maximizes cross-entropy with the true, unchanged labels using the sign of the input gradient.
- PGD starts with uniform noise in `[-epsilon, epsilon]`, clips to `[0,1]`, and retains a fixed original image. Each update projects back to the original image's L-infinity ball, then clips pixels. Fresh detached input leaves prevent accumulation across iterations.
- Current attacks use `autograd.grad` only for inputs, run in eval mode, return detached tensors, and restore the caller's mode even on failure. Bounds are checked at initialization and every PGD step with tolerance `1e-6`. No normalization mismatch exists.
- Clean/adversarial accuracy use every test example. ASR counts misclassifications among originally correct examples only. The existing convention returns zero when no samples were originally correct; that case should be interpreted as undefined ASR. Losses are weighted by batch size. Confusion rows are targets and columns predictions, with digit labels 0–9.
- Both defenses generate attacks against the current model, concatenate clean/adversarial images and duplicate the unchanged labels, clear gradients before attacks, and optimize the detached combined batch in train mode. Validation happens only on the fixed training-set holdout. Test evaluation is after training and checkpoint saving; there is no test-based model selection in these scripts.
- Baseline and defenses share the same loader configuration and attack suite. Main PGD uses epsilon 0.3, alpha 0.01, 40 steps, random start. Training PGD uses epsilon 0.3, alpha 0.01, 10 steps. Baseline adds 0.05/0.10/0.20/0.30 PGD sweeps. Seed 42 resets before each evaluation attack, aligning random starts for a fixed device/runtime and batch order.

## Real bugs

1. The staged `src/pgd_evaluate_defence.py` computed its root as `src`, producing nonexistent `src/attacks` and `src/checkpoints` paths. It also selected the FGSM-trained checkpoint. The existing working-tree wrapper fixed paths but still defaulted to FGSM. The wrapper now defaults to `checkpoints/pgd_adversarially_trained_model.pt`; an explicit checkpoint still overrides it.
2. Clean evaluation divided by zero for an empty loader. It now raises an informative `ValueError`, matching attack evaluation. This has no effect on the nonempty official test set.

The staged attack helpers called `model.zero_grad()` and `loss.backward()`, clearing and populating parameter gradients, and left the model in eval mode. Those are undesirable helper side effects, already removed in the working tree. Importantly, the staged defense loops explicitly switched back to train mode and cleared optimizer gradients after attack generation, so these side effects do not establish contaminated training updates or invalid old checkpoints. Input gradients were fresh in each PGD step even in the staged implementation.

## Improvements without an established numerical bug

- Shared attack evaluation, checkpoint loading, seeding, device selection, and timing synchronization; removed duplicate evaluators and unused code.
- Explicit checkpoint CLI and safe state-dict loading; preserve all checkpoint filenames.
- Automatic attack assertions and optional per-batch or whole-test-set statistics.
- Baseline epsilon sweep using unchanged alpha 0.01 and 40 steps.
- Regression tests for gradients, projection, random start, modes, training loops, metrics, checkpoint selection, and dataset split.
- Runtime reproducibility settings are documented. The copied `.venv` points to a missing interpreter; recreate it using the README setup. Dependencies were not added to `requirements.txt`.

## Validation

All 14 regression tests passed using `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /tmp/mnist-audit-env/bin/python -m unittest discover -s tests -v`. Both defense scripts ran their actual five-epoch loops on tiny synthetic batches with checkpoint/result writes mocked; the full models were not retrained. Syntax compilation, all three evaluation CLI help commands, and `git diff --check` passed.

Runtime: Python 3.11.15, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, NumPy 2.4.6, scikit-learn 1.9.0; CPU with two PyTorch threads. These differ from the original training environment, so small floating-point/loss and timing differences are expected. A temporary environment supplied PyTorch; existing conda packages supplied scikit-learn. Project dependencies and the copied `.venv` were not modified.

Full official clean-test accuracy was reproduced for all saved checkpoints:

| Checkpoint | Accuracy | Examples |
| --- | ---: | ---: |
| `clean_model.pt` | 98.53% | 10,000 |
| `adversarially_trained_model.pt` | 98.54% | 10,000 |
| `pgd_adversarially_trained_model.pt` | 98.95% | 10,000 |

Baseline diagnostic PGD sweep, first 1,024 test examples in official order, batch size 64, alpha 0.01, 40 steps, random start, seed 42 reset before each attack:

| Epsilon | Adversarial accuracy |
| --- | ---: |
| 0.05 | 91.50% |
| 0.10 | 67.97% |
| 0.20 | 0.88% |
| 0.30 | 0.00% (0/1,024) |

The same subset had 98.34% clean accuracy. These are explicitly subset diagnostics, not full-test robustness numbers. Full attack evaluation is still the default in the production scripts.

Both defended checkpoints also passed the shared attack suite on the first 64 test examples. PGD accuracy at epsilon 0.3 was 0/64 for FGSM training and 39/64 (60.94%) for PGD training. The FGSM-trained checkpoint scored 64/64 against FGSM epsilon 0.3 on this smoke batch despite its 0/64 PGD result. This discrepancy is an observed limitation of that checkpoint, not evidence that FGSM accuracy establishes resistance to PGD; no training changes were made to improve it. Do not report these smoke results as full-test metrics or infer a specific training failure solely from them.

Every diagnostic sanity assertion passed. Across epsilon-0.3 attacks, maximum L-infinity perturbation was 0.3000000417, within `epsilon + 1e-6`; adversarial pixel minima/maxima were 0/1. Checkpoint SHA-256 hashes were unchanged before/after validation and are recorded with runtime, sample counts, losses, ASR, and confusion matrices in `results/audit_validation.json`. The matching console output, including pixel statistics, is in `results/audit_validation.txt`. Existing JSON files were preserved.

## Interpretation and reruns

A baseline with high clean MNIST accuracy can plausibly have zero accuracy under untargeted pixel-space PGD at epsilon 0.3. The new diagnostic independently reproduced zero accuracy on the first 1,024 test examples; the old full-10,000-example attack result was not rerun. A single random-start, final-iterate PGD run is an empirical attack, not a robustness certificate; the sweep need not be strictly monotone for finite attacks.

No retraining is required by a demonstrated mathematical correctness fix. Rerun any result attributed to the PGD checkpoint that was obtained using the old PGD entry point: it either failed on paths or selected the FGSM checkpoint. Rerun all attack evaluations with the shared seeded suite for a consistent comparison and to obtain the requested sweep and sanity statistics. Historical timings should be rerun if comparing timing after synchronization changes. Existing JSON results remain historical records, not relabeled as new runs.

## Files

Modified during this review: `attacks/evaluate_attacks.py`, `evaluate_defended.py`, `src/evaluate.py`, `src/pgd_evaluate_defence.py`, `tests/test_correctness.py`, `README.md`. Added: `AUDIT.md`, `results/audit_validation.json`, `results/audit_validation.txt`.

Previously present working-tree changes retained and audited: `attacks/fgsm.py`, `attacks/pgd.py`, `attacks/sanity.py`, `src/runtime.py`, `src/train.py`, `defenses/adversarial_training.py`, `defenses/pgd_adversarial_training.py`, plus earlier edits in the modified files above.

Before committing, stage the final revised files, including untracked helpers and tests. The index still contains earlier source versions. This audit does not change staging, create a commit, or push.
