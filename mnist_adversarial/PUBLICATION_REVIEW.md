# Final review before GitHub publication

Reviewed on 20 October 2026. The accepted experiment is `results/full_run_20261019_212556_9hj6oN/`. Its `run_status.json` and `evaluation/metrics.json` both say `complete`.

## Review result

The recorded experiment, current model files, source, and numerical results are consistent. No further training or attack rerun is required by this review. The assignment report is `PROJECT_REPORT.md`; the short summary is `results.md`.

Checks performed against the saved outputs:

- All 14 correctness tests passed in the GPU run. The single torchvision/Pillow deprecation warning did not fail the tests.
- All 15 attack settings evaluated 10,000 examples. Each confusion matrix totals 10,000 and its row totals match the official MNIST test labels.
- Every clean and attacked accuracy matches its matrix diagonal. Every printed matrix, accuracy, loss, and attack-success rate matches the full-precision JSON within its printed rounding precision.
- ASR values correspond to integer successful-attack counts on initially correct samples, with valid possible counts of corrected originally wrong samples. This is consistent with the implemented denominator; ASR is not computed over all 10,000 examples.
- All recorded L-infinity distances are within epsilon + 1e-6, and all 15 sanity log entries keep pixels in [0,1].
- Saved split arrays contain 55,000 unique training indices and 5,000 unique validation indices. They are disjoint and cover the official 60,000-image training partition. Test indices are the official 0-9,999 order. The run's split test also passed its seed-42 generator check.
- The executed source hashes matched both the local experiment source and the saved source snapshot before publication-only edits. The current Python training/attack/evaluation code and shell runner remain unchanged by this review.
- Current checkpoints match the SHA-256 values saved in `evaluation/metrics.json` and the run's checkpoint copies. The run recorded unchanged checkpoint files during evaluation.
- All eight dataset hashes match the completed run. The raw data was not changed by this review.
- Root-level clean-training metric JSON files exactly match the completed run's copies.
- The successful `.tar.gz` archive contains matching status, metrics, and all three checkpoints. It is retained locally; the extracted run folder is the version included in Git.
- Current Python files pass syntax parsing, and `run_all.sh` passes Bash syntax checking. No new training or model evaluation was run locally as part of this review.

## Accepted result summary

| Model | Clean accuracy | FGSM epsilon 0.3 | PGD epsilon 0.3 |
| --- | ---: | ---: | ---: |
| Baseline | 98.84% | 10.99% | 0.00% |
| FGSM-trained | 98.54% | 99.56% | 0.00% |
| PGD-trained | 98.99% | 77.75% | 55.08% |

PGD uses alpha 0.01, 40 steps, and one random-start trajectory. These are empirical attack results, not certified robustness claims. Inference timing varied between training-ending and verification passes; the report labels the timing pass and does not rank models by these single measurements.

## Publication edits

Only documentation and packaging were changed in this review:

- Created `PROJECT_REPORT.md` using the completed run, including training settings/history, clean metrics and inference timing, attack results, defense results, checks, observations, and evidence links.
- Created `CONFUSION_MATRICES.md` with all 18 final clean/attacked matrices.
- Updated `results.md` and README to use the final results rather than the old 98.53% baseline experiment.
- Labeled `AUDIT.md` as historical. Its earlier subset experiments remain intact but were not used for the final report.
- Added dataset attribution and an integrity manifest in `data/README.md` and `data/SHA256SUMS`; added checkpoint documentation in `checkpoints/README.md`.
- Updated `.gitignore` to exclude local archives, backup copies, and the three failed setup attempts, while retaining raw MNIST data, `.pt` checkpoints, and the successful run folder. No files were deleted.

The `.gitignore` publication edit intentionally differs from the version recorded before the run. The original version remains in `results/full_run_20261019_212556_9hj6oN/source/.gitignore`. This does not change the model or experiment settings. The run's source hashes document the experiment's original source version rather than later publication documents.

## Files to include

Include all Python source, `scripts/`, `tests/`, `run_all.sh`, `requirements.txt`, report/summary/matrix documents, README, `data/`, `checkpoints/`, the root current metrics, and `results/full_run_20261019_212556_9hj6oN/`. The successful run retains its manifest, versions, source snapshot, split indices, training logs, test/evaluation output, full-precision metrics, and matching checkpoints. The old audit files are explicitly historical.

The following remain locally but are ignored for publication: `.venv/`, caches, `mnist_final.zip`, the empty `.tar.gz`, the successful run's compressed archive, backup checkpoint folders, and unsuccessful setup folders. The 129,370,906-byte `mnist_final.zip` would exceed GitHub's regular file limit if added, which is why archives are excluded.

The dataset totals 66,544,770 bytes, about 63.46 MiB. Its largest individual file is 47,040,016 bytes (44.86 MiB), and each model checkpoint is about 1.61 MiB. They fit regular Git without Git LFS. Use command-line Git: the largest dataset file exceeds the browser upload limit of 25 MiB, while regular Git blocks individual files larger than 100 MiB. [GitHub size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

From the repository root:

```bash
sha256sum -c data/SHA256SUMS
sha256sum -c checkpoints/SHA256SUMS
git add -A
git diff --cached --stat
git status --short
```

Stage again because the index contains older versions of source and checkpoints. Review the staged files before committing. No staging, commit, or push was performed by this review. The checkout has no configured remote or existing commit at review time; use the intended GitHub repository URL when publishing, then add that real URL to the combined assignment submission.
