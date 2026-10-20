# MNIST: Attacks and Defenses

We trained a PyTorch CNN on MNIST and tested it with FGSM and PGD attacks. We then compared FGSM and PGD adversarial training using the same model architecture and test images.

## Final results

All figures below use the official 10,000-image test set from the completed run `full_run_20261019_212556_9hj6oN`.

| Model | Clean accuracy | FGSM epsilon 0.3 | PGD epsilon 0.3 |
| --- | --- | --- | --- |
| Baseline | 98.84% | 10.99% | 0.00% |
| FGSM-trained | 98.54% | 99.56% | 0.00% |
| PGD-trained | 98.99% | 77.75% | 55.08% |

PGD evaluation used alpha 0.01, 40 steps, and one random start. Baseline PGD accuracy was **94.46%, 73.93%, 1.45%, and 0.00%** at epsilon **0.05, 0.10, 0.20, and 0.30** respectively.

## What we found

The baseline handled clean digits well but failed against PGD at epsilon 0.3. FGSM training scored highly against FGSM yet still failed against PGD. PGD training gave the best protection in this comparison, reaching 55.08% PGD accuracy while keeping 98.99% clean accuracy. It improved resistance to this attack, but did not make the model immune to attacks.

All 14 correctness tests passed, and all perturbation/pixel checks passed. The checkpoint hashes match the completed evaluation. See [PROJECT_REPORT.md](PROJECT_REPORT.md) for the full report, [CONFUSION_MATRICES.md](CONFUSION_MATRICES.md) for the matrices, and [the saved metrics](results/full_run_20261019_212556_9hj6oN/evaluation/metrics.json) for full-precision results.
