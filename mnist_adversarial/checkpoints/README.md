# Final checkpoints

These files match the completed run `full_run_20261019_212556_9hj6oN` and its `evaluation/metrics.json`:

| File | Training | Clean accuracy | PGD-40 accuracy, epsilon 0.3 |
| --- | --- | ---: | ---: |
| `clean_model.pt` | Clean images | 98.84% | 0.00% |
| `adversarially_trained_model.pt` | Clean + FGSM images, epsilon 0.3 | 98.54% | 0.00% |
| `pgd_adversarially_trained_model.pt` | Clean + PGD images, epsilon 0.3, alpha 0.01, 10 steps | 98.99% | 55.08% |

Evaluation PGD uses alpha 0.01, 40 steps, and one random start. All models use `src/model.py`; each was trained for five epochs and evaluated on all 10,000 official test images. Each checkpoint is about 1.61 MiB and can be included in regular Git.

Verify the exact files from the repository root:

```bash
sha256sum -c checkpoints/SHA256SUMS
```

The files contain a `model_state_dict` plus experiment metadata. Evaluation loads them using `weights_only=True` and maps tensors to the selected CPU/GPU device. Evaluation does not train the model or overwrite checkpoint files. Training commands overwrite their corresponding filenames, so retain the completed run's matching copies and hashes when reporting its results.
