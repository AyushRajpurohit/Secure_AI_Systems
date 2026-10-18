# Integration status

The repository is organized by function and uses shared PyTorch modules.

## Current contribution
- Training-data poisoning experiments across seeds 42, 43 and 44.
- Trusted-snapshot integrity verification and recovery.
- Threat-model and privacy discussion draft.
- SAST evidence and reviewed findings.

## Shared configuration
Use src/mnist_training.py, config/experiment_config.json and
config/split_indices.npz. Apply split indices to the official MNIST
training-set order.

## Remaining integration
- Add FGSM, PGD and adversarial-training implementation and results.
- Clean notebook imports and paths to use the shared repository modules.
- Merge report sections and write the main reproduction README.
- Scan the final combined first-party application source.

The poisoning notebook preserves the executed experiment record and
currently retains its original Drive paths and inline definitions.
