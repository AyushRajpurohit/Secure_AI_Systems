# MNIST Classifier: Training, Attacks and Defenses

We trained a CNN to recognize handwritten digits, then used FGSM and PGD attacks to test how easily its predictions could be changed. We also trained two defended models and compared their performance on clean and attacked images.

This report covers the completed model-training, evasion-attack, retesting, and adversarial-training experiments. All results below come from the successful run **full_run_20261019_212556_9hj6oN**, completed on 19 October 2026. Every test evaluation used the official **10,000 MNIST test images**.

## 1. Dataset and model

MNIST contains 28 by 28 grayscale images of digits from 0 to 9. We split the official training partition into **55,000 training images and 5,000 validation images**, using seed 42. The official test partition was kept separate from training. The generated indices are saved in [split_indices.npz](results/full_run_20261019_212556_9hj6oN/split_indices.npz); this file records the split produced by the code.

Images were converted to tensors in **[0,1]**, with no normalization transform. This makes epsilon a direct bound on pixel-value changes.

All three models use the same CNN:

```text
Input: 1 x 28 x 28
3x3 convolution, 32 channels, padding 1 -> ReLU -> 2x2 max pooling
3x3 convolution, 64 channels, padding 1 -> ReLU -> 2x2 max pooling
Flatten 64 x 7 x 7 -> Linear(3136, 128) -> ReLU -> Linear(128, 10)
```

The model produces 10 raw scores, called logits. The predicted digit is the class with the highest score. Cross-entropy is used for training and attack generation.

## 2. Training setup

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Source batch size | 64 |
| Epochs per model | 5 |
| Random seed | 42 |
| Checkpoint selection | Final epoch |

Python, NumPy, and PyTorch RNGs were seeded. cuDNN benchmarking was disabled and deterministic cuDNN behavior was requested. The models were checked on validation data after each epoch, but the final checkpoint was saved after epoch five rather than selected using test results.

The run used **Python 3.10.22**, **PyTorch 2.7.0+cu126**, **torchvision 0.22.0+cu126**, **NumPy 1.26.4**, and **scikit-learn 1.7.0**, with an **NVIDIA RTX 6000 Ada Generation GPU** and CUDA 12.6. The environment and configuration are saved in [the run manifest](results/full_run_20261019_212556_9hj6oN/run_manifest.json) and [environment list](results/full_run_20261019_212556_9hj6oN/environment.txt).

### Baseline training progress

| Epoch | Training loss | Validation loss | Validation accuracy |
| --- | --- | --- | --- |
| 1 | 0.187969 | 0.078711 | 97.60% |
| 2 | 0.051064 | 0.047304 | 98.72% |
| 3 | 0.034602 | 0.044358 | 98.82% |
| 4 | 0.026287 | 0.053054 | 98.40% |
| 5 | 0.019994 | 0.040166 | 98.76% |

Training loss fell from 0.187969 to 0.019994. The final validation accuracy was 98.76%. The full training history is saved in [clean_run_metadata.json](results/full_run_20261019_212556_9hj6oN/clean_run_metadata.json).

## 3. Performance on clean images

The following table uses the separate verification pass for all three models, so its losses and timings come from the same evaluation stage.

| Model | Correct / 10,000 | Clean accuracy | Loss | Forward-pass time (s) | Average time (ms/image) |
| --- | --- | --- | --- | --- | --- |
| Baseline | 9884 | 98.84% | 0.034715 | 0.386299 | 0.038630 |
| FGSM-trained | 9854 | 98.54% | 0.046315 | 0.076807 | 0.007681 |
| PGD-trained | 9899 | 98.99% | 0.028975 | 0.035487 | 0.003549 |

The baseline made **116 mistakes**, the FGSM-trained model made **146**, and the PGD-trained model made **101**. PGD training retained high clean accuracy in this run; its clean score was 0.15 percentage points above the baseline. This is one seeded experiment, so that small difference is not evidence of a general clean-accuracy improvement.

### Baseline confusion matrix

Rows are true digits and columns are predicted digits, both in order 0-9.

```text
True / predicted digit:  0    1    2    3    4    5    6    7    8    9
0   979    0    0    1    0    0    0    0    0    0
1     0 1132    0    1    0    0    1    0    1    0
2     2    0 1027    0    0    0    0    2    1    0
3     1    0    1 1002    0    2    0    0    4    0
4     0    0    2    0  959    0    1    0    2   18
5     1    0    0    4    0  884    1    0    0    2
6     7    2    1    0    1    6  939    0    2    0
7     0    2   15    2    0    0    0 1001    1    7
8     5    0    2    0    0    1    0    0  961    5
9     0    0    1    1    2    4    0    1    0 1000
```

The matrix contains 10,000 predictions, with 9,884 on the diagonal. The most frequent single confusion was digit 4 being predicted as 9, in 18 cases. All three clean matrices and all attacked matrices are included in [CONFUSION_MATRICES.md](CONFUSION_MATRICES.md).

## 4. Red-team attacks

We implemented **untargeted white-box FGSM and PGD** in PyTorch. The attacker knows the model and the true label, and uses input gradients to increase cross-entropy. The attacker changes image pixels within an L-infinity bound; the model weights and true labels remain unchanged.

Both attacks exercise [OWASP ML01:2023 Input Manipulation](https://owasp.github.io/www-project-machine-learning-security-top-10/docs/ML01_2023-Input_Manipulation_Attack.html). FGSM provides a fast, one-step attack. PGD tests repeated updates within the same threat model.

| Attack | Evaluation settings |
| --- | --- |
| FGSM | One sign-gradient step; epsilon 0.1, 0.2, 0.3; no random start |
| Main PGD | Epsilon 0.3, alpha 0.01, 40 steps, random start |
| Baseline PGD sweep | Epsilon 0.05, 0.10, 0.20, 0.30; alpha 0.01 and 40 steps for each |

PGD used one random-start trajectory per image. After each step, it projected the perturbed image back into the allowed distance from the fixed original image and clipped pixels to [0,1]. FGSM also clipped pixels to [0,1]. Model evaluation mode was used during generation, and gradients were computed for the input rather than accumulated into model parameters.

### Metric definitions

Clean and adversarial accuracy count correct predictions over **all 10,000 test images**. Loss is mean cross-entropy, weighted correctly for the final partial batch.

Attack success rate counts how often an attack changes an originally correct prediction into a wrong one. Its denominator is the number of originally correct predictions: **9,884 for the baseline, 9,854 for FGSM training, and 9,899 for PGD training**. This differs from the denominator of adversarial accuracy, so the two percentages do not necessarily add to 100%.

### Baseline attack results

| Attack | Epsilon | Adversarial accuracy | Loss | Attack success rate |
| --- | --- | --- | --- | --- |
| FGSM | 0.1 | 85.22% | 0.490997 | 13.78% |
| FGSM | 0.2 | 39.54% | 2.409991 | 60.00% |
| FGSM | 0.3 | 10.99% | 4.776482 | 88.88% |
| PGD-40 | 0.05 | 94.46% | 0.181305 | 4.43% |
| PGD-40 | 0.1 | 73.93% | 0.912398 | 25.20% |
| PGD-40 | 0.2 | 1.45% | 8.312603 | 98.53% |
| PGD-40 | 0.3 | 0.00% | 16.994822 | 100.00% |

The baseline's PGD accuracy fell from **94.46% at epsilon 0.05** to **73.93%, 1.45%, and 0.00%** as epsilon increased to 0.10, 0.20, and 0.30. At epsilon 0.3, PGD caused every test image to be classified incorrectly: the confusion-matrix diagonal is entirely zero. High clean accuracy therefore did not imply resistance to the evaluated attack.

## 5. Blue-team adversarial training

Both defended models started from a fresh initialization using the same CNN. For every training batch, adversarial images were regenerated against the current model and combined with the clean images in a **50/50 mix**. A source batch of 64 gives up to 128 images for a defense update. Labels were duplicated unchanged, and mean cross-entropy was computed over the combined batch.

FGSM training used **epsilon 0.3**. PGD training used **epsilon 0.3, alpha 0.01, 10 steps, and a random start**. Both ran for five epochs with Adam and learning rate 0.001. Adversarial images were detached before the optimizer update, and the weight update ran in training mode. The saved final validation accuracies were 98.26% for FGSM training and 98.94% for PGD training, as recorded in the [FGSM training log](results/full_run_20261019_212556_9hj6oN/fgsm_training.txt) and [PGD training log](results/full_run_20261019_212556_9hj6oN/pgd_training.txt).

For evaluation, attacks were generated against each defended model itself. Both defenses used the same test loader and evaluation attack settings as the baseline. Evaluation PGD used 40 steps, compared with 10 steps during PGD training.

### FGSM-trained model

| Attack | Epsilon | Adversarial accuracy | Loss | Attack success rate |
| --- | --- | --- | --- | --- |
| FGSM | 0.1 | 90.02% | 0.391899 | 8.68% |
| FGSM | 0.2 | 96.94% | 0.103964 | 2.37% |
| FGSM | 0.3 | 99.56% | 0.011994 | 0.30% |
| PGD-40 | 0.3 | 0.00% | 42.024042 | 100.00% |

FGSM accuracy increased with epsilon and reached **99.56% at epsilon 0.3**, even exceeding this model's clean accuracy. However, PGD reduced its accuracy to **0.00%**. The result shows why the high one-step attack score cannot be used as evidence of protection against the evaluated iterative attack. These final scores alone do not establish the exact cause of that behavior.

### PGD-trained model

| Attack | Epsilon | Adversarial accuracy | Loss | Attack success rate |
| --- | --- | --- | --- | --- |
| FGSM | 0.1 | 96.17% | 0.116506 | 2.85% |
| FGSM | 0.2 | 90.15% | 0.319599 | 8.93% |
| FGSM | 0.3 | 77.75% | 0.725074 | 21.46% |
| PGD-40 | 0.3 | 55.08% | 1.499860 | 44.36% |

The PGD-trained model correctly classified **5,508 of 10,000 attacked images** under PGD at epsilon 0.3, compared with zero for the baseline and FGSM-trained models. It gave the strongest protection among the three models under this evaluation, although it still misclassified 44.92% of attacked images.

## 6. Limitations

All results here are full-test results from one seed and the stated attack configurations. There were no additional PGD restarts or best-iterate tracking. These measurements do not certify resistance to every possible allowed perturbation.

## 7. Observations

The baseline recognized clean digits accurately but failed under PGD at epsilon 0.3. FGSM training produced a high FGSM score without improving PGD accuracy in this experiment. PGD training provided a clear improvement against the evaluated PGD attack while retaining 98.99% clean accuracy. The comparison shows the value of evaluating a defense with an iterative attack rather than relying on its score against the attack used during training.

## Code and result files

- Model, data, and baseline training: `src/model.py`, `src/data.py`, `src/train.py`.
- Clean metrics and timing: `src/evaluate.py`.
- Attack generation and bounds: `attacks/fgsm.py`, `attacks/pgd.py`, `attacks/sanity.py`.
- Defense training: `defenses/adversarial_training.py`, `defenses/pgd_adversarial_training.py`.
- Evaluation: `attacks/evaluate_attacks.py`, `evaluate_defended.py`, `scripts/verify_saved_models.py`.
- Complete run: [results/full_run_20261019_212556_9hj6oN/](results/full_run_20261019_212556_9hj6oN/).
- Full-precision metrics and all matrices: [evaluation/metrics.json](results/full_run_20261019_212556_9hj6oN/evaluation/metrics.json).
- Tests, sanity statistics, and printed results: [evaluation/evaluation.txt](results/full_run_20261019_212556_9hj6oN/evaluation/evaluation.txt).
- Dataset hashes: [data/SHA256SUMS](data/SHA256SUMS).
- Exact model files: `checkpoints/clean_model.pt`, `checkpoints/adversarially_trained_model.pt`, `checkpoints/pgd_adversarially_trained_model.pt`; identities are recorded in `checkpoints/SHA256SUMS`.

Values in the tables are rounded for readability. The linked JSON retains the saved full precision.
