# Secure AI Systems: red and blue teaming an MNIST classifier

Repository: https://github.com/AyushRajpurohit/Secure_AI_Systems

**Submitted by:** Ayush Shaitan Singh Rajpurohit (CS26MTECH12014) & Jayesh Belsare (CS26MTECH12001)

## 1. Objective and application scope

We evaluate a local PyTorch MNIST classification pipeline against inference-time input manipulation and training-label poisoning. The scope includes loading and preprocessing, the 55k/5k split, CNN definition, training, checkpoint saving/loading, clean/adversarial evaluation, attack/defense code and utilities. Assets include data and labels, code/configuration, weights, trusted references, dependencies, runtime resources and result records. There is no deployed public API, production feedback loop or real private-user dataset.

The system flows from external data/dependencies into preprocessing and split, training, stored checkpoints, inference inputs and prediction outputs. Trust boundaries occur at downloads, candidate-data ingestion, model-file loading, notebook/package imports, mounted storage, and output/artifact sharing. Potential runtime permission abuse is within threat modeling; it was not demonstrated experimentally.

Two experimental tracks use the same CNN and exactly matching saved splits. They have separate trained baselines and execution environments. Within-track comparisons are matched; cross-track baseline numbers are not substituted for one another.

## 2. Shared model and experimental differences

MNIST pixels are float32 [N,1,28,28] in [0,1], without mean/std normalization. The CNN is Conv(1→32,3×3,padding1), ReLU, MaxPool2, Conv(32→64,3×3,padding1), ReLU, MaxPool2, Flatten, Linear(3136→128), ReLU, Linear(128→10). It returns logits and uses cross-entropy. Training uses Adam, learning rate .001, batch size 64, five epochs. Official training data is split into 55,000 training/5,000 validation examples; all 10,000 official test examples are held out.

| Setting | Input-manipulation track | Poisoning track |
|---|---|---|
| Training seed | 42 | 42,43,44 |
| Split | Seed-42 random_split; saved indices verified identical | Saved seed-42 randperm indices |
| Checkpoint selection | Final epoch, for baseline and both defenses | Lowest clean validation loss |
| Shuffle RNG | Global torch RNG | Separate seeded training-loader generator |
| Evaluation batch | 64 | 256 |
| Checkpoint format | Dict containing model_state_dict and metadata | Plain state_dict |
| Environment | Python3.10.22, torch2.7.0+cu126, torchvision0.22.0+cu126, numpy1.26.4, sklearn1.7.0 | Python3.13.16, torch2.11.0+cpu, torchvision0.26.0+cpu, numpy2.1.3 |
| Hardware | NVIDIA RTX6000 Ada GPU | CPU; exact CPU not recorded |

The original implementations remain as documented experiment tracks. The exported shared trainer has explicit config/device/validation-loader arguments and passed a model-parity check and tiny training-interface smoke test in Colab; those checks are not a full scientific rerun through shared imports.

## 3. Attacker models and OWASP ML mapping

**Inference attacker (ML01:2023 Input Manipulation):** white-box access to the model, true labels and input gradients; untargeted objective of increasing classification error. The attacker changes pixels within a specified L-infinity budget, keeps pixels in [0,1], and cannot alter labels or model weights. The same 10,000 test images, including initially incorrect ones, are attacked for each setting.

**Training-data attacker (ML02:2023 Data Poisoning):** knows the pipeline and can change a bounded subset of training labels from 1 to 7, without changing pixels. Cannot change validation/test data, training/verification code, trusted reference or recovery snapshot. The objective is increased prediction of 7 for true test ones. This is post-reference tampering; an attacker controlling the initial trusted source is outside the demonstrated defense model.

FGSM and PGD are two methods in ML01, not two different OWASP risk categories. Label poisoning supplies ML02. Direct checkpoint modification would be ML10 and is discussed, not attacked. Feedback-loop model skewing (ML08 in the class notes) is not demonstrated by this static-label experiment.

## 4. STRIDE threat model

Mappings follow the supplied Class12 ML-specific definitions. Associations are contextual, not a claim that STRIDE and OWASP are one-to-one taxonomies.

| Category | Threat in the scoped pipeline | OWASP association | Evidence/control status |
|---|---|---|---|
| Spoofing | Impersonate an approved dataset, job or checkpoint producer | ML06/ML10 depending on artifact | Source identity/provenance verification recommended; protected reference is assumed |
| Tampering | Alter pixels at inference, labels before training, weights or outputs | Demonstrated ML01/ML02; potential ML10/ML09 | Pixel constraints and data gate implemented; signed output/model authenticity not enforced |
| Repudiation | Deny a data change, training execution or checkpoint promotion | ML02/ML10 context | Configs, hashes and logs retained; no tamper-proof audit system |
| Information disclosure | Expose secrets/weights or infer sensitive training information | ML05, potential ML03/ML04 | Restricted access/minimal output recommended; extraction/inversion/membership attacks not tested |
| Denial of service | Exhaust training/inference memory, compute or storage | General system threat; malicious inputs can involve ML01 | Resource limits/quotas recommended; no DoS experiment |
| Elevation of privilege | Unsafe loading, dependencies or imported notebook/package code accesses runtime/Drive | ML06 | weights_only loading and reviewed code; runtime isolation and supply-chain authenticity not established |

The checkpoint loader uses `torch.load(..., weights_only=True, map_location=device)` and expects `model_state_dict`; this reduces unsafe loading exposure but does not authenticate model provenance. Arbitrary supplied checkpoint paths are not restricted to an approved registry. The poisoning gate is demonstrated in its protected experiment path; the separate adversarial data loader does not enforce it globally.

## 5. LINDDUN privacy discussion

MNIST is public. These concerns address a hypothetical form-digit service, not measured privacy leaks in this experiment.

| Category | Potential real-service issue | Recommended control |
|---|---|---|
| Linkability | Join requests through account/document identifiers | Minimize identifiers and separate identity from prediction logs |
| Identifiability | Identify submitters through document context/metadata | Crop needed content; remove unnecessary metadata |
| Non-repudiation | Persistent logs bind a person to sensitive submissions | Limit logging detail, access and retention |
| Detectability | Infer a record's existence or participation in training | Assess membership exposure and minimize unnecessary outputs |
| Disclosure | Expose uploaded forms, identifiers or logs | Enforce access control/protected storage |
| Unawareness | Users do not understand retention or training reuse | Clear notice and meaningful processing controls |
| Non-compliance | Conflict with consent, purpose, policy or applicable obligations | Define and enforce purpose, retention and deletion procedures |

Privacy non-repudiation concerns excessive evidence of association; security repudiation concerns insufficient accountability. No private dataset, privacy attack, differential privacy, consent workflow or retention enforcement was implemented here.

## 6. Red-team implementation

### FGSM and PGD

FGSM takes x + epsilon*sign(input gradient of cross-entropy), then clips to [0,1]. Tested epsilons: .1,.2,.3. PGD evaluation uses step .01 and 40 iterations, one uniform random start in the permitted box, projection around the original image after every step and clipping to [0,1]. Baseline PGD epsilons: .05,.1,.2,.3; defenses are evaluated at .3. PGD returns the final iterate, with no multiple restarts or strongest-iterate selection.

Attacks use evaluation mode, image-only gradients through autograd.grad, detach intermediate images, restore the original model mode and do not update model weights. Shape, finite values, range and epsilon bounds are checked (tolerance 1e-6). The observed max change at epsilon .3 was .3000000417, within that tolerance. The final code uses explicit exceptions for bound checking rather than assertions that disappear under Python optimization.

Adversarial accuracy uses all 10,000 images. Evasion ASR counts initially correct images that become incorrect, divided by initially correct images: 9,884 baseline, 9,854 FGSM-trained, 9,899 PGD-trained. It is not necessarily one minus adversarial accuracy. No-empty-correct fallback behavior occurred in these runs.

### Label poisoning

There are 6,190 training ones. A fixed seed-42 permutation creates nested sets of 619,1,857,3,095 records (10%,30%,50% of source class; 1.1255%,3.3764%,5.6273% of all training records). Only labels change to 7. Original arrays and clean validation/test data remain unchanged. A fresh CNN is trained for each condition and training seed. Poisoned training deliberately bypasses the protected gate to model the vulnerable path.

The notebook's nested selection code is authoritative; the standalone `targeted_label_flip` helper implements the earlier choice-based exploratory trial and is not interchangeable with the final selector. Poisoning targeted error is test ones predicted as 7 divided by all 1,135 test ones, unlike evasion ASR.

## 7. Input-manipulation results

Accepted run: `mnist_adversarial/results/full_run_20261019_212556_9hj6oN/evaluation/metrics.json`. Original GPU results, one training seed:

| Model | Clean accuracy (%) | Clean CE | FGSM .3 accuracy (%) | PGD .3 accuracy (%) |
|---|---:|---:|---:|---:|
| Baseline | 98.84 | .03471524 | 10.99 | 0.00 |
| FGSM-trained | 98.54 | .04631526 | 99.56 | 0.00 |
| PGD-trained | 98.99 | .02897462 | 77.75 | 55.08 |

Baseline attack sweep:

| Attack | Epsilon | Accuracy (%) | CE | ASR (%) |
|---|---:|---:|---:|---:|
| FGSM | .1 | 85.22 | .490997 | 13.7798 |
| FGSM | .2 | 39.54 | 2.409991 | 59.9960 |
| FGSM | .3 | 10.99 | 4.776482 | 88.8810 |
| PGD | .05 | 94.46 | .181305 | 4.4314 |
| PGD | .1 | 73.93 | .912398 | 25.2023 |
| PGD | .2 | 1.45 | 8.312603 | 98.5330 |
| PGD | .3 | 0.00 | 16.994822 | 100.0000 |

Defended attack sweeps:

| Model | Attack | Epsilon | Accuracy (%) | CE | ASR (%) |
|---|---|---:|---:|---:|---:|
| FGSM-trained | FGSM | .1 | 90.02 | .391899 | 8.6767 |
| FGSM-trained | FGSM | .2 | 96.94 | .103964 | 2.3747 |
| FGSM-trained | FGSM | .3 | 99.56 | .011994 | .3044 |
| FGSM-trained | PGD | .3 | 0.00 | 42.024042 | 100.0000 |
| PGD-trained | FGSM | .1 | 96.17 | .116506 | 2.8488 |
| PGD-trained | FGSM | .2 | 90.15 | .319599 | 8.9302 |
| PGD-trained | FGSM | .3 | 77.75 | .725074 | 21.4567 |
| PGD-trained | PGD | .3 | 55.08 | 1.499860 | 44.3580 |

FGSM-trained accuracy rises with epsilon in the FGSM sweep and exceeds clean accuracy at .3, while PGD completely defeats that model. Report this non-monotonic single-step behavior rather than interpreting it as broad robustness. Attack accuracy for one constructed point per image need not be monotonic with epsilon; it is not certified worst-case robust accuracy. The current evidence does not determine the precise cause. PGD training offers better resistance to the tested iterative attack, but one start and one training seed do not establish resistance to all attacks.

## 8. Adversarial-training controls

Both defenses start fresh models with seed42, train five epochs with Adam .001, and save final-epoch weights. They regenerate attacks against the current model for each source batch. Updates use 50/50 clean and attacked examples in a combined batch, with unchanged labels. FGSM training uses epsilon .3. PGD training uses epsilon .3, step .01 and ten iterations with one random start; evaluation uses forty. Evaluation attacks are regenerated against each defended model, not transferred from the clean baseline.

All three saved checkpoints were clean-evaluated again on CPU during integration review: accuracies match 98.84%,98.54%,98.99%. Losses differ slightly from original GPU values due to floating-point/runtime differences; original numbers above are retained. A first-64-image FGSM/PGD smoke check also completed. It is not a repeated full adversarial evaluation and must not replace the accepted run.

## 9. Poisoning results

Mean ± sample standard deviation across training seeds42,43,44, with fixed split and fixed attack selection:

| Condition | Accuracy (%) | CE | Class-1 recall (%) | 1→7 error (%) |
|---|---:|---:|---:|---:|
| Clean | 98.9133 ± .1115 | .0339 ± .0033 | 99.7651 ± .2217 | .0294 ± .0509 |
| 10% | 98.8533 ± .0351 | .0468 ± .0061 | 99.5595 ± .5502 | .1175 ± .1346 |
| 30% | 98.8600 ± .0700 | .0669 ± .0067 | 99.3833 ± .0881 | .3231 ± .0509 |
| 50% | 96.0067 ± 1.3582 | .1081 ± .0090 | 74.6549 ± 12.1245 | 24.8752 ± 11.8417 |

At 50%, every training seed has substantially increased targeted errors (15.6828%–38.2379%). Mean overall accuracy falls 2.9066 percentage points while mean class-1 recall falls 25.1102 points. At lower budgets, overall accuracy barely moves but loss increases in every matched seed. Three seeds quantify training variation only; no significance test or universal poisoning threshold is claimed. Clean validation/checkpoint selection can mitigate observed impact and is part of the defender model.

Individual metrics, confusion matrices and selected epochs are in `results/poisoning/`. All13 saved clean/poisoned/recovered matrices reproduce recorded classification metrics. No full poisoning models were retrained during integration review.

## 10. Dataset-integrity defense and recovery

A clean snapshot records SHA-256 hashes of contiguous images/labels including dtype and shape metadata. Protected training checks counts and both hashes before use, rejecting mismatches. Gate tests accept clean data and reject all three label-poisoned variants; a separate one-pixel modification is rejected. Hashes identify a mismatch, not individual corrupt records, semantic correctness or authenticity of the original source. Folder separation does not itself protect the reference.

The saved recovery log blocks the 30% candidate and loads/verifies the trusted clean snapshot without consulting attack indices. The recovered model is evaluated once, seed 42:

| Condition | Accuracy (%) | CE | Class-1 recall (%) | 1→7 error (%) |
|---|---:|---:|---:|---:|
| Clean | 98.83 | .0375 | 99.7357 | .0881 |
| 50% poisoned | 94.45 | .1170 | 60.9692 | 38.2379 |
| Verified recovery | 98.83 | .0375 | 99.7357 | .0881 |

Historical recovered-model metadata says `rejected_attack=poisoned_050`, inconsistent with the 30% recovery log. Original files are unchanged and a separate clarification records this issue. The table compares model performance; it does not imply a separately logged 50% recovery sequence. All candidates' rejection is independently supported by gate tests.

Recovery restores trusted clean training, not tolerance to poisoned data. References, snapshots and verifier code must be protected; poisoning before snapshot creation or concurrent mutation after verification defeats the assumption. Production controls would require access restrictions, immutable/versioned provenance and verification-to-use protection. Dataset-wide gate outcomes do not establish per-record detection/false-positive rates.

## 11. Inference and control overhead

The adversarial track sums synchronized GPU forward times over the complete test loader, excluding transfer/preprocessing, without explicit warmup/repeated timing. Poisoning times the first 256 images already on CPU after ten warmups, fifty repeats, reporting median/p95. These are different protocols and hardware.

Seed42 CPU median batch times were 108.53 ms clean,117.25ms (10%),115.72ms (30%),101.84ms (50%),106.17ms recovered. Initial single gate checks took approximately 67–137 ms. GPU original single-pass values varied strongly by model/order. None establishes that poisoning or a defense speeds inference. Use stored timing definitions and avoid cross-track speed claims.

## 12. SAST findings, remediation and validation

Bandit 1.9.4 scans current first-party data/model/training/evaluation/attack/defense/utility/test code and the complete current notebook export. Third-party source, historical copies, datasets, weights and results are excluded. Bash runner review is separate. A preliminary export attempt without IPython had a parse error and is retained as failed evidence; the completed before/after scans have no parse errors.

| Stage | Scope | LOW findings |
|---|---|---:|
| Completed combined initial scan |18 original active Python files plus notebook export |13 |
| Updated combined scan |19 active Python files (adds scan utility) plus notebook export |11 |

Four initial B101 warnings identify `assert` in attack-output validation. Assertions can disappear with `python -O`; the full runner guarded against that mode but individual attack entry points did not. The checks now raise explicit ValueError. The two exception tests were updated. Valid-input attack calculations and saved metrics are unchanged; original results refer to original source snapshots, not retroactively to revised code.

The remaining 11 warnings are 3 B404 imports and 8 B603 subprocess calls: notebook Bandit orchestration, verifier unittest execution, and the new scan utility. The final count equals 13 minus 4 resolved assertions plus 2 warnings from the new utility. All are LOW/HIGH-confidence pattern detections. Fixed Python module invocations, argument lists and shell=False (default or explicit) were reviewed; no command injection was identified under the stated trusted-code/executable model. Warnings were not suppressed. This is a review conclusion, not proof against all malicious runtime configurations.

Existing 14 correctness tests pass. Bound rejection remains enabled under optimized Python. Bash syntax and both ignore-file discovery layouts were checked; the full retraining runner was not executed. Raw scans, source hashes, findings reviews, tests and results checks are under `security/sast/combined_review/`.

The saved 18 adversarial confusion matrices and 13 poisoning/recovery matrices match reported accuracies; saved selected epochs match minimum validation loss. Checkpoint hashes match accepted-run hashes and remain unchanged after review evaluation. Loss cannot be recomputed from confusion matrices; saved full-run loss is retained as evidence.

## 13. Limitations and conclusion

This study demonstrates ML01 evasion and ML02 label poisoning, with complementary defenses. PGD training improves accuracy against the tested iterative attack; FGSM training's high single-step score conceals complete PGD failure. Concentrated label poisoning damages class-specific behavior despite a modest global poisoning fraction; protected-snapshot gating and recovery prevent the tested post-snapshot modifications from entering protected training.

Results are limited to MNIST, this architecture, stated attacks and environments. Adversarial training uses one seed and one PGD start. Poisoning uses three training seeds but one split/attack sample. Attack parameter-selection history is not established by saved files and needs confirmation by the experimenter. Privacy and production deployment controls are recommendations, not measured implementations. Dependency-vulnerability scanning, signed artifacts, isolation and tamper-proof auditing are not supplied.

Separate baselines, checkpoint-selection rules and timings are explicitly preserved. Original evidence is retained; no full retraining or repository push occurred in this review. The root README describes verification/reproduction and the original notebook's remaining Colab/Drive dependencies.

## Evidence and references

- Supplied assignment brief, tasks1–7 and deliverables.
- Supplied Class12 notes: STRIDE/LINDDUN and OWASP ML01/ML02/ML06/ML08/ML10 definitions and controls.
- Accepted adversarial run: `mnist_adversarial/results/full_run_20261019_212556_9hj6oN/` (metrics, environment, logs, split and historical code).
- Poisoning/recovery evidence: `results/poisoning/` and executed notebook.
- Combined source review: `security/sast/combined_review/`.
- Full adversarial matrices: `mnist_adversarial/CONFUSION_MATRICES.md`; poisoning matrices are in individual JSON records and PNGs.
