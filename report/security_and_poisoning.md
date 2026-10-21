> Historical component draft. The authoritative integrated report is [REPORT.md](REPORT.md). Its corrected paths, later smoke-test status, expanded scan and combined scope supersede older statements below.

# Secure MNIST: security engineering and training-data integrity

## 1. System and experimental design

The Colab application trains a CNN to classify MNIST digits. Data flows from the external MNIST source into local arrays, a fixed split, CNN training, saved weights and held-out evaluation. The experimental application has no public prediction API and no automatic feedback-based retraining.

Assets include dataset images and labels, split indices, source code, the trusted hash reference, the clean recovery snapshot, model checkpoints, experiment records and any credentials/storage accessible to the runtime. Trust boundaries occur when external packages/data enter the runtime, candidate data enters training, stored checkpoints enter evaluation, and notebook artifacts are shared. Google Drive persistence is used; folder names alone do not enforce security boundaries.

The official 60,000 MNIST training examples were split into 55,000 training and 5,000 validation examples with seed 42. The official 10,000-example test set was retained. Pixels were converted to float32 in [0,1]. The CNN uses 1→32 and 32→64 convolutions (3×3, padding 1), each followed by ReLU and 2×2 max pooling, then 3136→128→10 linear layers with an intermediate ReLU. Outputs are logits. Training uses Adam, learning rate 0.001, batch size 64 and five epochs. The checkpoint with lowest clean validation loss is evaluated. Evaluation batch size is 256.

The three training seeds are 42, 43 and 44. The split and attack selections remain fixed. Reported sample standard deviations therefore describe training-seed variability, not variability across data splits or independently sampled attacks. Seed controls do not guarantee identical results across hardware/library environments.

## 2. Attacker model

The attacker can alter a bounded subset of labels in a training-data working copy. The objective is to make genuine digit 1 test images be predicted as 7. Image bytes are unchanged. The attacker knows the dataset structure and pipeline, but cannot modify validation/test data, training code, verification code, the trusted reference or recovery snapshot. This models post-snapshot data tampering, not an attacker controlling the trusted source or Colab account.

This attack exercises ML02:2023 Data Poisoning: training labels are modified before learning. It is distinct from ML01 input manipulation at inference, ML10 direct modification of model artifacts, and the class notes' ML08 feedback-loop skewing scenario.

## 3. STRIDE threat model

Potential threats are separated from demonstrated attacks. Only the specified label-poisoning attack and integrity controls were experimentally evaluated here.

| STRIDE | System-specific threat | OWASP ML association | Control and status |
|---|---|---|---|
| Spoofing | Substitute data or checkpoints while claiming trusted origin | ML06; ML10 for a malicious model artifact | Provenance/source authentication and trusted artifact identity are recommended; source authentication was not independently implemented |
| Tampering | Alter training labels or image bytes before training | ML02 | Image/label hash verification and blocking gate implemented |
| Repudiation | Deny changing a dataset or producing a checkpoint | ML02/ML10 context | Seeds, configurations, hashes and run records retained; these are not tamper-proof audit logs |
| Information disclosure | Expose credentials, artifacts or sensitive future submissions through sharing; infer private training information | ML05 for model exposure; ML03/ML04 for inference attacks | Restricted sharing, secret management and minimal outputs recommended; inversion/membership attacks not tested |
| Denial of service | Exhaust compute, memory or storage with excessive workloads | No universal direct mapping; ML01 can apply to malicious inference inputs | Workload bounds and input checks recommended; not experimentally evaluated |
| Elevation of privilege | Untrusted code or unsafe artifact loading accesses runtime files or mounted storage | ML06 | Review code and loaders, constrain permissions and dependencies; runtime isolation not validated |

The same trusted validation data used for checkpoint selection may limit poisoning impact. This is part of the defender's modeled access, not an attacker capability.

## 4. LINDDUN privacy discussion

MNIST is public; this experiment does not establish exposure of identifiable customer records. The following concerns apply to a hypothetical real-world digit-recognition service processing forms.

| Category | Potential concern | Recommended control |
|---|---|---|
| Linkability | Join requests through user/document identifiers | Minimize persistent identifiers and separate identity data from prediction logs |
| Identifiability | Identify submitters from document context or metadata | Crop to needed content and remove unnecessary metadata |
| Non-repudiation | Persistent logs bind a person to a sensitive submission | Limit log detail, access and retention |
| Detectability | Infer participation or the existence of a record | Assess membership-inference exposure and minimize unnecessary outputs |
| Disclosure | Expose uploaded documents or logs | Enforce access controls and protected storage |
| Unawareness | Users do not understand retention or training reuse | Clearly explain collection, retention and reuse |
| Non-compliance | Processing conflicts with policy, consent or applicable obligations | Define purpose, retention and deletion procedures |

Privacy non-repudiation concerns excessive evidence of an association; security repudiation concerns insufficient accountability. These controls are recommendations, not implemented production claims.

## 5. Red-team implementation

There are 6,190 digit-1 examples in the training split. An attack-seed-42 permutation supplies nested selections: the 10% set is included in the 30% set, which is included in the 50% set. Selected labels change from 1 to 7. Fractions refer to source-class examples, not all training examples.

| Source-class budget | Modified labels | Entire training set |
|---|---:|---:|
| 10% | 619 | 1.1255% |
| 30% | 1,857 | 3.3764% |
| 50% | 3,095 | 5.6273% |

The original standalone 30% trial used a different sampling operation. Final classification comparisons use the nested attack files. The reusable `targeted_label_flip` helper reproduces the standalone choice-based trial; the notebook's permutation-based selection is authoritative for the reported nested experiments.

Every condition trains a fresh CNN. Poisoned runs intentionally bypass the integrity gate to measure an unprotected pipeline. Test images and labels remain clean. The targeted error rate is the number of true test ones predicted as seven divided by all true test ones (1,135 examples). Overall accuracy, mean cross-entropy loss, class-1 recall and confusion matrices supplement that metric.

## 6. Classification results

Mean ± sample standard deviation over three training seeds:

| Condition | Accuracy (%) | Loss | Class-1 recall (%) | 1→7 error (%) |
|---|---:|---:|---:|---:|
| Clean | 98.9133 ± 0.1115 | 0.0339 ± 0.0033 | 99.7651 ± 0.2217 | 0.0294 ± 0.0509 |
| 10% | 98.8533 ± 0.0351 | 0.0468 ± 0.0061 | 99.5595 ± 0.5502 | 0.1175 ± 0.1346 |
| 30% | 98.8600 ± 0.0700 | 0.0669 ± 0.0067 | 99.3833 ± 0.0881 | 0.3231 ± 0.0509 |
| 50% | 96.0067 ± 1.3582 | 0.1081 ± 0.0090 | 74.6549 ± 12.1245 | 24.8752 ± 11.8417 |

At 50%, targeted errors increased in every seed, ranging from 15.6828% to 38.2379%. Mean accuracy fell 2.9066 percentage points and mean class-1 recall fell 25.1102 points relative to clean training. Lower poisoning budgets barely changed overall accuracy, but increased loss in every matched seed. Three seeds do not establish a universal attack threshold or formal statistical significance.

All 12 saved confusion matrices independently reproduce their recorded accuracy, class-1 recall and targeted-error metrics. Selected epochs also match minimum recorded validation loss. Loss values are taken from the saved records; predictions/logits were not supplied for independent loss recomputation.

## 7. Blue-team integrity control and recovery

A clean reference records SHA-256 digests of contiguous image and label arrays, including dtype and shape metadata. The gate checks counts and hashes before protected training. It accepts unchanged data and rejects all three tested label-poisoned variants. A separate one-pixel change was also rejected in the notebook. Aggregate hashes detect a mismatch; they do not identify individual corrupted records or establish a population-level false-positive rate.

The saved recovery log explicitly demonstrates rejection of the 30% candidate followed by loading and verification of the clean snapshot, without using attack indices. A separate recovered-data model was trained with seed 42. Although its metadata names the 50% attack, that metadata is not a separate recorded 50%-candidate block/recovery sequence. All three candidate blocks are independently recorded in the integrity-gate results.

| Seed-42 condition | Accuracy (%) | Loss | Class-1 recall (%) | 1→7 error (%) |
|---|---:|---:|---:|---:|
| Clean | 98.83 | 0.0375 | 99.7357 | 0.0881 |
| 50% poisoned | 94.45 | 0.1170 | 60.9692 | 38.2379 |
| Verified recovery | 98.83 | 0.0375 | 99.7357 | 0.0881 |

Recovery restored baseline performance using clean data and the same training settings. This is prevention/recovery from tampering, not a learner trained to tolerate poisoned labels. Recovery was evaluated for one training seed. The control cannot detect poisoning already present when the trusted reference is created, or survive an attacker who controls the reference, verifier or training process. Verification-to-use integrity also assumes no concurrent modification after checking; immutable data access would be needed in production.

Initial verification measurements were 67–137 ms across single checks. These are descriptive observations, not a reliable timing comparison. Seed-42 median forward-pass batch times were 108.53 ms clean, 117.25 ms (10%), 115.72 ms (30%), 101.84 ms (50%), and 106.17 ms recovered. The benchmark uses a batch of 256 already on-device, ten warmups and fifty measured runs. It excludes preprocessing and transfers. Timing variation does not demonstrate poisoning-induced speed changes.

## 8. SAST and review

Bandit 1.9.4 initially reported zero findings in the two exported modules. The initial 40-code-cell notebook export had five LOW-severity findings: one B404 import warning and four B603 subprocess warnings. Review identified fixed Python/Bandit invocations with controlled argument lists and default shell=False. No exploitable command injection was identified under the attacker model; no vulnerability was deliberately added or warning suppressed.

The final review additionally scanned the later uploaded notebook, which includes scanner orchestration cells added after the initial snapshot. It reports seven LOW-severity findings: one B404 and six B603. The extra two calls retrieve Bandit's version and launch the notebook scan. They have the same controlled invocation pattern. This is a scope expansion, not a contradiction of the original five findings. Final reusable modules have zero findings.

Raw scans, review decisions, source snapshots and hashes are preserved. Notebook magic/shell lines are comments in the final review export and are excluded from direct Python scanning. SAST does not validate model robustness, package vulnerabilities, cloud permissions or the trusted reference's authenticity.

## 9. Packaging changes and validation limits

Original evidence is preserved under `evidence/`; the original notebook retains historical outputs. A new `src/mnist_training.py` makes config, device and validation_loader explicit keyword arguments instead of relying on notebook globals. No training algorithm or model architecture was changed. The revised modules passed syntax checks and Bandit. PyTorch is absent in the review environment, so the revised training interface has not been runtime-tested here and the recorded results belong to the original notebook implementation. A short Colab smoke-check is supplied in README.md.

The supplied ZIP omits the actual attack NPZ files, trusted snapshot/reference and model checkpoints. Their byte-level identity cannot be independently confirmed from this upload. Keep those original Drive files and add the attack artifacts or reproducible generation code to the submitted repository. The original notebook contains generation code and saved split indices are included.

## 10. Conclusion and integration

Concentrated label poisoning can substantially damage one class while affecting only a small fraction of the entire training set. Trusted-snapshot integrity checks blocked the tested modifications, and clean recovery restored the seed-42 baseline. Integrity controls complement adversarial training: they address training-data tampering rather than inference-time perturbations.
