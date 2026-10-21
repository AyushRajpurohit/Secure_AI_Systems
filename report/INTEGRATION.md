# Integration review — 21 October 2026

Authoritative combined narrative: [REPORT.md](REPORT.md). Earlier component reports describe their original runs and can contain superseded integration status.

## Completed

- Both adversarial defenses and all recorded FGSM/PGD experiments are present.
- Exact saved train/validation index order agrees across both tracks.
- Original checkpoints match accepted-run SHA-256 hashes.
- All 18 accepted adversarial confusion matrices and 13 poisoning/recovery confusion matrices reproduce their recorded classification metrics.
- Clean evaluation of all three saved adversarial models was rerun on CPU; accuracies match the original GPU run. Original metrics were not overwritten. CPU losses differ slightly from GPU floating-point values.
- FGSM/PGD smoke checks on the first 64 test images completed for all models; these are not replacements for the original full attack evaluation.
- Existing 14 correctness tests pass after replacing disabled-under-optimization attack assertions with explicit ValueError checks and updating the tests' expected exceptions.
- Bound rejection also checked with `python -O`.
- `run_all.sh` discovers ignore rules in either standalone or merged layout and snapshots the root ignore file without escaping the source snapshot directory.
- Combined before/after Bandit evidence and manual warning review are under `security/sast/combined_review/`.
- Root README and reproducible source-scan utility are supplied.

## Evidence clarification

`recovery_evidence.json` records the 30% candidate. `defended_recovery_seed42.json` contains a contradictory 50% rejected-attack label. Original JSON files were preserved; the correction is documented in `security/sast/combined_review/recovery_metadata_clarification.json`. Supported claims: all candidates were blocked in gate tests; the recorded 30% recovery sequence restored clean data; recovered seed-42 training restored baseline performance.

## Intentional limitations

The executed tracks have distinct trainers, shuffle behavior, checkpoints and timing methods. They are documented as two experiment tracks using the same architecture and exact split. Refactoring all executed paths is not necessary to report these results and must not be represented as already complete.

The original Colab notebook retains inline definitions and Drive paths. Root README explains its reproduction requirements. Its later package smoke tests are not evidence of a complete rerun through the shared trainer.

Original results, source snapshots, datasets and checkpoints have not been silently replaced. No models were retrained during this review. The full training runner was syntax/path checked, not executed end to end.

## Before submission

- Confirm whether attack settings were fixed beforehand or adjusted after inspecting test results; saved files do not establish that history.
- Review the combined report with both teammates and add required names/identifiers.
- Retain/share original poisoning artifacts from Drive if requested by the instructor.
- Apply reviewed file changes to the shared repository, then commit. A GitHub push was not performed by this offline ZIP review.
- Further source edits require a new final scan. Claims of broader PGD robustness require additional attack evaluation, not just report wording changes.
