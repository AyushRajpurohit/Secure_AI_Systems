"""Training-data poisoning and integrity verification utilities."""


import hashlib
import json
import numpy as np


class DatasetIntegrityError(RuntimeError):
    """Raised when candidate training data fails verification."""
    pass


def array_digest(array):
    """Hash shape, dtype, and canonical array bytes."""
    array = np.ascontiguousarray(array)

    metadata = json.dumps(
        {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def targeted_label_flip(
    clean_labels,
    source_class=1,
    target_class=7,
    fraction=0.30,
    seed=42,
):
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")

    if source_class == target_class:
        raise ValueError("Source and target classes must differ.")

    eligible_indices = np.flatnonzero(
        clean_labels == source_class
    )

    number_to_flip = int(
        np.floor(fraction * len(eligible_indices))
    )

    rng = np.random.default_rng(seed)

    selected_indices = np.sort(
        rng.choice(
            eligible_indices,
            size=number_to_flip,
            replace=False,
        )
    )

    poisoned_labels = clean_labels.copy()
    poisoned_labels[selected_indices] = target_class

    return poisoned_labels, selected_indices


def verify_training_data(images, labels, trusted_reference):
    checks = {
        "image_count_matches": (
            len(images) == trusted_reference["train_size"]
        ),
        "label_count_matches": (
            len(labels) == trusted_reference["train_size"]
        ),
        "images_match_trusted_snapshot": (
            array_digest(images)
            == trusted_reference["train_images_sha256"]
        ),
        "labels_match_trusted_snapshot": (
            array_digest(labels)
            == trusted_reference["train_labels_sha256"]
        ),
    }

    return {
        "accepted": all(checks.values()),
        "checks": checks,
    }


def require_verified_training_data(images, labels, trusted_reference):
    result = verify_training_data(
        images, labels, trusted_reference
    )

    if not result["accepted"]:
        failed = [
            name
            for name, passed in result["checks"].items()
            if not passed
        ]

        raise DatasetIntegrityError(
            "Training blocked. Failed checks: " + ", ".join(failed)
        )

    return result
