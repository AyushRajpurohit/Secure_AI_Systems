import math

import torch


def validate_attack_inputs(images, epsilon, alpha=None, steps=None):
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if alpha is not None and (not math.isfinite(alpha) or alpha < 0):
        raise ValueError("alpha must be finite and nonnegative")
    if steps is not None and (not isinstance(steps, int) or steps < 0):
        raise ValueError("steps must be a nonnegative integer")
    if not images.is_floating_point() or images.numel() == 0:
        raise ValueError("images must be a nonempty floating-point tensor")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("images must contain finite pixel values in [0, 1]")


@torch.no_grad()
def check_adversarial(images, adversarial, epsilon, *, verbose=False, tolerance=1e-6):
    """Check pixel-space bounds and return batch statistics."""
    if adversarial.shape != images.shape:
        raise ValueError("Adversarial shape changed")
    if not torch.isfinite(adversarial).all():
        raise ValueError("Nonfinite adversarial pixels")
    max_linf = (adversarial - images).abs().max().item()
    pixel_min = adversarial.min().item()
    pixel_max = adversarial.max().item()
    if max_linf > epsilon + tolerance:
        raise ValueError("L-infinity bound exceeded")
    if not 0.0 <= pixel_min <= pixel_max <= 1.0:
        raise ValueError("Pixels outside [0, 1]")
    if verbose:
        print(
            f"Attack sanity: max_linf={max_linf:.8f}, "
            f"pixel_min={pixel_min:.8f}, pixel_max={pixel_max:.8f}"
        )
    return max_linf, pixel_min, pixel_max
