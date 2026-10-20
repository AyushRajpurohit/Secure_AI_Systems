import torch

try:
    from .sanity import check_adversarial, validate_attack_inputs
except ImportError:
    from sanity import check_adversarial, validate_attack_inputs


def fgsm_attack(
    model,
    images,
    labels,
    criterion,
    epsilon: float,
    *,
    verbose: bool = False,
):
    """
    Untargeted L-infinity FGSM attack for model inputs in [0, 1].
    """
    validate_attack_inputs(images, epsilon)
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            x = images.detach().clone().requires_grad_(True)
            loss = criterion(model(x), labels)
            gradient, = torch.autograd.grad(loss, x)
            perturbed = (x + epsilon * gradient.sign()).clamp(0.0, 1.0).detach()
        check_adversarial(images, perturbed, epsilon, verbose=verbose)
        return perturbed
    finally:
        model.train(was_training)
