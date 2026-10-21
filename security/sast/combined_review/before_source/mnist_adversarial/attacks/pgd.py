import torch

try:
    from .sanity import check_adversarial, validate_attack_inputs
except ImportError:
    from sanity import check_adversarial, validate_attack_inputs


def pgd_attack(
    model,
    images,
    labels,
    criterion,
    epsilon: float = 0.3,
    alpha: float = 0.01,
    steps: int = 40,
    random_start: bool = True,
    *,
    verbose: bool = False,
):
    """
    Untargeted L-infinity PGD attack for model inputs in [0, 1].
    """
    validate_attack_inputs(images, epsilon, alpha, steps)
    was_training = model.training
    model.eval()
    try:
        original = images.detach().clone()
        adversarial = original.clone()
        if random_start:
            adversarial = (
                original + torch.empty_like(original).uniform_(-epsilon, epsilon)
            ).clamp(0.0, 1.0)
        check_adversarial(original, adversarial, epsilon)

        for _ in range(steps):
            with torch.enable_grad():
                adversarial = adversarial.detach().requires_grad_(True)
                loss = criterion(model(adversarial), labels)
                gradient, = torch.autograd.grad(loss, adversarial)
            with torch.no_grad():
                adversarial = adversarial + alpha * gradient.sign()
                perturbation = (adversarial - original).clamp(-epsilon, epsilon)
                adversarial = (original + perturbation).clamp(0.0, 1.0)
                check_adversarial(original, adversarial, epsilon)

        adversarial = adversarial.detach()
        check_adversarial(original, adversarial, epsilon, verbose=verbose)
        return adversarial
    finally:
        model.train(was_training)
