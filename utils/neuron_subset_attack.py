"""Gradient-ranked input-channel PGD for tensors shaped (batch, time, channels).

L2 radii are per (sample, time), matching this project's original trainer.
The ranking is a sensitivity heuristic for the attack loss, not neuron decoding
importance. Padding is excluded from both ranking and perturbation.
"""
import math

import torch


def validate_neuron_options(percent, refresh):
    if isinstance(percent, bool):
        raise ValueError("adv_neuron_percent must be a number in (0, 100].")
    try:
        percent = float(percent)
    except (TypeError, ValueError):
        raise ValueError("adv_neuron_percent must be a number in (0, 100].") from None
    if not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError("adv_neuron_percent must be finite and in (0, 100].")
    if refresh not in ("step", "batch"):
        raise ValueError("adv_neuron_refresh must be 'step' or 'batch'.")
    return percent


def valid_time_mask(x, lengths):
    if x.ndim != 3 or min(x.shape) < 1:
        raise ValueError("Expected a nonempty (batch, time, channels) input.")
    lengths = torch.as_tensor(lengths, device=x.device)
    if lengths.ndim != 1 or lengths.shape[0] != x.shape[0]:
        raise ValueError("Expected one input length per sample.")
    if (not torch.isfinite(lengths).all().item()
            or (lengths != lengths.long()).any().item()
            or ((lengths < 1) | (lengths > x.shape[1])).any().item()):
        raise ValueError("Input lengths must be integers between 1 and input time length.")
    return (torch.arange(x.shape[1], device=x.device)[None, :, None]
            < lengths[:, None, None])


def top_neuron_mask(grad, valid, count, norm):
    """Return (B,1,C) mask: top-k channels separately for each sample.

    Score = temporal L2 gradient norm for l2, temporal L1 norm for linf.
    All valid times of a selected channel are eligible for perturbation.
    """
    grad = torch.where(valid, grad.detach().float(), 0.0)
    if not torch.isfinite(grad).all().item():
        raise FloatingPointError("Nonfinite input gradient while selecting channels.")
    if norm == "l2":
        scores = grad.norm(p=2, dim=1)
    elif norm == "linf":
        scores = grad.abs().sum(dim=1)
    else:
        raise ValueError("adv_norm must be 'l2' or 'linf'.")
    indices = scores.topk(count, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return mask[:, None, :]


def project_delta(delta, support, epsilon, norm):
    """Hard-mask FIRST, then project; prevents stale channels accumulating."""
    delta = torch.where(support, delta, 0.0)
    if norm == "linf":
        return delta.clamp(min=-epsilon, max=epsilon)
    if norm == "l2":
        length = delta.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        return delta * (epsilon / length).clamp(max=1.0)
    raise ValueError("adv_norm must be 'l2' or 'linf'.")


def random_delta(x, support, epsilon, norm):
    if norm == "linf":
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    elif norm == "l2":
        noise = torch.where(support, torch.randn_like(x), 0.0)
        length = noise.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
        radius = torch.rand((x.shape[0], x.shape[1], 1), device=x.device,
                            dtype=x.dtype) * epsilon
        delta = noise / length * radius
    else:
        raise ValueError("adv_norm must be 'l2' or 'linf'.")
    return project_delta(delta, support, epsilon, norm)


def input_gradient(loss_fn, x):
    """Compute only d(loss)/d(input), without accumulating parameter .grad."""
    with torch.enable_grad():
        probe = x.detach().requires_grad_(True)
        loss = loss_fn(probe)
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError("Expected a finite scalar attack loss.")
        grad = torch.autograd.grad(loss, probe, only_inputs=True)[0]
    if not torch.isfinite(grad).all().item():
        raise FloatingPointError("Nonfinite attack gradient.")
    return grad.detach()


def neuron_subset_pgd(x, lengths, loss_fn, epsilon, norm, percent,
                      refresh="step", steps=10, alpha=None):
    """Return detached adversarial input and its final (B,1,C) channel mask.

    One clean-input probe selects the random-start support. 'step' reranks on
    each PGD gradient; 'batch' keeps that first mask for the whole attack.
    The model's train/eval mode is unchanged. No model optimizer step is made.
    """
    percent = validate_neuron_options(percent, refresh)
    if norm not in ("l2", "linf"):
        raise ValueError("adv_norm must be 'l2' or 'linf'.")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("adv_eps must be finite and nonnegative.")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    alpha = epsilon / 5.0 if alpha is None else alpha
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative.")
    x = x.detach()
    valid = valid_time_mask(x, lengths)
    count = max(1, math.ceil(x.shape[-1] * percent / 100.0))
    grad = input_gradient(loss_fn, x)
    mask = top_neuron_mask(grad, valid, count, norm)
    with torch.no_grad():
        support = valid & mask
        delta = random_delta(x, support, epsilon, norm)
        adv = torch.where(support, x + delta, x)
    for _ in range(steps):
        grad = input_gradient(loss_fn, adv)
        with torch.no_grad():
            if refresh == "step":
                mask = top_neuron_mask(grad, valid, count, norm)
            support = valid & mask
            grad = torch.where(support, grad, 0.0)
            if norm == "linf":
                direction = grad.sign()
            else:
                direction = grad / grad.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            # Re-ranking can remove channels; clear their old perturbations too.
            delta = project_delta(adv - x + alpha * direction, support, epsilon, norm)
            adv = torch.where(support, x + delta, x)
    return adv.detach(), mask.detach()
