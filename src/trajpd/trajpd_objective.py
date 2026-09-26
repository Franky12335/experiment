"""TrajPD's detached, decision-coupled information projection.

The only constraint preserves the direct teacher KL's full-logit gradient
projection. ``metric`` accounts for grouping unselected vocabulary probability.
All decision reductions run on device; no token/decision Python loops are used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CoupledTargets:
    target_log_probs: torch.Tensor
    token_weights: torch.Tensor
    decision_weights: torch.Tensor
    dual: torch.Tensor
    relative_residual: torch.Tensor
    absolute_residual: torch.Tensor
    bracket_iterations: int
    bisection_iterations: int


@torch.no_grad()
def solve_coupled_targets(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    metric: torch.Tensor,
    decision_ids: torch.Tensor,
    *,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
    max_bracket_iterations: int = 60,
    bisection_iterations: int = 80,
) -> CoupledTargets:
    """Solve min sum KL(Q||S) with w sum (Q-S)'M(T-S) >= sum (T-S)'M(T-S).

    Inputs are aligned [tokens, probability groups] log distributions and a
    positive diagonal metric. Decision IDs must be consecutive starting at zero.
    Compact probability arithmetic and scalar dual solves use float64 even when
    model logits use bf16. Tolerances apply to the dimensionless, direction-scaled
    projection; raw relative residual is also returned for monitoring.

    This is a target construction at the behavior policy, not an autograd path.
    Failure to bracket or satisfy the constraint raises, never swaps in a target.
    """
    if student_log_probs.ndim != 2 or student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("student and teacher log probabilities must be matching matrices")
    if metric.shape != student_log_probs.shape or decision_ids.shape != student_log_probs.shape[:1]:
        raise ValueError("metric and decision IDs must align with probability rows")
    if decision_ids.dtype != torch.long or decision_ids.device != student_log_probs.device:
        raise ValueError("decision IDs must be long tensors on the probability device")
    if teacher_log_probs.device != student_log_probs.device or metric.device != student_log_probs.device:
        raise ValueError("all coupled target inputs must share a device")
    if student_log_probs.shape[0] == 0 or student_log_probs.shape[1] < 2:
        raise ValueError("coupled target solve requires tokens and at least two probability groups")
    if absolute_tolerance < 0 or relative_tolerance <= 0 or max_bracket_iterations < 1 or bisection_iterations < 1:
        raise ValueError("invalid coupled solver tolerances or iteration limits")
    ls, lt, m = student_log_probs.double(), teacher_log_probs.double(), metric.double()
    if not bool((torch.isfinite(ls).all() & torch.isfinite(lt).all() & torch.isfinite(m).all() & (m > 0).all()).item()):
        raise ValueError("coupled target inputs must be finite and metric strictly positive")
    norms = torch.stack((torch.logsumexp(ls, -1), torch.logsumexp(lt, -1)))
    if not bool((norms.abs() <= 1e-5).all().item()):
        raise ValueError("grouped log probabilities must describe normalized full probability mass")
    # Remove only accepted floating-point reconstruction roundoff.
    ls, lt = ls - norms[0, :, None], lt - norms[1, :, None]
    if int(decision_ids.min().item()) != 0:
        raise ValueError("decision IDs must start at zero")
    count = int(decision_ids.max().item()) + 1
    sizes = torch.bincount(decision_ids, minlength=count).double()
    if bool((sizes == 0).any().item()):
        raise ValueError("decision IDs must be consecutive without empty decisions")

    def mean_by_decision(values: torch.Tensor) -> torch.Tensor:
        return torch.zeros(count, device=ls.device, dtype=torch.float64).scatter_add_(0, decision_ids, values) / sizes

    s, t = ls.exp(), lt.exp()
    lm = torch.logaddexp(ls, lt) - math.log(2.0)
    js = .5 * (s * (ls - lm) + t * (lt - lm)).sum(-1)
    weights = 1.0 + (mean_by_decision(js) / math.log(2.0)).clamp(0.0, 1.0)
    r = t - s
    direction = m * r
    scale = torch.zeros(count, device=ls.device, dtype=torch.float64).scatter_reduce_(
        0, decision_ids, direction.abs().amax(-1), reduce="amax", include_self=True
    )
    active = scale > 0
    safe_scale = torch.where(active, scale, torch.ones_like(scale))
    v = direction / safe_scale[decision_ids, None]
    target = mean_by_decision((r * v).sum(-1)) / weights

    def evaluate(dual_scaled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tilt = dual_scaled[decision_ids, None] * v
        tilt = tilt - tilt.amax(-1, keepdim=True)
        # log1p/expm1 avoid cancellation when S and T nearly agree. General
        # logsumexp handles concentrated distributions and very large tilts.
        shift = (s * torch.expm1(tilt)).sum(-1, keepdim=True)
        stable_norm = torch.log1p(shift.clamp(min=-.5))
        log_norm = torch.where(shift > -.5, stable_norm, torch.logsumexp(ls + tilt, -1, keepdim=True))
        log_ratio = tilt - log_norm
        lq = ls + log_ratio
        delta = torch.where(log_ratio.abs() < .5, s * torch.expm1(log_ratio.clamp(-.5, .5)), lq.exp() - s)
        value = mean_by_decision((delta * v).sum(-1))
        return value, lq

    lower = torch.zeros_like(target)
    upper = torch.ones_like(target)
    bracket_steps = 0
    for bracket_steps in range(1, max_bracket_iterations + 1):
        value, _ = evaluate(upper)
        missing = active & (value < target)
        if not bool(missing.any().item()):
            break
        upper = torch.where(missing, upper * 2.0, upper)
    else:
        raise RuntimeError("TrajPD target constraint could not be bracketed; optimizer update must abort")
    # Fixed iterations keep every decision synchronized and remove convergence
    # dependence on partitioning a batch into microbatches.
    for _ in range(bisection_iterations):
        midpoint = (lower + upper) * .5
        value, _ = evaluate(midpoint)
        below = value < target
        lower = torch.where(active & below, midpoint, lower)
        upper = torch.where(active & ~below, midpoint, upper)
    dual_scaled = torch.where(active, (lower + upper) * .5, torch.zeros_like(upper))
    value, log_q = evaluate(dual_scaled)
    residual = (value - target).abs()
    tolerance = absolute_tolerance + relative_tolerance * target.abs()
    if not bool((torch.isfinite(log_q).all() & torch.isfinite(residual).all() & (residual <= tolerance).all()).item()):
        raise RuntimeError("TrajPD target projection residual exceeded tolerance; optimizer update must abort")
    relative = torch.where(target > 0, residual / target.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(target))
    return CoupledTargets(
        target_log_probs=log_q,
        token_weights=weights[decision_ids],
        decision_weights=weights,
        dual=dual_scaled / safe_scale,
        relative_residual=relative,
        absolute_residual=residual * safe_scale,
        bracket_iterations=bracket_steps,
        bisection_iterations=bisection_iterations,
    )
