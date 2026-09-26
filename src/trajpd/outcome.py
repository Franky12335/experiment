"""Pure task-return control variates for a complete TrajPD rollout update."""
from __future__ import annotations

import torch


@torch.no_grad()
def outcome_advantages(returns: torch.Tensor) -> torch.Tensor:
    """Cross-prompt leave-one-out return reference; no length cost or reward rescaling."""
    if returns.ndim != 1 or returns.numel() < 2:
        raise ValueError("outcome advantages require at least two aligned trajectories")
    if not bool(torch.isfinite(returns).all()) or not bool(((returns >= 0) & (returns <= 1)).all()):
        raise ValueError("task returns must be finite and in [0, 1]")
    values = returns.to(torch.float64)
    other_trajectory_count = values.numel() - 1
    other_return_sum = values.sum() - values
    leave_one_out_reference = other_return_sum / other_trajectory_count
    advantages = values - leave_one_out_reference
    return advantages.to(torch.float32)
