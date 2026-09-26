"""TrajPD: outcome policy gradient and coupled, decision-level teacher projection.

Only assistant actions contribute. Both terms use token sums and the same fixed
numerical scale; global batch normalization is applied exactly once in the reducer.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from verl.utils import tensordict_utils
from verl.utils.metric import AggregationType, Metric

from src.trajpd.schema import (
    DECISION_INDEX, OUTCOME_ADVANTAGE, TEACHER_TOPK_LOG_PROBS,
    TEACHER_TOPK_TOKEN_IDS,
)
from src.trajpd.trajpd_objective import solve_coupled_targets


def _require_tensor(condition: torch.Tensor, message: str) -> None:
    if condition.numel() != 1:
        raise ValueError("runtime checks must reduce to one value")
    if condition.device.type == "cpu":
        if not bool(condition.item()):
            raise ValueError(message)
    else:
        torch._assert_async(condition, message)


class _GroupedLogProbs(torch.autograd.Function):
    """Exact selected-token and unselected-mass log probabilities.

Backward recomputes chunked FP32 softmax instead of retaining a dense FP32
vocabulary tensor. Unselected mass and its conditional second moment are computed
directly, never by subtracting nearly equal probability sums. The sampled token
may also occur in the teacher selection, so its gradient uses scatter_add_.
"""

    @staticmethod
    def forward(ctx: Any, logits: torch.Tensor, teacher_ids: torch.Tensor,
                sampled_ids: torch.Tensor, chunk_size: int):
        if logits.ndim != 2 or teacher_ids.ndim != 2:
            raise ValueError("logits and selected ids must be matrices")
        n, vocab = logits.shape
        if teacher_ids.shape[0] != n or sampled_ids.shape != (n,):
            raise ValueError("token ids must align with logits")
        if not 0 < teacher_ids.shape[1] < vocab or chunk_size < 1:
            raise ValueError("selection must leave unselected tokens; chunk size must be positive")
        for ids in (teacher_ids, sampled_ids):
            if ids.dtype != torch.long or ids.device != logits.device:
                raise ValueError("token ids must be long tensors on the logits device")
            _require_tensor(((ids >= 0) & (ids < vocab)).all(), "token id outside vocabulary")
        ordered = teacher_ids.sort(dim=-1).values
        _require_tensor((ordered[:, 1:] != ordered[:, :-1]).all(), "duplicate teacher token ids")
        k = teacher_ids.shape[1]
        partition = torch.empty((n, k + 1), device=logits.device, dtype=torch.float32)
        sampled = torch.empty(n, device=logits.device, dtype=torch.float32)
        moment = torch.empty(n, device=logits.device, dtype=torch.float32)
        for lo in range(0, n, chunk_size):
            hi = min(lo + chunk_size, n)
            z = logits[lo:hi].float()
            _require_tensor(torch.isfinite(z).all(), "student logits must be finite")
            z = z - z.amax(dim=-1, keepdim=True)
            normalizer = torch.logsumexp(z, -1)
            partition[lo:hi, :k] = z.gather(1, teacher_ids[lo:hi]) - normalizer[:, None]
            sampled[lo:hi] = z.gather(1, sampled_ids[lo:hi, None]).squeeze(1) - normalizer
            unselected = z.clone()
            unselected.scatter_(1, teacher_ids[lo:hi], -torch.inf)
            unselected_lse = torch.logsumexp(unselected, -1)
            partition[lo:hi, k] = unselected_lse - normalizer
            # Shift before doubling: avoids cancellation between large logsumexp values.
            conditional_logp = unselected - unselected_lse[:, None]
            moment[lo:hi] = torch.logsumexp(2 * conditional_logp, -1).exp()
        ctx.save_for_backward(logits, teacher_ids, sampled_ids)
        ctx.chunk_size = chunk_size
        ctx.mark_non_differentiable(moment)
        return partition, sampled, moment

    @staticmethod
    def backward(ctx: Any, grad_partition: torch.Tensor | None,
                 grad_sampled: torch.Tensor | None, _grad_moment: torch.Tensor | None):
        logits, teacher_ids, sampled_ids = ctx.saved_tensors
        n = logits.shape[0]
        grad = torch.empty_like(logits)
        for lo in range(0, n, ctx.chunk_size):
            hi = min(lo + ctx.chunk_size, n)
            z = logits[lo:hi].float()
            z = z - z.amax(dim=-1, keepdim=True)
            gp = (grad_partition[lo:hi].float() if grad_partition is not None
                  else z.new_zeros((hi-lo, teacher_ids.shape[1] + 1)))
            gs = grad_sampled[lo:hi].float() if grad_sampled is not None else z.new_zeros(hi-lo)
            gz = -torch.softmax(z, -1) * (gp.sum(-1) + gs)[:, None]
            gz.scatter_add_(1, teacher_ids[lo:hi], gp[:, :-1])
            unselected = z.clone()
            unselected.scatter_(1, teacher_ids[lo:hi], -torch.inf)
            gz.add_(torch.softmax(unselected, -1) * gp[:, -1:])
            gz.scatter_add_(1, sampled_ids[lo:hi, None], gs[:, None])
            grad[lo:hi] = gz.to(logits.dtype)
        return grad, None, None, None


def compute_grouped_student_log_probs(logits: torch.Tensor, teacher_ids: torch.Tensor,
                                     sampled_ids: torch.Tensor, *, chunk_size: int = 256):
    return _GroupedLogProbs.apply(logits, teacher_ids, sampled_ids, chunk_size)


def build_teacher_topk_partition_log_probs(topk_log_probs: torch.Tensor):
    """Reconstruct the teacher partition, auditing finite-precision mass repair.

Teacher servers return rounded selected log probabilities. If their sum rounds
to or slightly above one, reserve one FP32 epsilon of unselected mass and
rescale the selected mass. Larger inconsistencies abort. The returned correction
is logged; the projection guarantee concerns this explicitly reconstructed target.
"""
    if topk_log_probs.ndim != 2 or topk_log_probs.shape[1] < 1:
        raise ValueError("teacher log probabilities must have a nonempty selection axis")
    values = topk_log_probs.detach().double()
    _require_tensor(torch.isfinite(values).all(), "teacher log probabilities must be finite")
    mass_log = torch.logsumexp(values, -1, keepdim=True)
    _require_tensor((mass_log <= math.log1p(1e-4)).all(), "teacher selected probability mass exceeds one")
    boundary = math.log1p(-torch.finfo(torch.float32).eps)
    repaired_mass_log = mass_log.clamp_max(boundary)
    correction = (mass_log.exp() - repaired_mass_log.exp()).squeeze(-1)
    selected = values + repaired_mass_log - mass_log
    unselected = torch.log(-torch.expm1(repaired_mass_log))
    return torch.cat((selected, unselected), -1), correction


def compute_forward_kl(target_log_probs: torch.Tensor, student_log_probs: torch.Tensor):
    if target_log_probs.shape != student_log_probs.shape:
        raise ValueError("teacher and student partition shapes differ")
    # FP64 only on the compact partition; preserve the analytical FKL gradient.
    values = (target_log_probs.exp() * (target_log_probs - student_log_probs.double())).sum(-1)
    _require_tensor(torch.isfinite(values).all(), "nonfinite forward KL")
    _require_tensor((values >= -1e-5).all(), "forward KL negative beyond rounding tolerance")
    return values  # Never clamp the loss: a clamp would silently zero its gradient.


LOSS_NAMES = ("joint_loss_contributions", "outcome_loss_contributions", "distillation_loss_contributions")
TOKEN_METRICS = ("teacher_fkl", "target_fkl", "decision_weight", "dual", "projection_relative_residual",
                 "projection_absolute_residual", "solver_bracket_iterations", "solver_bisection_iterations",
                 "unselected_conditional_second_moment", "teacher_mass_repair", "teacher_selected_mass",
                 "student_selected_mass")


def _metadata(data: Any, name: str):
    value = tensordict_utils.get_non_tensor_data(data=data, key=name, default=None)
    if value is None:
        raise ValueError(f"TrajPD requires explicit metadata {name!r}")
    return value


def compute_active_response_outputs(*, student_logits: torch.Tensor, data: Any,
                                    selected_prediction_positions: torch.Tensor):
    if student_logits.ndim != 3 or student_logits.shape[0] != 1:
        raise ValueError("student logits must be [1, active_tokens, vocab]")
    logits = student_logits.squeeze(0)
    alpha = float(_metadata(data, "distillation_alpha"))
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("distillation_alpha must be finite and positive")
    keys = ("input_ids", "responses", "response_mask", TEACHER_TOPK_TOKEN_IDS,
            TEACHER_TOPK_LOG_PROBS, DECISION_INDEX, OUTCOME_ADVANTAGE)
    rows = list(zip(*(data[key].unbind() for key in keys), strict=True))
    selected_ids, sampled_ids, positions, active_rows = [], [], [], []
    sequence_offset = 0
    for ids, response, mask, teacher_ids, _, _, _ in rows:
        active = torch.nonzero(mask.bool(), as_tuple=False).flatten()
        if active.numel() == 0:
            raise ValueError("training trajectory has no assistant action")
        prompt_length = ids.numel() - response.numel()
        if prompt_length < 1:
            raise ValueError("training trajectory must have a nonempty prompt")
        active_rows.append(active)
        positions.append(sequence_offset + prompt_length - 1 + active)
        sequence_offset += ids.numel()
        selected_ids.append(teacher_ids[active].to(logits.device, dtype=torch.long))
        sampled_ids.append(response[active].to(logits.device, dtype=torch.long))
    expected = torch.cat(positions).to(logits.device, dtype=torch.long)
    if selected_prediction_positions is None or not torch.equal(expected, selected_prediction_positions.to(logits.device)):
        raise ValueError("selected logits must cover every assistant prediction exactly once")
    if expected.numel() != logits.shape[0]:
        raise ValueError("active logits count differs from assistant action count")
    student_logp, sampled_logp, second_moment = compute_grouped_student_log_probs(
        logits, torch.cat(selected_ids), torch.cat(sampled_ids))
    losses = {name: [] for name in LOSS_NAMES}
    metrics = {name: [] for name in TOKEN_METRICS}
    offset = 0
    for (ids, response, _, teacher_ids, teacher_logp, decisions, advantage), active in zip(rows, active_rows, strict=True):
        n = active.numel()
        sl = slice(offset, offset + n)
        s = student_logp[sl]
        t, mass_repair = build_teacher_topk_partition_log_probs(teacher_logp[active].to(logits.device))
        decision_ids = decisions[active].to(logits.device, dtype=torch.long)
        metric = torch.ones_like(s)
        metric[:, -1] = second_moment[sl]
        result = solve_coupled_targets(
            s.detach(), t, metric, decision_ids,
            absolute_tolerance=float(_metadata(data, "solver_atol")),
            relative_tolerance=float(_metadata(data, "solver_rtol")),
            max_bracket_iterations=int(_metadata(data, "solver_max_bracket_iterations")),
            bisection_iterations=int(_metadata(data, "solver_max_iterations")))
        advantage = advantage.to(logits.device, dtype=torch.float32).detach().reshape(())
        _require_tensor(torch.isfinite(advantage), "outcome advantage must be finite")
        outcome = -advantage * sampled_logp[sl]
        fkl = compute_forward_kl(result.target_log_probs, s)
        distill = result.token_weights * fkl
        row_positions = (ids.numel() - response.numel() - 1 + active).to(logits.device)
        for name, value in zip(LOSS_NAMES, (outcome + alpha * distill, outcome, distill), strict=True):
            losses[name].append(value.new_zeros(ids.numel()).index_copy(0, row_positions, value))
        stats = (compute_forward_kl(t, s.detach()), fkl.detach(), result.token_weights,
                 result.dual[decision_ids], result.relative_residual[decision_ids].abs(),
                 result.absolute_residual[decision_ids].abs(),
                 s.new_full((n,), result.bracket_iterations), s.new_full((n,), result.bisection_iterations),
                 second_moment[sl], mass_repair, t[:, :-1].exp().sum(-1),
                 s[:, :-1].detach().exp().sum(-1))
        for name, value in zip(TOKEN_METRICS, stats, strict=True):
            metrics[name].append(value.detach().sum())
        offset += n
    return {**{name: torch.cat(value).unsqueeze(0) for name, value in losses.items()},
            **{name + "_contributions": torch.stack(value).unsqueeze(0) for name, value in metrics.items()}}


def compute_trajpd_loss(*, config: Any, model_output: Any = None, data: Any = None,
                       dp_group: Any = None, student_logits: torch.Tensor | None = None,
                       data_format: str = "thd", selected_prediction_positions: torch.Tensor | None = None):
    del config, dp_group, data_format
    if student_logits is not None:
        return compute_active_response_outputs(student_logits=student_logits, data=data,
                                               selected_prediction_positions=selected_prediction_positions)
    dp_size = int(data["dp_size"])
    token_count = int(data["batch_num_tokens"])
    if dp_size < 1 or token_count <= 0:
        raise ValueError("global assistant token count and data parallel size must be positive")
    # The engine all-reduces the action loss mask before microbatching.
    # DDP averages gradients; compensate once, then sum microbatch contributions.
    scale = dp_size / token_count
    totals = {name: model_output[name].values().sum() * scale for name in LOSS_NAMES}
    token_scale = scale
    names = ["loss/total_objective", "loss/outcome", "loss/distillation"]
    values = [totals[name].detach() for name in LOSS_NAMES]
    for name in TOKEN_METRICS:
        names.append("distillation/" + name + "_mean")
        values.append(model_output[name + "_contributions"].values().sum().detach() * token_scale)
    metrics = {name: Metric(value=value, aggregation=AggregationType.SUM)
               for name, value in zip(names, torch.stack(values).cpu().tolist(), strict=True)}
    return totals["joint_loss_contributions"], metrics
