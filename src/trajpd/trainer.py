"""VeRL trainer for objective-preserving on-policy distillation."""

from __future__ import annotations

import asyncio
import logging
import math
import json
from pathlib import Path
import time
from collections import Counter
from functools import partial
from typing import Any

import transfer_queue
import ray
import torch
from transfer_queue import KVBatchMeta
from verl.experimental.teacher_loop.teacher_manager import AsyncTeacherLLMServerManager
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.ray_utils import auto_await
from verl.workers.config import ActorConfig

from src.trajpd.policy_loss import compute_trajpd_loss
from src.trajpd.outcome import outcome_advantages
from src.trajpd.schema import (
    DECISION_INDEX,
    TEACHER_TOPK_LOG_PROBS,
    TEACHER_TOPK_TOKEN_IDS,
    TRAJECTORY_RETURN,
    OUTCOME_ADVANTAGE,
)
from src.trajpd.public_teacher import (
    extract_response_teacher_topk_logprobs,
)
from src.trajpd.config import TrajPDConfig
from src.trajpd.workers import TrajPDActorRolloutRefWorker


logger = logging.getLogger(__name__)


ACTOR_FIELDS = [
    "prompts",
    "responses",
    "input_ids",
    "position_ids",
    "response_mask",
    "loss_mask",
    "multi_modal_inputs",
    TEACHER_TOPK_TOKEN_IDS,
    TEACHER_TOPK_LOG_PROBS,
    DECISION_INDEX,
    TRAJECTORY_RETURN,
    OUTCOME_ADVANTAGE,
]


def validate_current_policy_step(tags: list[dict[str, Any]], expected_step: int) -> int:
    from numbers import Integral
    if isinstance(expected_step, bool) or not isinstance(expected_step, Integral) or expected_step < 0:
        raise ValueError("expected policy version must be a nonnegative integer")
    steps: set[int] = set()
    for tag in tags:
        start = tag.get("min_global_steps")
        stop = tag.get("max_global_steps")
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in (start, stop)):
            raise ValueError("trajectory policy versions must be explicit nonnegative integers")
        if start != stop:
            raise ValueError("a Student trajectory crossed a policy update")
        steps.add(int(start))
    if steps != {expected_step}:
        raise ValueError(
            f"trajectory policy steps {sorted(steps)} do not equal {expected_step}"
        )
    return expected_step


class TrajPDTrainer(PPOTrainerSync):

    def __init__(self, config):
        super().__init__(config)
        self.method = TrajPDConfig.create_from_config(config)
        self._seen_training_prompt_ids: set[str] = set()
        self._coverage_tracking_start_step: int | None = None
        self.teacher_server_manager: AsyncTeacherLLMServerManager | None = None

    def _load_checkpoint(self):
        """Restore the data position with model state so explicit resume does not silently replay the schedule."""
        mode = self.config.trainer.resume_mode
        if mode == "disable":
            return super()._load_checkpoint()
        if mode == "auto":
            from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
            checkpoint = find_latest_ckpt_path(str(Path(self.config.trainer.default_local_dir).resolve()))
            if checkpoint is None:
                raise FileNotFoundError("TrajPD resume requested but no checkpoint exists; use resume_mode=disable for a fresh run")
        elif mode == "resume_path":
            checkpoint = self.config.trainer.resume_from_path
            if not checkpoint:
                raise ValueError("resume_path requires resume_from_path")
        else:
            raise ValueError(f"unsupported TrajPD resume mode: {mode}")
        if not (Path(checkpoint) / "data.pt").is_file():
            raise FileNotFoundError(f"TrajPD resume requires dataloader state: {checkpoint}/data.pt")
        result = super()._load_checkpoint()
        self._archive_resumed_outputs()
        return result

    def _archive_resumed_outputs(self):
        import re
        import tempfile

        rollout_dir = self.config.trainer.rollout_data_dir
        folders = [(Path(rollout_dir).resolve(), r"step_(\d+)\.jsonl")] if rollout_dir else []
        for folder, pattern in folders:
            pending = []
            for path in folder.glob("step_*.jsonl"):
                match = re.fullmatch(pattern, path.name)
                if match and int(match[1]) > self.global_steps:
                    pending.append(path)
            if not pending:
                continue
            archive_root = folder / "_resume_archive"
            archive_root.mkdir(exist_ok=True)
            archive = Path(tempfile.mkdtemp(prefix=f"after_step_{self.global_steps}_", dir=archive_root))
            for path in pending:
                path.rename(archive / path.name)
            logger.info("Archived %d post-checkpoint outputs to %s", len(pending), archive)

    def _init_resource_pool_mgr(self) -> None:
        """Share the GPU pool because teacher scoring and student generation run in separate phases."""
        super()._init_resource_pool_mgr()
        for role in (Role.ActorRollout, Role.ActorRolloutRef):
            if role in self.role_worker_mapping:
                self.role_worker_mapping[role] = ray.remote(TrajPDActorRolloutRefWorker)
        if not bool(self.config.distillation.phase_shared_pool):
            raise ValueError("TrajPD requires phase_shared_pool=true")
        teacher_pool = self.resource_pool_manager.mapping[Role.TeacherModel]
        if teacher_pool == "global_pool":
            return
        self.resource_pool_manager.resource_pool_spec.pop(teacher_pool)
        self.resource_pool_manager.mapping[Role.TeacherModel] = "global_pool"

    def record_prompt_coverage(
        self, tags: list[dict[str, Any]], metrics: dict[str, float]
    ) -> None:
        if self._coverage_tracking_start_step is None:
            self._coverage_tracking_start_step = int(self.global_steps)
        counts = Counter(str(tag.get("sample_id", "")).strip() for tag in tags)
        counts.pop("", None)
        if not counts:
            raise ValueError("training trajectories contain no sample_id tags")
        expected_trajectories = self.method.trajectories_per_prompt
        malformed = {
            sample_id: count
            for sample_id, count in counts.items()
            if count != expected_trajectories
        }
        if malformed:
            raise ValueError(
                "training batch does not contain complete same-prompt trajectory groups: "
                f"{dict(list(malformed.items())[:5])}"
            )
        self._seen_training_prompt_ids.update(counts)
        expected = len(self.train_dataset)
        metrics["training/unique_prompts_seen_this_process"] = float(
            len(self._seen_training_prompt_ids)
        )
        metrics["training/prompt_coverage_this_process"] = min(
            1.0, len(self._seen_training_prompt_ids) / expected
        )
        metrics["training/full_prompt_coverage_this_process"] = float(
            len(self._seen_training_prompt_ids) == expected
        )
        if (
            self.global_steps >= self.total_training_steps
            and self._coverage_tracking_start_step == 1
            and len(self._seen_training_prompt_ids) != expected
        ):
            logger.warning(
                "Fresh training run finished after seeing %d/%d unique prompts; "
                "the checkpoint is kept because training length is user-configurable.",
                len(self._seen_training_prompt_ids),
                expected,
            )

    @staticmethod
    def record_rollout_stage_timings(
        tags: list[dict[str, Any]], metrics: dict[str, float]
    ) -> None:
        """Concurrent trajectory timings overlap; summing them would overstate step wall time."""
        fields = {
            "student_generation": "agent_generate_seconds",
            "tool_calls": "agent_tool_seconds",
            "reward_scoring": "reward_scoring_seconds",
            "teacher_scoring": "teacher_scoring_seconds",
        }
        for metric_name, tag_name in fields.items():
            values = [float(tag[tag_name]) for tag in tags if tag.get(tag_name) is not None]
            if not values:
                continue
            metrics[f"rollout/{metric_name}_seconds_mean"] = sum(values) / len(values)
            metrics[f"rollout/{metric_name}_seconds_max"] = max(values)

    @staticmethod
    def record_rollout_distribution_metrics(
        tags: list[dict[str, Any]], metrics: dict[str, float]
    ) -> None:
        lengths = sorted(int(tag.get("response_len", 0)) for tag in tags)
        if not lengths:
            return

        def quantile(probability: float) -> float:
            index = min(len(lengths) - 1, math.ceil(probability * len(lengths)) - 1)
            return float(lengths[max(index, 0)])

        metrics["rollout/response_length_mean"] = sum(lengths) / len(lengths)
        for label, probability in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
            metrics[f"rollout/response_length_{label}"] = quantile(probability)
        metrics["rollout/response_length_max"] = float(lengths[-1])

        termination_counts = Counter(
            str(tag.get("termination_reason", "unknown")) for tag in tags
        )
        total = float(len(tags))
        for name in ("assistant_turns", "tool_calls_used", "wait_count", "turn_length_cutoffs", "task_return"):
            metrics[f"rollout/{name}_mean"] = sum(float(t.get(name, 0)) for t in tags)/total
        for family in ("math", "code"):
            subset = [t for t in tags if t.get("task_family") == family]
            if subset:
                metrics[f"rollout/{family}/count"] = float(len(subset))
                metrics[f"rollout/{family}/task_return_mean"] = sum(t["task_return"] for t in subset)/len(subset)
                metrics[f"rollout/{family}/assistant_tokens_mean"] = sum(t["assistant_token_count"] for t in subset)/len(subset)
        for reason, count in termination_counts.items():
            safe_reason = reason.replace("/", "_").replace(":", "_")
            metrics[f"rollout/termination/{safe_reason}_rate"] = count / total

    def _log_rollout_data(self, batch, timing_raw, rollout_data_dir):
        """Stream records before queue cleanup so saving trajectories does not require retaining a second batch copy."""
        with marked_timer("dump_rollout_generations", timing_raw):
            data = transfer_queue.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id,
                select_fields=["prompts", "responses", "response_mask", "reward_model", "data_source", "extra_fields"])
            destination = Path(rollout_data_dir)
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / f"step_{self.global_steps}.jsonl").open("x", encoding="utf-8") as stream:
                for i, (key, tag) in enumerate(zip(batch.keys, batch.tags, strict=True)):
                    prompt = data["prompts"][i].tolist()
                    response = data["responses"][i].tolist()
                    mask = data["response_mask"][i].tolist()
                    reward_model = tu.unwrap_non_tensor_data(data["reward_model"][i])
                    extra = tu.unwrap_non_tensor_data(data["extra_fields"][i])
                    record = {"key": key, **tag, "step": int(self.global_steps),
                        "prompt": self.tokenizer.decode(prompt, skip_special_tokens=False),
                        "response": self.tokenizer.decode(response, skip_special_tokens=False),
                        "data_source": tu.unwrap_non_tensor_data(data["data_source"][i]),
                        "ground_truth": reward_model["ground_truth"],
                        "final_answer_text": extra["final_answer_text"],
                        "tool_statuses": extra["tool_statuses"] if "tool_statuses" in extra else [],
                        "protocol_errors": extra["protocol_errors"],
                        "reward_details": extra["reward_extra_info"]}
                    record.update(trajectory_id=f"step_{self.global_steps}_sample_{i}",
                        messages=extra.get("rollout_messages", []),
                        tool_events=extra.get("tool_events", []),
                        decision_prefix_snapshots=extra.get("decision_prefix_snapshots", []))
                    if self.global_steps in (1, 50, 100):
                        case_dir = destination.parent / "cases" / f"step_{self.global_steps}"
                        case_dir.mkdir(parents=True, exist_ok=True)
                        family = str(tag.get("task_family", "unknown"))
                        bucket = family + ("_correct" if bool(extra.get("reward_extra_info", {}).get("acc", False)) else "_incorrect")
                        existing = list(case_dir.glob(bucket + "_*.json"))
                        if len(existing) < 4:
                            case_record = dict(record, prompt_ids=prompt, response_ids=response, response_mask=mask)
                            (case_dir / (bucket + "_" + record["trajectory_id"] + ".json")).write_text(json.dumps(case_record, ensure_ascii=False))
                    if self.config.observability.rollout_save_token_ids:
                        record.update(prompt_ids=prompt, response_ids=response, response_mask=mask)
                    stream.write(json.dumps(record, ensure_ascii=False)+"\n")

    def on_init_end(self):
        actor_config: ActorConfig = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.actor
        )
        self.actor_rollout_wg.set_loss_fn(
            partial(
                compute_trajpd_loss,
                config=actor_config,
            )
        )
        teacher_client = self.get_teacher_client()
        if teacher_client is None or len(teacher_client) != 1:
            raise ValueError("TrajPD requires exactly one frozen Teacher checkpoint")
        self.teacher_server_manager = AsyncTeacherLLMServerManager(
            config=self.config,
            teacher_client=teacher_client,
        )
        super().on_init_end()

    def on_sample_end(self):
        with marked_timer("student_sleep", self.timing_raw):
            self.checkpoint_manager.sleep_replicas()

    def on_step_end(self):
        
        
        with marked_timer("update_weights", self.timing_raw):
            self._pending_sync_metrics = self.checkpoint_manager.update_weights(self.global_steps) or {}

    @auto_await
    async def score_teacher_batch(
        self, batch: KVBatchMeta, metrics: dict[str, float]
    ) -> KVBatchMeta:
        """Sleep rollout replicas before scoring so the teacher can reuse their GPU memory."""
        if self.teacher_server_manager is None or self.teacher_model_manager is None:
            raise RuntimeError("Teacher scoring manager is not initialized")
        fetch_started = time.perf_counter()
        data = transfer_queue.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["prompts", "responses"],
        )
        metrics["time/teacher_queue_fetch_seconds"] = time.perf_counter() - fetch_started
        prepared = []

        def prepare_requests():
            started = time.perf_counter()
            for index in range(len(batch)):
                prompt_ids = data["prompts"][index].tolist()
                response_ids = data["responses"][index].tolist()
                if not prompt_ids or not response_ids:
                    raise ValueError("Teacher scoring requires non-empty prompt and response")
                prepared.append(([*prompt_ids, *response_ids], len(prompt_ids), len(response_ids)))
            metrics["time/teacher_cpu_prepare_seconds"] = time.perf_counter() - started

        async def wake_teacher():
            started = time.perf_counter()
            try:
                await self.teacher_model_manager.wake_up()
            finally:
                metrics["time/teacher_wake_seconds"] = time.perf_counter() - started

        max_concurrency = int(
            self.config.distillation.teacher_models.teacher_model.inference.max_num_seqs
        )
        if max_concurrency < 1:
            raise ValueError("Teacher max_num_seqs must be positive")
        semaphore = asyncio.Semaphore(max_concurrency)

        async def score_one(index: int):
            sequence_ids, prompt_length, response_length = prepared[index]
            queued_at = time.perf_counter()
            async with semaphore:
                started = time.perf_counter()
                queue_seconds = started - queued_at
                teacher_ids, teacher_logprobs = (
                    await self.teacher_server_manager.compute_teacher_logprobs_single(
                        sequence_ids=sequence_ids,
                    )
                )
            topk_ids, topk_logprobs = extract_response_teacher_topk_logprobs(
                teacher_ids,
                teacher_logprobs,
                prompt_length,
                response_length,
                topk=int(self.config.distillation.distillation_loss.topk),
            )
            return topk_ids, topk_logprobs, time.perf_counter() - started, queue_seconds

        try:
            
            startup = await asyncio.gather(wake_teacher(), asyncio.to_thread(prepare_requests), return_exceptions=True)
            for result in startup:
                if isinstance(result, BaseException):
                    raise result
            scoring_started = time.perf_counter()
            outputs = await asyncio.gather(
                *(score_one(i) for i in range(len(batch))), return_exceptions=True
            )
            metrics["time/teacher_score_seconds"] = time.perf_counter() - scoring_started
            
            
            for output in outputs:
                if isinstance(output, BaseException):
                    raise output
        finally:
            sleep_started = time.perf_counter()
            await self.teacher_model_manager.sleep()
            metrics["time/teacher_sleep_seconds"] = time.perf_counter() - sleep_started

        data = None
        prepared = None
        pack_started = time.perf_counter()
        topk_ids, topk_logprobs, durations, queue_durations = zip(
            *outputs, strict=True
        )
        fields = {
            TEACHER_TOPK_TOKEN_IDS: torch.nested.as_nested_tensor(
                list(topk_ids), layout=torch.jagged
            ),
            TEACHER_TOPK_LOG_PROBS: torch.nested.as_nested_tensor(
                list(topk_logprobs), layout=torch.jagged
            ),
        }
        
        del outputs, topk_ids, topk_logprobs
        transfer_queue.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=tu.get_tensordict(fields),
        )
        metrics["time/teacher_pack_and_put_seconds"] = time.perf_counter() - pack_started
        for tag, duration in zip(batch.tags, durations, strict=True):
            tag["teacher_scoring_seconds"] = float(duration)
        metrics["teacher/requests"] = float(len(batch))
        metrics["teacher/request_seconds_mean"] = sum(durations) / len(durations)
        metrics["teacher/request_seconds_max"] = max(durations)
        metrics["teacher/request_queue_seconds_mean"] = sum(queue_durations) / len(queue_durations)
        metrics["teacher/request_queue_seconds_max"] = max(queue_durations)
        return batch

    def update_student_actor(
        self,
        batch: KVBatchMeta,
        trajectory_count: int,
        metrics: dict[str, float],
    ) -> None:
        role = "actor"
        if role not in self.actor_rollout_wg._dispatch_info:
            self.actor_rollout_wg._dispatch_info[role] = (
                self.actor_rollout_wg._query_dispatch_info(role)
            )
        dp_size = max(self.actor_rollout_wg._dispatch_info[role]) + 1
        if self.use_reference_policy:
            raise RuntimeError("TrajPD must not construct a reference policy")
        if len(batch) != trajectory_count:
            raise ValueError("TrajPD cannot pad or truncate a rollout update")
        validate_current_policy_step(batch.tags, self.global_steps - 1)
        batch.extra_info.update(
            {
                "calculate_entropy": False,
                "active_response_logits": True,
                "max_active_tokens": int(self.config.data.interaction_budget.max_generated_tokens),
                "global_batch_size": len(batch),
                "global_trajectory_count": trajectory_count,
                "dp_size": dp_size,
                "mini_batch_size": len(batch),
                "epochs": 1,
                "seed": int(self.config.actor_rollout_ref.actor.data_loader_seed),
                "dataloader_kwargs": {"shuffle": False},
                "temperature": 1.0,
                "policy_version": int(self.global_steps - 1),
                "distillation_alpha": self.method.distillation_alpha,
                "solver_atol": self.method.solver_atol,
                "solver_rtol": self.method.solver_rtol,
                "solver_max_iterations": self.method.solver_max_iterations,
                "solver_max_bracket_iterations": self.method.solver_max_bracket_iterations,
            }
        )
        actor_started = time.perf_counter()
        output = self.actor_rollout_wg.update_actor(batch)
        metrics["time/actor_backward_optimizer_seconds"] = (
            time.perf_counter() - actor_started
        )
        metrics.update(reduce_metrics(rename_dict(output["metrics"], "actor/")))

    def prepare_outcome_advantages(self, batch: KVBatchMeta, metrics: dict) -> None:
        """Compute raw-return LOO advantages on the complete batch before DP balancing."""
        values = transfer_queue.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id,
            select_fields=[TRAJECTORY_RETURN],
        )
        returns = torch.stack([
            value.reshape(()) for value in values[TRAJECTORY_RETURN].unbind()
        ]).cpu()
        if returns.numel() != len(batch):
            raise ValueError("global return batch is incomplete")
        advantages = outcome_advantages(returns)
        transfer_queue.kv_batch_put(
            keys=batch.keys, partition_id=batch.partition_id,
            fields=tu.get_tensordict({OUTCOME_ADVANTAGE: advantages}),
        )
        metrics.update({
            "outcome/raw_return_mean": returns.float().mean().item(),
            "outcome/advantage_abs_mean": advantages.abs().mean().item(),
            "outcome/advantage_mean": advantages.mean().item(),
        })

    def step(self, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        self._add_batch_to_generate()
        self.on_sample_begin()
        try:
            with marked_timer("on_policy_rollout", timing_raw, color="red"):
                batch, replay_metrics = self.replay_buffer.sample(
                    global_steps=self.global_steps,
                    partition_id="train",
                    batch_size=int(self.config.data.train_batch_size),
                )
                metrics.update(replay_metrics)
        finally:
            self.on_sample_end()
        
        
        expected_count = int(self.config.data.train_batch_size) * int(self.method.trajectories_per_prompt)
        failed_groups = sum(
            value for name, value in replay_metrics.items()
            if name.endswith("/rollout_failure/evicted_samples")
        )
        if failed_groups or len(batch) != expected_count:
            raise RuntimeError(
                f"Incomplete TrajPD rollout batch: received {len(batch)}/{expected_count} trajectories; "
                f"evicted failed samples={failed_groups}. Inspect rollout worker errors. "
                "Refusing padding or partial-batch optimization."
            )
        
        self.record_prompt_coverage(batch.tags, metrics)
        validate_current_policy_step(batch.tags, self.global_steps - 1)
        with marked_timer("teacher_scoring", timing_raw, color="yellow"):
            batch = self.score_teacher_batch(batch, metrics)
        self.record_rollout_stage_timings(batch.tags, metrics)
        self.record_rollout_distribution_metrics(batch.tags, metrics)
        metrics["rollout/infrastructure_failure_groups"] = float(
            sum(
                value
                for name, value in replay_metrics.items()
                if name.endswith("/rollout_failure/evicted_samples")
            )
        )
        trajectory_count = len(batch)
        self.prepare_outcome_advantages(batch, metrics)
        batch = super()._balance_batch(batch, metrics, logging_prefix="response_seqlen")
        batch.fields = list(ACTOR_FIELDS)
        with marked_timer("actor_update", timing_raw, color="blue"):
            self.update_student_actor(batch, trajectory_count, metrics)
        return batch

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch) -> None:
        metrics["training/global_step"] = float(global_steps)
        metrics["training/epoch"] = float(epoch)
        for name, seconds in timing_raw.items():
            metrics[f"time/{name}_seconds"] = float(seconds)
        trajectory_count = len(batch)
        response_tokens = sum(int(tag.get("response_len", 0)) for tag in batch.tags)
        assistant_tokens = sum(int(tag.get("assistant_token_count", 0)) for tag in batch.tags)
        observation_tokens = sum(int(tag.get("observation_token_count", 0)) for tag in batch.tags)
        token_counts_complete = all(
            "assistant_token_count" in tag and "observation_token_count" in tag
            for tag in batch.tags
        )
        if token_counts_complete:
            metrics["rollout/assistant_tokens"] = assistant_tokens
            metrics["rollout/observation_tokens"] = observation_tokens
            metrics["rollout/protocol_tokens"] = sum(int(tag.get("protocol_token_count", 0)) for tag in batch.tags)
        preemptions = [int(tag.get("generation_preemptions", -1)) for tag in batch.tags]
        known_preemptions = [value for value in preemptions if value >= 0]
        if known_preemptions:
            metrics["rollout/generation_preemptions"] = sum(known_preemptions)
        step_seconds = float(timing_raw.get("step", 0.0))
        rollout_seconds = float(
            timing_raw.get("on_policy_rollout", 0.0)
        )
        teacher_seconds = float(timing_raw.get("teacher_scoring", 0.0))
        actor_seconds = float(timing_raw.get("actor_update", 0.0))
        if step_seconds > 0:
            metrics["throughput/trajectories_per_second"] = (
                trajectory_count / step_seconds
            )
            metrics["throughput/response_tokens_per_second"] = (
                response_tokens / step_seconds
            )
        if rollout_seconds > 0:
            if token_counts_complete:
                metrics["throughput/rollout_assistant_tokens_per_second"] = assistant_tokens / rollout_seconds
        if teacher_seconds > 0:
            metrics["throughput/teacher_response_tokens_per_second"] = response_tokens / teacher_seconds
        if actor_seconds > 0:
            metrics["throughput/actor_response_tokens_per_second"] = (
                response_tokens / actor_seconds
            )
