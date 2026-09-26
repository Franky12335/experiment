"""Student-trajectory rollouts scored by a public-prefix Teacher."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import logging
import math
from typing import Any

import ray
import torch
import transfer_queue
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import (
    AgentLoopOutput,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from src.trajpd.schema import (
    DECISION_INDEX,
    DECISION_PREFIX_SNAPSHOTS,
    REWARD_EXTRA_INFO,
    TRAJECTORY_RETURN,
    DecisionPrefixSnapshot,
)
logger = logging.getLogger(__name__)

_TRAINABLE_TRAJECTORY_STATUSES = {
    "evaluated",
    "missing_code",
    "argument_failure",
    "repeated_failed_tool_call",
    "tool_timeout",
}


def validate_trainable_trajectory_status(reward_info: Any) -> str:
    """Infrastructure failures are not evidence of student failure and must not enter the training batch."""
    if not isinstance(reward_info, dict):
        raise ValueError("trajectory reward is missing reward_extra_info")
    status = str(reward_info.get("trajectory_status", "")).strip()
    parse_failure = status.startswith("parse_failure:") and bool(
        status.removeprefix("parse_failure:").strip()
    )
    if status not in _TRAINABLE_TRAJECTORY_STATUSES and not parse_failure:
        detail = str(reward_info.get("infrastructure_error", "")).strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"trajectory status {status!r} is not trainable{suffix}")
    return status


def validate_trainable_trajectory_return(reward_info: Any) -> float:
    """Reject infrastructure outcomes rather than training on a fabricated zero return."""
    if not isinstance(reward_info, dict) or TRAJECTORY_RETURN not in reward_info:
        raise ValueError(f"trajectory reward is missing {TRAJECTORY_RETURN}")
    validate_trainable_trajectory_status(reward_info)
    trajectory_return = float(reward_info[TRAJECTORY_RETURN])
    if not math.isfinite(trajectory_return) or not 0.0 <= trajectory_return <= 1.0:
        raise ValueError("trajectory_return must be finite and in [0, 1]")
    return trajectory_return


def validate_reward_tool_status_consistency(
    reward_info: Any, tool_extra_fields: dict[str, Any]
) -> None:
    """Preserve tool status through scoring so infrastructure errors cannot become ordinary task failures."""
    if not isinstance(reward_info, dict):
        raise ValueError("trajectory reward is missing reward_extra_info")
    validate_trainable_trajectory_status(reward_info)
    if str(reward_info.get("termination_reason")) != str(tool_extra_fields.get("termination_reason")):
        raise ValueError("reward must preserve the rollout termination reason")


def build_decision_index(snapshots: Any, response_mask: Any) -> torch.Tensor:
    """Tool observations remain outside the loss; decision boundaries must align with sampled-token masks."""
    if not snapshots:
        raise ValueError("a trainable trajectory must record at least one decision")
    mask = torch.as_tensor(response_mask, dtype=torch.bool)
    if mask.ndim != 1 or int(mask.numel()) < 1:
        raise ValueError("response mask must be a nonempty vector")
    response_length = int(mask.numel())
    decision_index = torch.full((response_length,), -1, dtype=torch.long)
    previous_stop = 0
    initial_prompt_length: int | None = None
    for expected_decision, raw_snapshot in enumerate(snapshots):
        snapshot = DecisionPrefixSnapshot.create_from_mapping(raw_snapshot)
        if initial_prompt_length is None:
            initial_prompt_length = snapshot.initial_prompt_length
        elif snapshot.initial_prompt_length != initial_prompt_length:
            raise ValueError("decision snapshots do not share one initial prompt length")
        if snapshot.decision_index != expected_decision:
            raise ValueError("decision snapshots are not consecutively indexed")
        if snapshot.response_start < previous_stop:
            raise ValueError("decision response spans overlap or move backward")
        start = max(0, snapshot.response_start)
        stop = min(snapshot.response_stop, response_length)
        if stop <= start:
            previous_stop = snapshot.response_stop
            continue
        decision_index[start:stop] = snapshot.decision_index
        previous_stop = snapshot.response_stop
    if int((decision_index >= 0).sum().item()) == 0:
        raise ValueError("no decision token survived response truncation")
    if not torch.equal(decision_index >= 0, mask):
        raise ValueError(
            "decision snapshots do not cover exactly the Student-generated tokens"
        )
    return decision_index


async def clear_prompt_outputs(
    uid: str, partition: str, session_start: int, count: int
) -> None:
    output_keys = [
        f"{uid}_{session_id}_0"
        for session_id in range(session_start, session_start + count)
    ]
    if output_keys:
        await transfer_queue.async_kv_clear(
            keys=output_keys, partition_id=partition
        )


@ray.remote
class RolloutTrajectoryLimiter:
    """Release generation capacity before postprocessing so finished trajectories allow queued generation to start.
    Separate postprocessing slots bound that work; the trainer still waits for the complete batch."""

    def __init__(self, capacity: int):
        if int(capacity) < 1:
            raise ValueError("global trajectory capacity must be positive")
        self.capacity = int(capacity)
        self.active = 0
        self.condition = asyncio.Condition()

    async def acquire(self) -> None:
        async with self.condition:
            await self.condition.wait_for(lambda: self.active < self.capacity)
            self.active += 1

    async def release(self) -> None:
        async with self.condition:
            if self.active < 1:
                raise RuntimeError("global trajectory limiter release is unbalanced")
            self.active -= 1
            self.condition.notify(1)


@ray.remote
class TrajPDAgentLoopWorker(AgentLoopWorker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        transfer_queue.init()
        self.background_tasks: set[asyncio.Task] = set()
        self.trajectory_limiter = None
        self._generation_slot = ContextVar("generation_slot", default=False)
        self.postprocess_slots = asyncio.Semaphore(4)
        self.initial_token_cache = {}
        if not self.distillation_enabled:
            raise ValueError(
                "bidirectional step-level on-policy distillation must be enabled"
            )
        if self.teacher_client is not None or hasattr(self, "teacher_server_manager"):
            raise ValueError(
                "phase-shared Teacher scoring must remain trainer-managed"
            )

    def set_trajectory_limiter(self, limiter: ray.actor.ActorHandle) -> None:
        if self.background_tasks:
            raise RuntimeError("cannot replace trajectory limiter during rollout")
        self.trajectory_limiter = limiter

    async def _run_bounded_trajectory(self, *args, **kwargs):
        from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
        expected_target = "src.trajpd.tool_loop.ProjectToolAgentLoop"
        name = kwargs.get("agent_name")
        registered = _agent_loop_registry.get(name, {})
        if name != "tool_agent" or registered.get("_target_") != expected_target:
            raise RuntimeError("TrajPD trajectory must execute ProjectToolAgentLoop")
        if self.trajectory_limiter is None:
            raise RuntimeError("TrajPD trajectory limiter was not installed")
        await self.trajectory_limiter.acquire.remote()
        token = self._generation_slot.set(True)
        try:
            kwargs["initial_token_cache"] = self.initial_token_cache
            return await self._run_agent_loop(*args, **kwargs)
        finally:
            if self._generation_slot.get():
                await self.trajectory_limiter.release.remote()
            self._generation_slot.reset(token)

    @staticmethod
    def build_decision_tensors(output: AgentLoopOutput) -> dict[str, torch.Tensor]:
        decision_index = build_decision_index(
            output.extra_fields.get(DECISION_PREFIX_SNAPSHOTS), output.response_mask
        )
        return {DECISION_INDEX: decision_index}

    async def generate_sequences(self, batch: TensorDict) -> None:
        validate = bool(batch["validate"]) if "validate" in batch else False
        batch.pop("validate", None)
        config = self.config.actor_rollout_ref.rollout
        sampling_params = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "logprobs": config.calculate_log_probs,
        }
        if validate:
            sampling_params.update(
                top_p=config.val_kwargs.top_p,
                top_k=config.val_kwargs.top_k,
                temperature=config.val_kwargs.temperature,
            )
        if "agent_name" not in batch:
            batch["agent_name"] = NonTensorData(config.agent.default_agent_loop)
        trajectory_info = await get_trajectory_info(
            batch["global_steps"], batch["index"], validate
        )
        for index in range(len(batch)):
            prompt: dict[str, Any] = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    prompt[key] = value[index]
                elif isinstance(value, NonTensorStack):
                    prompt[key] = value[index].data
                elif isinstance(value, NonTensorData):
                    prompt[key] = value.data
                else:
                    raise TypeError(f"unsupported prompt field {type(value)}")
            task = asyncio.create_task(
                self.process_prompt_rollouts(
                    prompt, sampling_params, trajectory_info[index]
                )
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    async def process_prompt_rollouts(
        self,
        prompt: dict[str, Any],
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
    ) -> None:
        uid = prompt["uid"]
        partition = "val" if trajectory["validate"] else "train"
        session_start = 0
        count = 0
        try:
            await transfer_queue.async_kv_put(
                key=uid, partition_id=partition, tag={"status": "running"}
            )
            rollout = self.config.actor_rollout_ref.rollout
            count = int(
                prompt.pop(
                    "__rollout_n__",
                    rollout.val_kwargs.n if trajectory["validate"] else rollout.n,
                )
            )
            if count < 1:
                raise ValueError("rollout count must be positive")
            if not trajectory["validate"] and int(count) != int(
                self.config.algorithm.trajpd.trajectories_per_prompt
            ):
                raise ValueError(
                    "training rollout must materialize one complete prompt group"
                )
            session_start = int(prompt.pop("__rollout_session_start__", 0))
            do_sample = prompt.pop("__do_sample__", True)
            if not trajectory["validate"] and not bool(do_sample):
                raise ValueError(
                    "training requires sampled on-policy trajectories; greedy rollout is invalid"
                )
            params = dict(sampling_params)
            results = await asyncio.gather(
                *(
                    self._run_bounded_trajectory(
                        params,
                        trajectory=trajectory,
                        trace=False,
                        session_id=session_id,
                        **prompt,
                    )
                    for session_id in range(session_start, session_start + count)
                ),
                return_exceptions=True,
            )
            failures = [
                result for result in results if isinstance(result, BaseException)
            ]
            if failures:
                first_failure = failures[0]
                raise RuntimeError(
                    f"{len(failures)}/{len(results)} rollout session(s) failed; "
                    f"first failure: {type(first_failure).__name__}: {first_failure}"
                ) from first_failure
            await transfer_queue.async_kv_put(
                key=uid, partition_id=partition, tag={"status": "finished"}
            )
        except Exception:
            logger.exception(
                "bidirectional step-level on-policy distillation rollout failed for uid=%s",
                uid,
            )
            await clear_prompt_outputs(uid, partition, session_start, count)
            await transfer_queue.async_kv_put(
                key=uid, partition_id=partition, tag={"status": "failure"}
            )

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        kwargs.pop("initial_token_cache")
        if self._generation_slot.get():
            self._generation_slot.set(False)
            await self.trajectory_limiter.release.remote()
        async with self.postprocess_slots:
            return await self._finish_trajectory(output, validate, **kwargs)

    async def _finish_trajectory(
        self,
        output: AgentLoopOutput | list[AgentLoopOutput],
        validate: bool,
        **kwargs,
    ) -> None:
        uid, session_id = kwargs["uid"], kwargs["session_id"]
        outputs = output if isinstance(output, list) else [output]
        if len(outputs) != 1:
            raise ValueError(
                "one rollout session must return exactly one complete trajectory"
            )
        item = outputs[0]
        if validate:
            await self._compute_score(outputs, kwargs=kwargs)
            full_response_tensors = {}
        else:
            await self._compute_score(outputs, kwargs=kwargs)
            full_response_tensors = self.build_decision_tensors(item)
        reward_info = item.extra_fields.get(REWARD_EXTRA_INFO)
        validate_reward_tool_status_consistency(reward_info, item.extra_fields)
        trajectory_return = (
            None if validate else validate_trainable_trajectory_return(reward_info)
        )

        field = item.as_dict()
        prompts = field["prompts"]
        responses = field["responses"]
        input_ids = torch.cat([prompts, responses])
        attention_mask = torch.ones_like(input_ids)
        multi_modal_inputs = self._compute_multi_modal_inputs(item, input_ids)
        position_ids = self._compute_position_ids(
            input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
        ).squeeze(0)
        field.update(kwargs)
        if "privileged_context" in field:
            raise ValueError("Actor trajectory contains obsolete privileged_context")
        field.pop("multi_modal_data", None)
        field.pop("rollout_log_probs", None)
        field["loss_mask"] = field["response_mask"]
        field["input_ids"] = input_ids
        field["position_ids"] = position_ids
        field["multi_modal_inputs"] = multi_modal_inputs
        if not validate:
            field[TRAJECTORY_RETURN] = torch.tensor(
                trajectory_return, dtype=torch.float32
            )
            field.update(full_response_tensors)
        key = f"{uid}_{session_id}_0"
        extra = field["extra_fields"]
        extra.pop(DECISION_PREFIX_SNAPSHOTS, None)
        prompt_extra_info = kwargs.get("extra_info")
        if not isinstance(prompt_extra_info, dict):
            raise ValueError("training prompt is missing extra_info identity")
        sample_id = str(prompt_extra_info.get("sample_id", "")).strip()
        if not validate and not sample_id:
            raise ValueError("training prompt is missing sample_id")
        tag = {
            "status": "success",
            "prompt_uid": uid,
            "session_id": int(session_id),
            "prompt_len": len(item.prompt_ids),
            "response_len": len(item.response_ids),
            "assistant_token_count": int(sum(item.response_mask)),
            "observation_token_count": int(extra["observation_tokens"]),
            "protocol_token_count": len(item.response_ids) - int(sum(item.response_mask)) - int(extra["observation_tokens"]),
            "generation_preemptions": (
                int(getattr(item.metrics, "num_preempted"))
                if getattr(item.metrics, "num_preempted", None) is not None else -1
            ),
            "seq_len": len(item.prompt_ids) + len(item.response_ids),
            "global_steps": kwargs["global_steps"],
            "min_global_steps": extra.get("min_global_steps"),
            "max_global_steps": extra.get("max_global_steps"),
            "termination_reason": extra.get("termination_reason", "unknown"),
            "tool_failure": extra.get("tool_failure"),
            "sample_id": sample_id,
            "task_family": "code" if "code" in str(kwargs.get("data_source", "")).lower() else "math",
            "task_return": float(trajectory_return) if trajectory_return is not None else 0.0,
            "assistant_turns": len(extra.get("turn_token_counts", [])),
            "tool_calls_used": int(extra.get("tool_calls_used", 0)),
            "wait_count": int(extra.get("wait_count", 0)),
            "turn_length_cutoffs": int(extra.get("turn_length_cutoffs", 0)),
            "agent_generate_seconds": float(item.metrics.generate_sequences),
            "agent_tool_seconds": float(item.metrics.tool_calls),
            "reward_scoring_seconds": float(item.metrics.compute_score),
        }
        await transfer_queue.async_kv_batch_put(
            keys=[key],
            fields=list_of_dict_to_tensordict([field]),
            tags=[tag],
            partition_id="val" if validate else "train",
        )


class TrajPDAgentLoopManager(AgentLoopManagerTQ):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = TrajPDAgentLoopWorker
        capacity = int(
            self.rollout_config.agent.get("max_concurrent_trajectories", 128)
        )
        self.trajectory_limiter = (
            RolloutTrajectoryLimiter.remote(capacity)
        )

    async def _init_agent_loop_workers(self):
        await super()._init_agent_loop_workers()
        if self.trajectory_limiter is not None:
            await asyncio.gather(
                *(
                    worker.set_trajectory_limiter.remote(
                        self.trajectory_limiter
                    )
                    for worker in self.agent_loop_workers
                )
            )
