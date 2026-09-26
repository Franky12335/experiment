"""Method-critical configuration and prompt/context validation."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict
from pyarrow import parquet
from verl.trainer.config.algorithm import AlgoConfig


@dataclass(frozen=True, slots=True)
class TrajPDConfig:
    """Required, fixed TrajPD objective and numerical solver settings."""

    trajectories_per_prompt: int
    distillation_alpha: float
    solver_atol: float
    solver_rtol: float
    solver_max_iterations: int
    solver_max_bracket_iterations: int

    @classmethod
    def create_from_config(cls, config: DictConfig) -> "TrajPDConfig":
        raw = OmegaConf.to_container(config.algorithm.trajpd, resolve=True)
        required = set(cls.__dataclass_fields__)
        if not isinstance(raw, dict) or set(raw) != required:
            supplied = set(raw) if isinstance(raw, dict) else set()
            raise ValueError(
                f"algorithm.trajpd fields differ: missing={sorted(required-supplied)}, "
                f"unknown={sorted(supplied-required)}"
            )
        integers = {"trajectories_per_prompt", "solver_max_iterations", "solver_max_bracket_iterations"}
        for name in integers:
            if isinstance(raw[name], bool) or not isinstance(raw[name], int):
                raise ValueError(f"{name} must be an integer")
        for name in required - integers:
            if isinstance(raw[name], bool):
                raise ValueError(f"{name} must be numeric, not boolean")
        value = cls(**{
            key: int(raw[key]) if key in integers else float(raw[key])
            for key in required
        })
        value.validate()
        return value

    def validate(self) -> None:
        if self.trajectories_per_prompt != 1:
            raise ValueError("TrajPD uses one fresh trajectory per prompt")
        for name in ("distillation_alpha", "solver_atol", "solver_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.solver_atol >= 1 or self.solver_rtol >= 1:
            raise ValueError("solver tolerances must be less than one")
        if self.solver_max_iterations < 1 or self.solver_max_bracket_iterations < 1:
            raise ValueError("solver iteration limits must be positive")


def configure_distillation_training(
    config: DictConfig,
) -> TrajPDConfig:
    """Keep one update per fresh batch because the sampled-policy gradient assumes unchanged parameters during loss construction."""
    allowed = set(AlgoConfig.__dataclass_fields__) | {"trajpd", "_target_"}
    unknown = set(config.algorithm) - allowed
    if unknown:
        raise ValueError(f"Unsupported algorithm configuration fields: {sorted(unknown)}")
    if "trajpd" not in config.algorithm:
        raise ValueError("algorithm.trajpd must be configured")
    method = TrajPDConfig.create_from_config(config)
    if int(config.data.train_batch_size) * method.trajectories_per_prompt < 2:
        raise ValueError("an update requires at least two trajectories")
    if int(OmegaConf.select(config, "distillation.distillation_loss.topk", default=0)) < 1:
        raise ValueError("Teacher probability support must be nonempty")
    
    if OmegaConf.select(config, "async_training.partial_rollout", default=False):
        raise ValueError("TrajPD requires complete trajectories; partial rollout is disabled")
    if str(config.trainer.v1.trainer_mode) != "sync":
        raise ValueError("TrajPD requires synchronous outer-loop updates")
    if int(config.trainer.v1.sampler.max_off_policy_threshold) != 1:
        raise ValueError("sampler threshold is an inclusive span and must be one; exact policy version is enforced separately")
    if not bool(OmegaConf.select(config, "distillation.phase_shared_pool", default=False)):
        raise ValueError("TrajPD requires a phase-shared Student/Teacher GPU pool")
    if OmegaConf.select(config, "algorithm.filter_groups.enable", default=False):
        raise ValueError("TrajPD cannot silently filter on-policy prompt groups")
    if config.actor_rollout_ref.actor.strategy != "fsdp2":
        raise ValueError("TrajPD requires its registered fsdp2 engine")
    if config.actor_rollout_ref.model.external_lib != "src.trajpd.decision_engine":
        raise ValueError("TrajPD requires its custom dense engine module")
    if config.actor_rollout_ref.rollout.agent.default_agent_loop != "tool_agent":
        raise ValueError("TrajPD requires the project tool_agent loop")
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.agent.get("max_concurrent_prompt_groups", 0)) != 0 or int(
        rollout.agent.get("max_concurrent_prompt_groups_global", 0)
    ) != 0:
        raise ValueError("TrajPD uses max_concurrent_trajectories; obsolete group admission must be disabled")
    if int(rollout.agent.get("max_concurrent_trajectories", 128)) < 1:
        raise ValueError("TrajPD requires positive bounded trajectory capacity")
    budget = config.data.interaction_budget
    if (budget.max_assistant_turns < 2 or rollout.multi_turn.max_assistant_turns != budget.max_assistant_turns
        or not 0 <= budget.max_tool_calls < budget.max_assistant_turns or rollout.multi_turn.max_parallel_calls != 1):
        raise ValueError("budget must reserve a final assistant turn and allow one tool call per turn")
    if rollout.multi_turn.max_user_turns is not None:
        raise ValueError("max_user_turns must not replace the explicit tool-call counter")
    if budget.max_new_tokens_per_turn < 1 or budget.final_answer_max_new_tokens < 1:
        raise ValueError("per-turn token budgets must be positive")
    turn_capacity = (budget.max_assistant_turns - 1) * budget.max_new_tokens_per_turn + budget.final_answer_max_new_tokens
    generated = budget.max_generated_tokens
    if not budget.max_new_tokens_per_turn + budget.final_answer_max_new_tokens <= generated <= turn_capacity:
        raise ValueError("total generation budget must be reachable within configured turns and reserve the final answer")
    if generated + budget.max_tool_calls * budget.max_tool_observation_tokens + budget.max_protocol_tokens > config.data.max_response_length:
        raise ValueError("generation, observation and protocol budgets exceed response capacity")
    actor = config.actor_rollout_ref.actor
    if (
        config.data.get("apply_chat_template_kwargs", {}).get("enable_thinking")
        is not False
    ):
        raise ValueError("distillation rollout must use enable_thinking=false")
    if int(rollout.n) != method.trajectories_per_prompt:
        raise ValueError("rollout.n must equal trajectories_per_prompt")
    if bool(rollout.calculate_log_probs):
        raise ValueError(
            "rollout.calculate_log_probs must be disabled; the objective recomputes "
            "current Student log-probabilities from selected action logits"
        )
    if bool(actor.get("use_rollout_log_probs", False)):
        raise ValueError(
            "actor.use_rollout_log_probs must be disabled; rollout log-probabilities "
            "do not enter the TrajPD objective"
        )
    if not (
        float(rollout.temperature) == 1.0
        and 0.0 < float(rollout.top_p) <= 1.0
        and int(rollout.top_k) == -1
    ):
        raise ValueError(
            "rollout requires temperature 1, top_p in (0,1], and disabled top_k"
        )
    if int(actor.ppo_epochs) != 1 or bool(actor.shuffle):
        raise ValueError("on-policy updates require one unshuffled actor epoch")
    if int(actor.ppo_mini_batch_size) != int(config.data.train_batch_size):
        raise ValueError(
            "VeRL V1 actor.ppo_mini_batch_size must count prompt groups and equal "
            "data.train_batch_size; rollout.n expands it to the selected-action batch"
        )
    if str(config.algorithm.adv_estimator) != "reinforce":
        raise ValueError("TrajPD declares reinforce; advantages are computed globally by its trainer")
    if bool(OmegaConf.select(config, "critic.enable", default=False)):
        raise ValueError("TrajPD has no critic")
    if bool(actor.use_kl_loss) or float(actor.kl_loss_coef) != 0.0:
        raise ValueError("TrajPD has no reference worker or extra KL loss")
    if bool(OmegaConf.select(config, "algorithm.use_kl_in_reward", default=False)):
        raise ValueError("TrajPD uses raw task returns, with no reference KL reward shaping")
    if float(actor.get("entropy_coeff", 0.0)) != 0.0:
        raise ValueError("TrajPD has exactly two losses; entropy bonus must be zero")
    if int(actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError("TrajPD requires complete decisions on each rank; sequence parallelism must be one")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise ValueError("set CUDA_VISIBLE_DEVICES before launch")
    device_ids = [item.strip() for item in visible.split(",") if item.strip()]
    if len(device_ids) != 4:
        raise ValueError("the formal phase-shared TrajPD schedule requires exactly four GPUs")
    if len(device_ids) != len(set(device_ids)):
        raise ValueError("CUDA_VISIBLE_DEVICES contains duplicate GPU ids")
    shared_gpus = len(device_ids)
    if int(config.data.train_batch_size) % shared_gpus:
        raise ValueError(
            "prompt groups per update must be divisible by the Student GPU count"
        )
    teacher = config.distillation.teacher_models.teacher_model
    teacher_world_size = (
        int(teacher.inference.tensor_model_parallel_size)
        * int(teacher.inference.data_parallel_size)
        * int(teacher.inference.pipeline_model_parallel_size)
    )
    if teacher_world_size != shared_gpus:
        raise ValueError("the phase-shared Teacher must span all four GPUs")
    if not bool(teacher.inference.free_cache_engine):
        raise ValueError("phase-shared Teacher inference must release weights and KV cache")
    if not bool(config.actor_rollout_ref.rollout.free_cache_engine):
        raise ValueError("phase-shared Student rollout must release weights and KV cache")
    with open_dict(config):
        config.trainer.n_gpus_per_node = shared_gpus
        config.distillation.n_gpus_per_node = shared_gpus
    return method


@dataclass(frozen=True, slots=True)
class PublicPromptPreflight:
    prompt_count: int
    maximum_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class PromptSchedule:
    prompt_count: int
    prompt_batch_size: int
    total_training_steps: int
    prompt_slots: int
    planned_prompt_coverage: float
    repeated_prompt_slots: int
    steps_per_epoch: int
    total_epochs: int


def configure_prompt_schedule(
    *, config: DictConfig, prompt_count: int
) -> PromptSchedule:
    prompt_batch_size = int(config.data.train_batch_size)
    total_training_steps = int(config.trainer.total_training_steps)
    if prompt_batch_size < 1 or total_training_steps < 1:
        raise ValueError("training batch size and step count must be positive")
    prompt_slots = prompt_batch_size * total_training_steps
    planned_prompt_coverage = min(1.0, prompt_slots / prompt_count)
    repeated_prompt_slots = max(0, prompt_slots - prompt_count)
    steps_per_epoch = max(1, prompt_count // prompt_batch_size)
    required_epochs = math.ceil(total_training_steps / steps_per_epoch)
    configured_epochs = int(config.trainer.total_epochs)
    total_epochs = max(configured_epochs, required_epochs)
    if total_epochs != configured_epochs:
        with open_dict(config):
            config.trainer.total_epochs = total_epochs
    return PromptSchedule(
        prompt_count=prompt_count,
        prompt_batch_size=prompt_batch_size,
        total_training_steps=total_training_steps,
        prompt_slots=prompt_slots,
        planned_prompt_coverage=planned_prompt_coverage,
        repeated_prompt_slots=repeated_prompt_slots,
        steps_per_epoch=steps_per_epoch,
        total_epochs=total_epochs,
    )


def validate_public_training_prompts(
    *,
    paths: Sequence[str],
    tokenizer: Any,
    prompt_key: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    apply_chat_template_kwargs: Mapping[str, Any],
    max_prompt_length: int,
    max_response_length: int,
    interaction_budget: Any,
    student_max_model_len: int,
    teacher_max_model_len: int,
) -> PublicPromptPreflight:
    """Count tool schemas and budget instructions too, because they consume the context used by both scorers."""
    limits = {
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "student_max_model_len": student_max_model_len,
        "teacher_max_model_len": teacher_max_model_len,
    }
    if any(value < 1 for value in limits.values()):
        raise ValueError("prompt and context limits must be positive")
    maximum_sequence_length = max_prompt_length + max_response_length
    if maximum_sequence_length > student_max_model_len:
        raise ValueError(
            "public prompt and response budgets exceed the Student context"
        )
    if maximum_sequence_length > teacher_max_model_len:
        raise ValueError("public trajectory budget exceeds the Teacher scorer context")

    prompt_count = 0
    maximum_prompt_tokens = 0
    template_kwargs = dict(apply_chat_template_kwargs)
    tools = [dict(schema) for schema in tool_schemas]
    for path in paths:
        source = Path(path)
        dataset = parquet.ParquetFile(source)
        columns = set(dataset.schema_arrow.names)
        if "privileged_context" in columns:
            raise ValueError(
                f"{source} contains obsolete privileged_context; regenerate public data"
            )
        if prompt_key not in columns:
            raise ValueError(f"{source} is missing public prompt column {prompt_key!r}")
        row_offset = 0
        for batch in dataset.iter_batches(columns=[prompt_key], batch_size=256):
            prompts = batch.column(0).to_pylist()
            for local_index, messages in enumerate(prompts):
                if not isinstance(messages, list) or not messages:
                    raise ValueError(
                        f"{source} row {row_offset + local_index} has no public messages"
                    )
                from src.trajpd.tool_loop import budget_prompt
                tokenized = tokenizer.apply_chat_template(
                    budget_prompt(messages, interaction_budget),
                    tokenize=True,
                    add_generation_prompt=True,
                    tools=tools,
                    **template_kwargs,
                )
                token_ids = (
                    tokenized["input_ids"]
                    if isinstance(tokenized, Mapping)
                    else tokenized
                )
                length = len(token_ids)
                if length < 1 or length > max_prompt_length:
                    raise ValueError(
                        f"{source} row {row_offset + local_index} public prompt has "
                        f"{length} tokens; required range is [1, {max_prompt_length}]"
                    )
                maximum_prompt_tokens = max(maximum_prompt_tokens, length)
            prompt_count += len(prompts)
            row_offset += len(prompts)
    if prompt_count < 1:
        raise ValueError("training parquet files contain no public prompts")
    return PublicPromptPreflight(
        prompt_count=prompt_count,
        maximum_prompt_tokens=maximum_prompt_tokens,
    )
