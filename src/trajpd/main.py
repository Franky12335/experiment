"""Hydra/Ray entrypoint for trajectory-adaptive on-policy distillation."""

from __future__ import annotations

from pathlib import Path
from src.paths import validate_runtime_paths
from src.trajpd.run_manifest import ensure_run_manifest

import hydra
import ray
import transfer_queue
from omegaconf import OmegaConf

from verl.trainer import main_ppo
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from src.trajpd.agent_loop import (
    TrajPDAgentLoopManager,
)
from src.trajpd.public_teacher import (
    validate_tokenizer_and_canonical_prompt_compatibility,
)
from src.trajpd.config import (
    configure_prompt_schedule,
    configure_distillation_training,
    validate_public_training_prompts,
)
from src.trajpd.runtime import (
    configure_cuda_runtime_library_path,
    merge_project_config,
    validate_vendored_verl_source,
    validate_path_exists,
)
from src.trajpd.trainer import (
    TrajPDTrainer,
)


VERL_CONFIG_PATH = str(Path(main_ppo.__file__).parent / "config")
PROJECT_CONFIG = Path(__file__).resolve().parents[2] / "configs/training/trajpd.yaml"


@ray.remote
class TrajPDTaskRunner:
    def run(self, config) -> None:
        from src.seed import seed_everything
        seed_everything()
        transfer_queue.init(config.transfer_queue)
        trainer = None
        exit_code = 1
        try:
            trainer = TrajPDTrainer(config=config)
            trainer.init()
            manager = TrajPDAgentLoopManager.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
            exit_code = 0
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=exit_code)
            finally:
                transfer_queue.close()


@hydra.main(config_path=VERL_CONFIG_PATH, config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    from src.seed import seed_everything
    seed_everything()
    validate_vendored_verl_source()
    configure_cuda_runtime_library_path()
    config = merge_project_config(
        config, PROJECT_CONFIG,
    )
    auto_set_device(config)
    method = configure_distillation_training(config)
    OmegaConf.resolve(config)
    validate_runtime_paths(config, "training")
    run_dir = Path(config.trainer.default_local_dir)
    if config.trainer.resume_mode == "disable" and (
        (run_dir / "latest_checkpointed_iteration.txt").exists()
        or any(run_dir.glob("global_step_*"))
        or any((Path(config.observability.artifact_dir) / "rollouts").glob("step_*.jsonl"))
    ):
        raise ValueError("fresh TrajPD training requires a new run directory; existing checkpoints/rollouts must not be mixed with a new run")
    validate_path_exists(config.actor_rollout_ref.model.path, "STUDENT_MODEL_PATH")
    teacher_model_path = str(
        config.distillation.teacher_models.teacher_model.model_path
    )
    validate_path_exists(teacher_model_path, "TEACHER_MODEL_PATH")
    for path in config.data.train_files + config.data.val_files:
        validate_path_exists(path, "training data")
    tool_config = OmegaConf.load(
        config.actor_rollout_ref.rollout.multi_turn.tool_config_path
    )
    tool_schemas = [
        OmegaConf.to_container(item.tool_schema, resolve=True)
        for item in tool_config.tools
    ]
    tokenizer = validate_tokenizer_and_canonical_prompt_compatibility(
        str(config.actor_rollout_ref.model.path),
        teacher_model_path,
        config.data.get("apply_chat_template_kwargs", {}),
        tool_schemas,
    )
    prompt_preflight = validate_public_training_prompts(
        paths=[str(path) for path in config.data.train_files],
        tokenizer=tokenizer,
        prompt_key=str(config.data.prompt_key),
        tool_schemas=tool_schemas,
        apply_chat_template_kwargs=config.data.get("apply_chat_template_kwargs", {}),
        max_prompt_length=int(config.data.max_prompt_length),
        max_response_length=int(config.data.max_response_length),
        interaction_budget=config.data.interaction_budget,
        student_max_model_len=int(config.actor_rollout_ref.rollout.max_model_len),
        teacher_max_model_len=int(
            config.distillation.teacher_models.teacher_model.inference.max_model_len
        ),
    )
    schedule = configure_prompt_schedule(
        config=config, prompt_count=prompt_preflight.prompt_count
    )
    run_dir = Path(config.trainer.default_local_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Public prompt preflight: "
        f"prompts={prompt_preflight.prompt_count}, "
        f"maximum_tokens={prompt_preflight.maximum_prompt_tokens}"
    )
    print(
        "Prompt schedule: "
        f"batch={schedule.prompt_batch_size}, "
        f"steps={schedule.total_training_steps}, "
        f"slots={schedule.prompt_slots}, "
        f"planned_coverage={schedule.planned_prompt_coverage:.2%}, "
        f"repeated_slots_after_first_pass={schedule.repeated_prompt_slots}, "
        f"steps_per_epoch={schedule.steps_per_epoch}, "
        f"epochs={schedule.total_epochs}"
    )
    print(
        "GPU allocation: "
        f"shared_student_teacher_pool={config.trainer.n_gpus_per_node}, "
        "schedule=student_rollout->frozen_teacher_scoring->student_update"
    )
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    ensure_run_manifest(config)
    run_ppo(
        config,
        task_runner_class=TrajPDTaskRunner,
    )


if __name__ == "__main__":
    main()
