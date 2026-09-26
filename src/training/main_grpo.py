"""Continue the Teacher SFT checkpoint with GRPO."""

from __future__ import annotations

from pathlib import Path
from src.paths import validate_runtime_paths

import hydra
import ray
import transfer_queue
from omegaconf import OmegaConf, open_dict

from verl.trainer import main_ppo
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from src.trajpd.runtime import (
    configure_cuda_runtime_library_path,
    merge_project_config,
    count_visible_gpus,
    validate_vendored_verl_source,
    validate_path_exists,
)
from src.training.grpo_trainer import TeacherGRPOTrainer


VERL_CONFIG_PATH = str(Path(main_ppo.__file__).parent / "config")


def configure_teacher_execution(config) -> str:
    gpu_count = count_visible_gpus()
    with open_dict(config):
        config.trainer.v1.trainer_mode = "sync"
        config.trainer.n_gpus_per_node = gpu_count
        config.actor_rollout_ref.rollout.nnodes = 0
        config.actor_rollout_ref.rollout.n_gpus_per_node = gpu_count
        config.actor_rollout_ref.rollout.checkpoint_engine.backend = "naive"
    return str(config.trainer.v1.trainer_mode)


@ray.remote
class TeacherGRPOTaskRunner:

    def run(self, config) -> None:
        from src.seed import seed_everything
        seed_everything()
        config.transfer_queue.enable = True
        transfer_queue.init(config.transfer_queue)
        try:
            trainer = TeacherGRPOTrainer(config=config)
            trainer.init()
            agent_loop_manager = AgentLoopManagerTQ.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(agent_loop_manager)
        finally:
            transfer_queue.close()


@hydra.main(config_path=VERL_CONFIG_PATH, config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    from src.seed import seed_everything
    seed_everything()
    validate_vendored_verl_source()
    configure_cuda_runtime_library_path()
    config = merge_project_config(config, "configs/training/grpo.yaml")
    trainer_mode = configure_teacher_execution(config)
    auto_set_device(config)
    OmegaConf.resolve(config)
    validate_runtime_paths(config, "training")
    validate_path_exists(
        config.actor_rollout_ref.model.path, "TEACHER_SFT_MODEL_PATH"
    )
    for path in config.data.train_files + config.data.val_files:
        validate_path_exists(path, "prepared GRPO data")
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    print(
        f"Teacher GRPO execution: {trainer_mode}; "
        f"trainer_gpus={config.trainer.n_gpus_per_node}; "
        f"rollout_gpus={config.trainer.n_gpus_per_node} (colocated)"
    )
    run_ppo(config, task_runner_class=TeacherGRPOTaskRunner)


if __name__ == "__main__":
    main()
