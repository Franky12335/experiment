"""Run held-out evaluation for one model."""

from __future__ import annotations

import os
from pathlib import Path
from src.paths import validate_runtime_paths

import hydra
from omegaconf import OmegaConf

from verl.trainer import main_ppo
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from src.trajpd.runtime import (
    validate_declared_gpu_count,
    configure_cuda_runtime_library_path,
    merge_project_config,
    validate_vendored_verl_source,
)
from src.evaluation.runner import EvaluationTaskRunner


VERL_CONFIG_PATH = str(Path(main_ppo.__file__).parent / "config")


@hydra.main(config_path=VERL_CONFIG_PATH, config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    from src.seed import seed_everything
    seed_everything()
    validate_vendored_verl_source()
    configure_cuda_runtime_library_path()
    config = merge_project_config(config, "configs/evaluation/default.yaml")
    if data_path := os.environ.get("EVAL_DATA_PATH"):
        config.data.train_files = [data_path]
        config.data.val_files = [data_path]
    if generation_dir := os.environ.get("EVAL_GENERATION_DIR"):
        config.trainer.validation_data_dir = generation_dir
    auto_set_device(config)
    OmegaConf.resolve(config)
    from src.evaluation.identity import validate_identity
    validate_identity(str(config.evaluation.method), os.environ.get("EVAL_REPORT_METHOD"))
    if str(config.evaluation.method) != os.environ["EVAL_METHOD"]:
        raise ValueError("Config method differs from EVAL_METHOD")
    from src.evaluation.protocol import validate_protocol
    validate_protocol(config)
    import json
    model_config = json.loads((Path(config.actor_rollout_ref.model.path) / "config.json").read_text())
    if int(config.actor_rollout_ref.rollout.max_model_len) > int(model_config["max_position_embeddings"]):
        raise ValueError("Evaluation context exceeds the model's declared position limit")

    validate_runtime_paths(config, "evaluation")
    validate_declared_gpu_count(config)
    if int(config.actor_rollout_ref.rollout.val_kwargs.n) != int(
        config.evaluation.samples_per_problem
    ):
        raise ValueError("evaluation sample count differs between protocol and rollout")
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    if benchmark := os.environ.get("EVAL_BENCHMARK"):
        config.evaluation.benchmarks = [benchmark]
    destination = Path(config.trainer.validation_data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, destination / "evaluation_config.yaml", resolve=True)
    run_ppo(config, task_runner_class=EvaluationTaskRunner)


if __name__ == "__main__":
    main()
