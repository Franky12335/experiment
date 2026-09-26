"""Train a Student or Teacher SFT checkpoint."""

from __future__ import annotations

from pathlib import Path
from src.paths import validate_runtime_paths

import hydra
from omegaconf import OmegaConf

from verl.trainer import sft_trainer
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.device import auto_set_device

from src.trajpd.runtime import (
    validate_declared_gpu_count,
    merge_project_config,
    validate_vendored_verl_source,
    validate_path_exists,
)
from src.training.sft_trainer import SeededSFTTrainer


VERL_CONFIG_PATH = str(Path(sft_trainer.__file__).parent / "config")


@hydra.main(config_path=VERL_CONFIG_PATH, config_name="sft_trainer_engine", version_base=None)
def main(config) -> None:
    from src.seed import seed_everything
    seed_everything()
    validate_vendored_verl_source()
    config = merge_project_config(config, "configs/training/sft.yaml")
    auto_set_device(config)
    OmegaConf.resolve(config)
    validate_runtime_paths(config, "training")
    validate_declared_gpu_count(config)
    validate_path_exists(config.model.path, "SFT_BASE_MODEL_PATH")
    for path in config.data.train_files + config.data.val_files:
        validate_path_exists(path, "prepared SFT data")
    initialize_global_process_group()
    try:
        trainer = SeededSFTTrainer(config=config)
        trainer.fit()
    finally:
        destroy_global_process_group()


if __name__ == "__main__":
    main()
