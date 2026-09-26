"""Canonical project paths and local model preflight."""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def project_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    return path

def output_path(value, category):
    path = project_path(value)
    root = PROJECT_ROOT / category
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Output must be a run/file beneath {root}: {path}")
    return path

def local_model(value):
    path = project_path(value)
    config = path / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Local HF model config missing: {config}")
    json.loads(config.read_text())
    indices = list(path.glob("*.index.json"))
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if not weights:
        raise FileNotFoundError(f"No HF model weights in {path}; export the actor checkpoint first")
    for index in indices:
        for shard in set(json.loads(index.read_text()).get("weight_map", {}).values()):
            if not (path / shard).is_file():
                raise FileNotFoundError(f"Missing model shard: {path / shard}")
    return path


def validate_runtime_paths(config, stage):
    """Validate the final merged config before constructing models or workers."""
    from omegaconf import OmegaConf, open_dict
    def set_path(key, validator):
        value = OmegaConf.select(config, key)
        if value is not None:
            OmegaConf.update(config, key, str(validator(value)), merge=False)
    with open_dict(config):
        for key in ("model.path", "actor_rollout_ref.model.path",
                    "distillation.teacher_models.teacher_model.model_path"):
            set_path(key, local_model)
        for key in ("model.tokenizer_path", "actor_rollout_ref.model.tokenizer_path"):
            def tokenizer_path(value):
                path = project_path(value)
                if not path.is_dir():
                    raise FileNotFoundError(path)
                return path
            set_path(key, tokenizer_path)
        for key in ("data.train_files", "data.val_files"):
            values = OmegaConf.select(config, key)
            if values:
                if isinstance(values, str):
                    values = [values]
                paths = [project_path(value) for value in values]
                for path in paths:
                    if not path.is_file():
                        raise FileNotFoundError(path)
                OmegaConf.update(config, key, [str(path) for path in paths], merge=False)
        for key in ("reward.custom_reward_function.path",
                    "reward.reward_manager.module.path",
                    "actor_rollout_ref.rollout.multi_turn.tool_config_path",
                    "actor_rollout_ref.rollout.agent.agent_loop_config_path"):
            def existing(value):
                path = project_path(value)
                if not path.is_file():
                    raise FileNotFoundError(path)
                return path
            set_path(key, existing)
        set_path("trainer.default_local_dir", lambda value: output_path(value, "logs/evaluation" if stage == "evaluation" else "checkpoints"))
        set_path("trainer.resume_from_path", lambda value: output_path(value, "checkpoints"))
        set_path("observability.artifact_dir", lambda value: output_path(value, "artifacts"))
        set_path("trainer.rollout_data_dir", lambda value: output_path(value, "artifacts"))
        set_path("trainer.validation_data_dir", lambda value: output_path(value, "evaluation" if stage == "evaluation" else "artifacts"))
