"""Runtime helpers shared by research entrypoints."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict


from src.paths import PROJECT_ROOT, project_path


def configure_cuda_runtime_library_path() -> str:
    """Scheduler subprocesses need the pip CUDA library paths because they do not inherit Python import resolution."""
    python_folder = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(sys.prefix) / "lib" / python_folder / "site-packages"
    candidates = (
        site_packages / "nvidia/cu13/lib",
        site_packages / "nvidia/cudnn/lib",
        site_packages / "nvidia/nccl/lib",
        site_packages / "nvidia/nvshmem/lib",
        site_packages / "nvidia/cusparselt/lib",
        site_packages / "torch/lib",
    )
    existing = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    available = [str(path) for path in candidates if path.is_dir()]
    if not any((Path(path) / "libcudart.so.13").is_file() for path in available):
        raise FileNotFoundError(
            f"CUDA 13 runtime library is missing under the active environment: {sys.prefix}"
        )
    ordered = list(dict.fromkeys([*available, *existing]))
    os.environ["LD_LIBRARY_PATH"] = ":".join(ordered)
    return os.environ["LD_LIBRARY_PATH"]


def validate_vendored_verl_source() -> Path:
    """Mixing installed and vendored framework modules can silently change the training interfaces."""
    import verl

    expected = (PROJECT_ROOT / "verl").resolve()
    actual = Path(verl.__file__).resolve()
    if not actual.is_relative_to(expected):
        raise RuntimeError(
            "VeRL must load from the repository source tree: "
            f"expected under {expected}, loaded {actual}"
        )
    required = (
        expected / "trainer/ppo/v1/trainer_sync.py",
        expected / "workers/engine/fsdp/transformer_impl.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"vendored VeRL lacks required training interfaces: {missing}"
        )
    return actual


def merge_project_config(
    config: DictConfig,
    relative_path: str | Path,
    command_line_overrides: Sequence[str] | None = None,
    parameter_path: str | Path | None = None,
) -> DictConfig:
    """Reapply explicit CLI values after project defaults so user overrides are not silently overwritten."""
    path = Path(relative_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if command_line_overrides is None:
        command_line_overrides = (
            HydraConfig.get().overrides.task if HydraConfig.initialized() else ()
        )
    with open_dict(config):
        merged = OmegaConf.merge(config, OmegaConf.load(path))
        if parameter_path is not None:
            parameters = Path(parameter_path)
            if not parameters.is_absolute():
                parameters = PROJECT_ROOT / parameters
            merged = OmegaConf.merge(merged, OmegaConf.load(parameters))
        if command_line_overrides:
            
            
            
            normalized_overrides = [
                value[2:]
                if value.startswith("++")
                else value[1:]
                if value.startswith("+")
                else value
                for value in command_line_overrides
            ]
            merged = OmegaConf.merge(
                merged, OmegaConf.from_dotlist(normalized_overrides)
            )
    return merged


def count_visible_gpus() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [value.strip() for value in raw.split(",") if value.strip()]
    if not devices:
        raise ValueError("set CUDA_VISIBLE_DEVICES before launch")
    if len(devices) != len(set(devices)):
        raise ValueError("CUDA_VISIBLE_DEVICES contains a duplicate GPU")
    return len(devices)


def validate_path_exists(path: str, label: str) -> Path:
    resolved = project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def validate_declared_gpu_count(config: DictConfig) -> None:
    declared = int(config.trainer.n_gpus_per_node)
    visible = count_visible_gpus()
    if declared != visible:
        raise ValueError(
            f"NUM_GPUS={declared} but CUDA_VISIBLE_DEVICES exposes {visible} devices"
        )
