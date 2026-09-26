"""Bind each TrajPD run to its objective, model paths and implementation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf


def ensure_run_manifest(config, *, source_dir=None):
    source = Path(source_dir) if source_dir else Path(__file__).resolve().parent
    identity = {
        "method": "TrajPD",
        "objective": "outcome_plus_coupled_projection",
        "settings": OmegaConf.to_container(config.algorithm.trajpd, resolve=True),
        "student": str(Path(config.actor_rollout_ref.model.path).resolve()),
        "teacher": str(Path(config.distillation.teacher_models.teacher_model.model_path).resolve()),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.glob("*.py"))},
    }
    run_dir = Path(config.trainer.default_local_dir)
    path = run_dir / "trajpd_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("identity") != identity:
            raise ValueError("TrajPD run identity changed; use a new run directory, not an incompatible resume")
        return path
    if str(config.trainer.resume_mode) != "disable":
        raise ValueError("TrajPD resume requires its matching run manifest; legacy checkpoints cannot resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity}
    with path.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path
