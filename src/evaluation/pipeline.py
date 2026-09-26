"""Run generation and aggregate held-out metrics."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from src.paths import PROJECT_ROOT, output_path
from src.evaluation.report import aggregate_evaluation_metrics


def run_evaluation(extra_args: list[str] | None = None) -> dict:
    from src.evaluation.identity import validate_identity
    method = validate_identity(os.environ["EVAL_METHOD"], os.environ.get("EVAL_REPORT_METHOD"))

    benchmark = os.environ.get("EVAL_BENCHMARK")
    run_dir = PROJECT_ROOT / "evaluation" / (benchmark or "summaries") / method / "new_run"
    generation_dir = Path(
        os.environ.get(
            "EVAL_GENERATION_DIR",
            str(run_dir),
        )
    )
    generation_dir = output_path(generation_dir, "evaluation")
    os.environ["EVAL_GENERATION_DIR"] = str(generation_dir)
    generation_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.main_evaluation",
            *(extra_args or []),
        ],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=True,
    )
    return aggregate_saved(method, generation_dir)


def aggregate_saved(method, generation_dir):
    """Aggregate committed rows without launching model generation."""
    import time
    from src.evaluation.observability import atomic_json
    from src.evaluation.identity import validate_identity
    validate_identity(method, os.environ.get("EVAL_REPORT_METHOD"))
    generation_dir = output_path(generation_dir, "evaluation")
    progress_path = generation_dir / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    progress.update(status="aggregating", updated_at=time.time(), eta_seconds=None)
    atomic_json(progress_path, progress)
    try:
        result = aggregate_evaluation_metrics(
            method, generation_dir, generation_dir / "evaluation_config.yaml", generation_dir / "metrics.json"
        )
        result["report_method"] = method
        atomic_json(generation_dir / "metrics.json", result)
    except Exception as exc:
        progress.update(status="failed", updated_at=time.time(), eta_seconds=None, error=str(exc))
        atomic_json(progress_path, progress)
        raise
    progress.pop("error", None)
    progress.update(status="completed", updated_at=time.time(), eta_seconds=0, eta_range_seconds=[0, 0])
    atomic_json(progress_path, progress)
    return result



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run held-out generation and metric aggregation"
    )
    _, hydra_args = parser.parse_known_args()
    result = run_evaluation(hydra_args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
