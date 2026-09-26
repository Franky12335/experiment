"""Run data preparation, the paper method, model export, and held-out evaluation."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import urllib.error
import urllib.request
from omegaconf import OmegaConf
from src.paths import PROJECT_ROOT, local_model, project_path
from src.seed import configured_seed

def validate_sandbox_endpoint(sandbox_url: str) -> None:
    """Execute one real request without HTTP proxies before starting a GPU stage."""
    payload = json.dumps(
        {
            "compile_timeout": 5,
            "run_timeout": 10,
            "code": (
                "import numpy as np, sympy as sp\n"
                "from scipy.optimize import brentq\n"
                "assert np.sqrt(4) == 2\n"
                "x = sp.Symbol('x')\n"
                "assert sp.expand(sp.factor(x*x-1)) == x*x-1\n"
                "assert abs(brentq(lambda x: x*x-4, 0, 3)-2) < 1e-8\n"
                "print('sandbox-preflight-ok')"
            ),
            "stdin": None,
            "memory_limit_MB": 1024,
            "language": "python",
            "files": {},
            "fetch_files": [],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        sandbox_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=20) as result:
            response = json.loads(result.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"SandboxFusion preflight failed for {sandbox_url!r}: {error}"
        ) from error
    if not isinstance(response, dict):
        raise RuntimeError(
            f"SandboxFusion preflight returned a non-object response: {response!r}"
        )
    run_result = response.get("run_result")
    if (
        response.get("status") != "Success"
        or not isinstance(run_result, dict)
        or run_result.get("return_code") != 0
        or "sandbox-preflight-ok" not in str(run_result.get("stdout", ""))
    ):
        raise RuntimeError(
            f"SandboxFusion preflight returned an invalid execution response: {response!r}"
        )

def run_module(name, *arguments, env=None):
    subprocess.run([sys.executable, "-m", name, *arguments], cwd=PROJECT_ROOT, env=env, check=True)


def export_actor(checkpoint: Path, target: Path) -> None:
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Actor checkpoint missing: {checkpoint}")
    merger = FSDPModelMerger(ModelMergerConfig(
        operation="merge", backend="fsdp",
        local_dir=str(checkpoint), hf_model_config_path=str(checkpoint / "huggingface"),
        target_dir=str(target),
    ))
    merger.merge_and_save()
    merger.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "prepare", "train", "evaluate", "student-sft", "teacher-sft", "teacher-grpo"], default="all")
    parser.add_argument("--base-model", help="Local base model for SFT, or the completed Teacher SFT model for GRPO")
    args = parser.parse_args()
    config = OmegaConf.load(PROJECT_ROOT / "configs/main.yaml")
    device_ids = [item.strip() for item in str(config.cuda_visible_devices).split(",") if item.strip()]
    if not device_ids or len(device_ids) != len(set(device_ids)):
        raise ValueError("cuda_visible_devices must list distinct GPU IDs")
    gpu_count = len(device_ids)
    seed = configured_seed()
    env = dict(os.environ)
    env.update(RUN_SEED=str(seed), PYTHONHASHSEED=str(seed),
               CUDA_VISIBLE_DEVICES=str(config.cuda_visible_devices), NUM_GPUS=str(gpu_count),
               NO_PROXY="localhost,127.0.0.1,::1", no_proxy="localhost,127.0.0.1,::1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", CUBLAS_WORKSPACE_CONFIG=":4096:8",
               PYTHONDONTWRITEBYTECODE="1", VERL_USE_EXTERNAL_PLUGINS="none")
    if args.stage in ("all", "prepare"):
        run_module("src.data_prep.prepare_experiment_data", env=env)
        if args.stage == "prepare":
            return
    if args.stage in ("all", "train"):
        env["STUDENT_MODEL_PATH"] = str(local_model(config.student_model))
        env["TEACHER_MODEL_PATH"] = str(local_model(config.teacher_model))
    if args.stage == "evaluate":
        local_model(config.exported_model)
    if args.stage in ("student-sft", "teacher-sft", "teacher-grpo") and not args.base_model:
        parser.error("--base-model is required for model preparation")
    if args.stage in ("student-sft", "teacher-sft"):
        stage_name = args.stage.replace("-", "_")
        env.update(SFT_BASE_MODEL_PATH=str(local_model(args.base_model)),
                   SFT_OUTPUT_DIR=str(PROJECT_ROOT / "checkpoints" / stage_name),
                   SFT_RUN_NAME=stage_name, SFT_SEQUENCE_PARALLEL_SIZE="1")
        run_module("torch.distributed.run", "--standalone", f"--nproc_per_node={gpu_count}", "-m", "src.training.main_sft", env=env)
        return
    endpoint = f"http://{config.sandbox_host}:{config.sandbox_port}/run_code"
    env["OQ_SANDBOX_URL"] = endpoint
    sandbox_env = dict(env, PYTHONPATH=str(PROJECT_ROOT / "SandboxFusion"))
    sandbox_python = Path(os.path.abspath(PROJECT_ROOT / str(config.sandbox_python)))
    tool_python = Path(os.path.abspath(PROJECT_ROOT / str(config.tool_python)))
    for executable in (sandbox_python, tool_python):
        if not executable.is_file():
            raise FileNotFoundError(f"Missing execution interpreter: {executable}; follow README installation")
    sandbox_env["SANDBOX_RUNTIME_PYTHON"] = str(tool_python)
    server = subprocess.Popen([sandbox_python, "-m", "uvicorn", "sandbox.server.server:app",
                               "--host", str(config.sandbox_host), "--port", str(config.sandbox_port)],
                              cwd=PROJECT_ROOT, env=sandbox_env)
    try:
        last_error = None
        for _ in range(60):
            if server.poll() is not None:
                raise RuntimeError("The local Python execution service exited during startup")
            try:
                validate_sandbox_endpoint(endpoint)
                last_error = None
                break
            except RuntimeError as exc:
                last_error = exc
                time.sleep(1)
        if last_error is not None:
            raise last_error
        if args.stage == "teacher-grpo":
            env.update(TEACHER_SFT_MODEL_PATH=str(local_model(args.base_model)), ROLLOUT_TP_SIZE="1")
            run_module("src.training.main_grpo", env=env)
            teacher_steps = int(OmegaConf.load(PROJECT_ROOT / "configs/training/grpo.yaml").trainer.total_training_steps)
            checkpoint = PROJECT_ROOT / "checkpoints/teacher_grpo" / f"global_step_{teacher_steps}/actor"
            export_actor(checkpoint, project_path(config.teacher_model))
            return
        if args.stage in ("all", "train"):
            env.update(DISTILLATION_RUN_DIR=str(project_path(config.training_output)),
                       ARTIFACT_DIR=str(PROJECT_ROOT / "artifacts/main"))
            run_module("src.trajpd.main", env=env)
            training_steps = int(OmegaConf.load(PROJECT_ROOT / "configs/training/trajpd.yaml").trainer.total_training_steps)
            checkpoint = project_path(config.training_output) / f"global_step_{training_steps}/actor"
            export_actor(checkpoint, project_path(config.exported_model))
        if args.stage in ("all", "evaluate"):
            benchmarks = OmegaConf.load(PROJECT_ROOT / "configs/evaluation/default.yaml").evaluation.benchmarks
            for benchmark in benchmarks:
                env.update(EVAL_METHOD="trajpd", EVAL_MODEL_PATH=str(local_model(config.exported_model)),
                           EVAL_TP_SIZE="1", EVAL_BENCHMARK=benchmark,
                           EVAL_DATA_PATH=str(PROJECT_ROOT / "data/prepared/eval" / f"{benchmark}.parquet"),
                           EVAL_GENERATION_DIR=str(PROJECT_ROOT / "evaluation" / benchmark / "trajpd/main"))
                run_module("src.evaluation.pipeline", env=env)
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

if __name__ == "__main__":
    main()
