"""Evaluation-only durable trajectories, review cases, and progress reporting."""
from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from collections import Counter, deque
from pathlib import Path

import ray
from omegaconf import OmegaConf
from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQ
from src.evaluation.tool_loop import EvaluationToolAgentLoop
from src.paths import PROJECT_ROOT, output_path


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Cannot save {type(value).__name__}")


def atomic_json(path, value, *, exclusive=False, jsonl=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, default=json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def locations(config):
    evaluation = output_path(config.trainer.validation_data_dir, "evaluation")
    relative = evaluation.relative_to(PROJECT_ROOT / "evaluation")
    return evaluation, PROJECT_ROOT / "artifacts/evaluation" / relative


def prepare_manifest(config):
    evaluation, artifacts = locations(config)
    protocol = {k: OmegaConf.to_container(config[k], resolve=True) for k in ("data", "evaluation", "reward")}
    protocol["evaluation"].pop("resume", None)
    protocol["rollout"] = OmegaConf.to_container(config.actor_rollout_ref.rollout, resolve=True)
    model = Path(config.actor_rollout_ref.model.path).resolve()
    protocol["model"] = str(model)
    protocol["weights"] = [{"name": p.name, "size_bytes": p.stat().st_size}
                           for p in sorted(model.glob("*.safetensors")) + sorted(model.glob("pytorch_model*.bin"))]
    protocol["data_files"] = [str(Path(p).resolve()) for p in config.data.val_files]
    path = evaluation / "resume_manifest.json"
    if path.exists():
        if not config.evaluation.get("resume", False):
            raise FileExistsError("Use --resume for a compatible interrupted evaluation")
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Resume model path, weight sizes, data paths or evaluation configuration differs")
    else:
        if any(evaluation.rglob("*.jsonl")):
            raise ValueError("Cannot resume unmanifested generations")
        atomic_json(path, protocol, exclusive=True)
    for name in ("success", "failure", "path_correct_answer_wrong", "path_wrong_answer_correct", "pending"):
        (artifacts / "cases/reviewed" / name).mkdir(parents=True, exist_ok=True)
    return evaluation, artifacts


class ObservedToolAgentLoop(EvaluationToolAgentLoop):
    async def _handle_pending_state(self, data, sampling_params):
        self.observed_data = data
        return await super()._handle_pending_state(data, sampling_params)

    async def run(self, sampling_params, **kwargs):
        output = await super().run(sampling_params, **kwargs)
        output.extra_fields["evaluation_messages"] = copy.deepcopy(self.observed_data.messages)
        return output


WorkerBase = AgentLoopWorkerTQ.__ray_metadata__.modified_class


class DurableEvaluationWorker(WorkerBase):
    async def configure_evaluation(self, config):
        import asyncio
        pending = list(getattr(self, "background_tasks", []))
        if pending:
            await asyncio.gather(*pending)
        self.config = config

    def _paths(self, kwargs):
        evaluation, artifacts = locations(self.config)
        source_index = int(kwargs["extra_info"]["source_index"])
        session_id = int(kwargs["session_id"])
        key = f"{source_index:06d}_{session_id:03d}"
        return key, evaluation / "trajectories" / f"{key}.jsonl", artifacts / "cases/pending_path_review" / f"step_{key}.jsonl"

    async def _run_agent_loop(self, sampling_params, trajectory, **kwargs):
        key, final, case = self._paths(kwargs)
        if final.exists():
            if not self.config.evaluation.get("resume", False):
                raise FileExistsError(final)
            saved = json.loads(case.read_text())
            restored = AgentLoopOutput.model_validate(saved["resume_output"])
            return await super()._agent_loop_postprocess(restored, trajectory["validate"], **kwargs)
        try:
            return await super()._run_agent_loop(sampling_params, trajectory, **kwargs)
        except Exception as error:
            _, artifacts = locations(self.config)
            atomic_json(artifacts / "errors" / f"{key}.json", {"key": key, "error": repr(error), "time": time.time()})
            raise

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        if not validate or isinstance(output, list):
            raise ValueError("Durable evaluation requires one final output per trajectory")
        await super()._agent_loop_postprocess(output, validate, **kwargs)
        key, final, case = self._paths(kwargs)
        reward = dict(output.extra_fields["reward_extra_info"])
        sampling = self.config.actor_rollout_ref.rollout.val_kwargs
        reward["generation_protocol"] = {
            "temperature": float(sampling.temperature), "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k), "do_sample": bool(sampling.do_sample),
            "samples_per_problem": int(sampling.n),
        }
        benchmark = str(reward.get("benchmark", kwargs["data_source"]))
        row = {"input": self.tokenizer.decode(output.prompt_ids, skip_special_tokens=True),
               "output": self.tokenizer.decode(output.response_ids, skip_special_tokens=True),
               "gts": kwargs["reward_model"]["ground_truth"], "score": output.reward_score,
               "step": int(kwargs["global_steps"]), **reward,
               "uid": f"{benchmark}_{key}", "session_id": int(kwargs["session_id"]),
               "source_index": int(kwargs["extra_info"]["source_index"]), "completed_at": time.time()}
        case_row = {**row, "key": row["uid"], "answer_correct": bool(row["acc"]),
                    "task_return": float(row["acc"]),
                    "path_review": {"verdict": "pending", "evidence": ""},
                    "messages": output.extra_fields.get("evaluation_messages", []),
                    "tool_statuses": output.extra_fields.get("tool_statuses", []),
                    "resume_output": output.model_dump()}
        # The evaluation file is the commit marker. A crash before it only leaves an
        # uncommitted case, which can be replaced when the trajectory is generated again.
        atomic_json(case, case_row)
        atomic_json(final, row, exclusive=True, jsonl=True)
        _, artifacts = locations(self.config)
        (artifacts / "errors" / f"{key}.json").unlink(missing_ok=True)


class DurableEvaluationManager(AgentLoopManagerTQ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_loop_workers_class = ray.remote(DurableEvaluationWorker)


class MonotonicEvaluationLogger:
    """Separate metric history steps from the model's fixed evaluation step zero."""
    def __init__(self, wrapped):
        self.wrapped, self.tick = wrapped, 0
        self.lock = threading.Lock()

    def log(self, data, step, **kwargs):
        with self.lock:
            self.wrapped.log(data={**data, "evaluation/model_step": step}, step=self.tick, **kwargs)
            self.tick += 1

    def __getattr__(self, name):
        return getattr(self.wrapped, name)


class EvaluationProgress:
    def __init__(self, config, questions, logger):
        self.evaluation, self.artifacts = locations(config)
        self.questions = questions
        self.samples = int(config.evaluation.samples_per_problem)
        self.total = questions * self.samples
        self.logger = logger
        self.started = time.monotonic()
        self.history = deque(maxlen=21)
        self.initial_completed = len(list((self.evaluation / "trajectories").glob("*.jsonl")))
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error = None

    def snapshot(self, status="running"):
        files = list((self.evaluation / "trajectories").glob("*.jsonl"))
        counts = Counter(p.stem.split("_")[0] for p in files)
        completed = len(files)
        if any(count > self.samples for count in counts.values()) or completed > self.total:
            raise ValueError("Trajectory count exceeds the configured protocol")
        elapsed = time.monotonic() - self.started
        self.history.append((elapsed, completed))
        recent_time, recent_count = self.history[0]
        recent_rate = (completed - recent_count) / max(elapsed - recent_time, 1)
        average_rate = (completed - self.initial_completed) / max(elapsed, 1)
        rates = [rate for rate in (recent_rate, average_rate) if rate > 0]
        eta = (self.total - completed) / recent_rate if recent_rate > 0 and completed - self.initial_completed >= 16 else None
        if completed == self.total:
            eta = 0.0
        result = {"status": status, "updated_at": time.time(), "elapsed_seconds": elapsed,
                  "completed_trajectories": completed, "total_trajectories": self.total,
                  "completed_questions": sum(n == self.samples for n in counts.values()),
                  "total_questions": self.questions, "resumed_trajectories": self.initial_completed,
                  "percent": 100 * completed / self.total, "trajectories_per_minute": recent_rate * 60,
                  "eta_seconds": eta, "eta_range_seconds": [(self.total-completed)/max(rates), (self.total-completed)/min(rates)] if rates and eta is not None else ([0.0, 0.0] if completed == self.total else None),
                  "eta_note": "Rolling estimate, affected by long responses and tool execution; null means insufficient recent progress",
                  "failed_trajectories": len(list((self.artifacts / "errors").glob("*.json")))}
        atomic_json(self.evaluation / "progress.json", result)
        print("EVAL_PROGRESS " + json.dumps(result), flush=True)
        self.logger.log(data={"progress/" + k: v for k, v in result.items() if isinstance(v, (int, float))}, step=0)
        return result

    def _run(self):
        try:
            while not self.stop.wait(15):
                self.snapshot()
        except Exception as error:
            self.error = error

    def start(self):
        self.snapshot()
        self.thread.start()

    def finish(self, status):
        self.stop.set()
        self.thread.join(timeout=30)
        if self.error:
            raise RuntimeError("Evaluation progress writer failed") from self.error
        result = self.snapshot(status)
        if status == "generation_complete" and result["completed_trajectories"] != self.total:
            raise RuntimeError("Evaluation returned with incomplete durable trajectories")
