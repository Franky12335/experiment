"""VeRL runner for held-out generation."""

from __future__ import annotations

from typing import Any

import ray
from omegaconf import OmegaConf
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.import_utils import load_class_from_fqn
from src.evaluation.observability import (DurableEvaluationManager, EvaluationProgress, MonotonicEvaluationLogger, prepare_manifest)


def filter_scalar_reward_metrics(
    reward_extra_infos: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Variable-length traces break the scalar metric reducer; retain them only in generation records."""
    scalar_types = (str, int, float, bool)
    return {
        key: values
        for key, values in reward_extra_infos.items()
        if values and all(isinstance(value, scalar_types) for value in values)
    }


class EvaluationTrainer(PPOTrainerSync):
    def _validate(self):
        import copy
        from pathlib import Path
        from omegaconf import open_dict
        from src.evaluation.pipeline import aggregate_saved
        from src.evaluation.protocol import validate_protocol
        self.logger = MonotonicEvaluationLogger(self.logger)
        base = copy.deepcopy(self.config)
        jobs = OmegaConf.to_container(base.evaluation.get("resident_jobs", []), resolve=True)
        if not jobs:
            jobs = [{"N": int(base.evaluation.samples_per_problem),
                     "benchmark": str(base.evaluation.benchmarks[0]),
                     "output": str(base.trainer.validation_data_dir)}]
        with open_dict(base):
            base.evaluation.pop("resident_jobs", None)
        all_metrics = {}
        for job in jobs:
            destination = Path(job["output"])
            if (destination / "metrics.json").exists() and base.evaluation.resume:
                continue
            config = copy.deepcopy(base)
            n, benchmark = int(job["N"]), job["benchmark"]
            root = Path(__file__).resolve().parents[2]
            with open_dict(config):
                config.data.train_files = [str(root / "data/prepared/eval" / f"{benchmark}.parquet")]
                config.data.val_files = list(config.data.train_files)
                config.evaluation.benchmarks = [benchmark]
                config.evaluation.samples_per_problem = n
                config.evaluation.primary_metric = f"acc/mean@{n}"
                config.evaluation.pass_k = [k for k in (8,16,32,64,96,128) if k <= n]
                config.actor_rollout_ref.rollout.val_kwargs.n = n
                config.trainer.validation_data_dir = str(destination)
                config.trainer.default_local_dir = str(root / "logs/evaluation" / destination.relative_to(root / "evaluation"))
            validate_protocol(config)
            self.config = config
            destination.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(config, destination / "evaluation_config.yaml", resolve=True)
            prepare_manifest(config)
            self._init_dataloader()
            self.agent_loop_manager.config = config
            self.agent_loop_manager.rollout_config = config.actor_rollout_ref.rollout
            ray.get([w.configure_evaluation.remote(config) for w in self.agent_loop_manager.agent_loop_workers])
            print(f"EVALUATION_START {benchmark} N={n}: existing model replicas retained", flush=True)
            progress = EvaluationProgress(config, len(self.val_dataset), self.logger)
            progress.start()
            try:
                super()._validate()
                progress.finish("generation_complete")
                result = aggregate_saved(str(config.evaluation.method), destination)
                summary = result["benchmarks"][benchmark]
                metrics = {f"evaluation/{benchmark}/N{n}/Avg@{n}": summary[f"acc/mean@{n}"]}
                metrics.update({f"evaluation/{benchmark}/N{n}/Pass@{k}": summary[f"pass@{k}"]
                                for k in config.evaluation.pass_k})
                self.logger.log(data=metrics, step=self.global_steps)
                all_metrics.update(metrics)
                print(f"EVALUATION_COMPLETE {benchmark} N={n}", flush=True)
            except BaseException:
                progress.finish("failed")
                raise
        return all_metrics or {"evaluation/already_complete": 1.0}

    def _dump_generations(self, *args, **kwargs):
        # DurableEvaluationWorker already commits every final trajectory atomically.
        # Writing another complete dump here would duplicate evaluation samples.
        return None


    def _val_metrics_update(
        self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns
    ) -> dict[str, float]:
        return super()._val_metrics_update(
            data_sources,
            sample_uids,
            filter_scalar_reward_metrics(reward_extra_infos_dict),
            sample_turns,
        )


@ray.remote
class EvaluationTaskRunner:

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def initialize_agent_loop_manager(self) -> None:
        teacher_client = self.trainer.get_teacher_client()
        if teacher_client is not None:
            raise RuntimeError(
                "Formal Student inference must not initialize a Teacher client. "
                "Remove every distillation/Teacher-policy block from the evaluation config."
            )
        manager_name = self.config.actor_rollout_ref.rollout.get("agent", {}).get(
            "agent_loop_manager_class"
        )
        manager_class = (
            load_class_from_fqn(manager_name, "AgentLoopManager")
            if manager_name
            else DurableEvaluationManager
        )
        self.agent_loop_manager = manager_class.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            
            
            
            teacher_client=None,
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config) -> None:
        import transfer_queue

        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        self.config = config
        transfer_queue.init(config.transfer_queue)
        try:
            self.trainer = EvaluationTrainer(config=config)
            self.trainer.init()
            self.initialize_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
        finally:
            transfer_queue.close()
