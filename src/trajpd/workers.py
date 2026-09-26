"""TrajPD worker binding and full-update invariants."""
from functools import partial

import torch
from verl.single_controller.base.decorator import (
    Dispatch, make_nd_compute_dataproto_dispatch_fn, register,
)
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from src.trajpd.decision_engine import TrajPDFSDPEngine
from src.trajpd.policy_loss import compute_trajpd_loss


def validate_actor_update(data, *, dp_size: int, previous_version: int | None) -> int:
    """Reject replay, local re-normalization, or multiple optimizer mini-batches."""
    version = int(tu.get_non_tensor_data(data, "policy_version", default=None))
    global_trajectory_count = int(tu.get_non_tensor_data(data, "global_trajectory_count", default=None))
    if version < 0 or (previous_version is not None and version != previous_version + 1):
        raise ValueError("TrajPD update must consume the next fresh behavior-policy version")
    if global_trajectory_count < 2 or int(data.shape[0]) * dp_size != global_trajectory_count:
        raise ValueError("TrajPD requires the complete unpadded global trajectory batch")
    if int(tu.get_non_tensor_data(data, "mini_batch_size", default=None)) != global_trajectory_count:
        raise ValueError("TrajPD requires exactly one optimizer mini-batch")
    if int(tu.get_non_tensor_data(data, "epochs", default=None)) != 1:
        raise ValueError("TrajPD forbids replaying a batch for multiple epochs")
    if tu.get_non_tensor_data(data, "dataloader_kwargs", default={}).get("shuffle", False):
        raise ValueError("TrajPD actor batch must not be reshuffled")
    return version


def validate_deterministic_actor(module: torch.nn.Module) -> None:
    """Training and frozen target construction must use a dropout-free policy."""
    for name, child in module.named_modules():
        if isinstance(child, torch.nn.modules.dropout._DropoutNd) and child.p != 0:
            raise ValueError(f"TrajPD requires dropout=0: {name} has p={child.p}")
        config = getattr(child, "config", None)
        for field in ("attention_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob", "resid_pdrop", "embd_pdrop", "attn_pdrop"):
            value = getattr(config, field, 0.0)
            if value is not None and float(value) != 0.0:
                raise ValueError(f"TrajPD requires {field}=0, found {value} in {name}")


class TrajPDTrainingWorker(TrainingWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        callback = loss_fn
        while isinstance(callback, partial):
            callback = callback.func
        if type(self.engine) is not TrajPDFSDPEngine or callback is not compute_trajpd_loss:
            raise RuntimeError("TrajPD requires its own dense engine and exactly its two-term loss")
        self.loss_fn = loss_fn

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data):
        version = validate_actor_update(
            data, dp_size=self.engine.get_data_parallel_size(),
            previous_version=getattr(self, "_last_trajpd_update_version", None),
        )
        validate_deterministic_actor(self.engine.module)
        # Mark before execution: failed/partial updates must abort, never retry on changed weights.
        self._last_trajpd_update_version = version
        return super().train_mini_batch(data)


class TrajPDActorRolloutRefWorker(ActorRolloutRefWorker):
    actor_worker_cls = TrajPDTrainingWorker

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if "actor" in self.role:
            self.loss_fn = self.actor.loss_fn
