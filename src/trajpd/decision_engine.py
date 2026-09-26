"""Project Student vocabulary logits at all assistant-token prediction positions."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed.tensor import DTensor

from verl.utils import tensordict_utils
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def compute_packed_active_prediction_positions(data: Any) -> torch.Tensor:
    input_ids = data["input_ids"]
    responses = data["responses"]
    response_mask = data["response_mask"]
    positions: list[torch.Tensor] = []
    sequence_offset = 0
    for row_ids, row_responses, row_mask in zip(
        input_ids.unbind(), responses.unbind(), response_mask.unbind(), strict=True
    ):
        prompt_length = row_ids.numel() - row_responses.numel()
        active = torch.nonzero(row_mask.to(torch.bool), as_tuple=False).flatten()
        positions.append(sequence_offset + prompt_length - 1 + active)
        sequence_offset += row_ids.numel()
    return torch.cat(positions).to(device=input_ids.values().device, dtype=torch.long)


@EngineRegistry.register(
    model_type="language_model",
    backend="fsdp2",
    device="cuda",
    vendor="nvidia",
)
class TrajPDFSDPEngine(FSDPEngineWithLMHead):

    def prepare_model_inputs(self, micro_batch):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        enabled = tensordict_utils.get_non_tensor_data(
            data=micro_batch,
            key="active_response_logits",
            default=False,
        )
        if not enabled:
            if not self.engine_config.forward_only:
                raise RuntimeError("TrajPD actor cannot fall back to the generic logits/loss path")
            return model_inputs, output_args
        if not tensordict_utils.get_non_tensor_data(
            data=micro_batch, key="use_remove_padding", default=True
        ):
            raise ValueError("selected logits require remove-padding execution")
        if self.ulysses_sequence_parallel_size != 1:
            raise ValueError("selected logits require Ulysses size 1")
        if int(output_args.get("pad_size", 0)) != 0:
            raise ValueError("packed action inputs cannot contain length padding")
        if tensordict_utils.get_non_tensor_data(
            data=micro_batch, key="use_fused_kernels", default=False
        ):
            raise ValueError("selected logits require the unfused Qwen3 LM head")
        active = compute_packed_active_prediction_positions(micro_batch)
        model_inputs["logits_to_keep"] = active
        output_args["active_prediction_positions"] = active
        output_args["active_total_input_tokens"] = int(
            micro_batch["input_ids"].values().numel()
        )
        return model_inputs, output_args

    def prepare_model_outputs(
        self,
        output,
        output_args,
        micro_batch,
        logits_processor_func,
    ):
        active = output_args.get("active_prediction_positions")
        if active is None:
            return super().prepare_model_outputs(
                output, output_args, micro_batch, logits_processor_func
            )
        if logits_processor_func is None:
            raise ValueError(
                "objective-preserving distillation requires a custom loss processor"
            )
        logits = output.logits.squeeze(0)
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        if logits.shape[0] != active.numel():
            raise ValueError("engine did not return the active-position logits")
        temperature = tensordict_utils.get_non_tensor_data(
            data=micro_batch, key="temperature", default=None
        )
        if not isinstance(temperature, (int, float)) or float(temperature) != 1.0:
            raise ValueError("TrajPD dense objective requires scalar temperature=1")
        outputs = logits_processor_func(
            student_logits=logits.unsqueeze(0),
            data=micro_batch,
            selected_prediction_positions=active,
        )
        total_tokens = int(output_args["active_total_input_tokens"])
        offsets = micro_batch["input_ids"].offsets()
        result: dict[str, torch.Tensor] = {}
        for name, value in outputs.items():
            value = value.squeeze(0)
            metric = name not in {"joint_loss_contributions", "outcome_loss_contributions", "distillation_loss_contributions"}
            if metric:
                metric_offsets = torch.arange(offsets.numel(), device=value.device)
                result[name] = torch.nested.nested_tensor_from_jagged(value, metric_offsets)
                continue
            if value.shape != (total_tokens,):
                raise ValueError(f"loss processor output {name!r} has an invalid shape")
            result[name] = torch.nested.nested_tensor_from_jagged(value, offsets)
        return result
