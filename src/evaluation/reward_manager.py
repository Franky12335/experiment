"""Protocol-aware DAPO reward manager for formal agent evaluation."""

from __future__ import annotations

import inspect
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager


class ProtocolAwareDAPORewardManager(DAPORewardManager):
    """The standard DAPO manager drops tool metadata needed to distinguish protocol errors from task outcomes."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ) -> None:
        super().__init__(
            config,
            tokenizer,
            compute_score,
            reward_router_address,
            reward_model_tokenizer,
        )
        sampling = config.actor_rollout_ref.rollout.val_kwargs
        from src.evaluation.identity import validate_method
        self.evaluation_method = validate_method(str(config.evaluation.method))
        self.generation_protocol = {
            "temperature": float(sampling.temperature),
            "top_p": float(sampling.top_p),
            "top_k": int(sampling.top_k),
            "do_sample": bool(sampling.do_sample),
            "samples_per_problem": int(sampling.n),
        }

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        data_item = data[-1:][0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(
            data_item.batch["attention_mask"][-response_length:].sum().item()
        )
        if valid_response_length <= 0:
            raise ValueError("reward computation received an empty response")
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields")
        if tool_extra_fields is not None:
            extra_info.update(dict(tool_extra_fields.items()))
        extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__")
        extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get(
            "reward_scores", {}
        )
        extra_info["response_token_count"] = valid_response_length

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        score_kwargs = {
            "data_source": data_source,
            "solution_str": response_str,
            "ground_truth": ground_truth,
            "extra_info": extra_info,
            **extra_reward_kwargs,
        }
        if inspect.iscoroutinefunction(self.compute_score):
            result = await self.compute_score(**score_kwargs)
        else:
            result = await self.loop.run_in_executor(
                None, lambda: self.compute_score(**score_kwargs)
            )

        reward_extra_info: dict[str, Any] = {}
        if isinstance(result, dict):
            reward = float(result["score"])
            reward_extra_info.update(result)
        else:
            reward = float(result)
            reward_extra_info["acc"] = reward

        overlong_reward = 0.0
        overlong = False
        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            buffer_length = int(self.overlong_buffer_cfg.len)
            expected_length = int(self.max_resp_len) - buffer_length
            exceed_length = valid_response_length - expected_length
            penalty_factor = float(self.overlong_buffer_cfg.penalty_factor)
            overlong_reward = min(
                -exceed_length / buffer_length * penalty_factor,
                0.0,
            )
            overlong = overlong_reward < 0.0
            reward += overlong_reward

        
        
        reward_extra_info["response_token_count"] = valid_response_length
        reward_extra_info["overlong_reward"] = float(overlong_reward)
        reward_extra_info["overlong"] = bool(overlong)
        reward_extra_info["evaluation_method"] = self.evaluation_method
        reward_extra_info["generation_protocol"] = dict(self.generation_protocol)
        return {
            "reward_score": reward,
            "reward_extra_info": reward_extra_info,
        }
