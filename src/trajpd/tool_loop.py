"""Configurable public-budget interaction with exact sampled-token accounting."""
from __future__ import annotations
import re
import asyncio
import json
from typing import Any
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.utils.profiler import simple_timer
from src.trajpd.schema import DECISION_PREFIX_SNAPSHOTS, DecisionPrefixSnapshot
from src.trajpd.tool_parser import Qwen3ToolParser


def merge_generation_policy_version(trajectory_fields: dict, generation_fields: dict) -> int:
    """Every assistant turn must identify one unchanged, nonnegative policy version."""
    from numbers import Integral

    def version_pair(fields: dict, source: str) -> int:
        values = [fields.get(name) for name in ("min_global_steps", "max_global_steps")]
        if any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0 for value in values):
            raise ValueError(f"{source} requires explicit nonnegative integer policy versions")
        if values[0] != values[1]:
            raise ValueError(f"{source} crossed a policy update")
        return int(values[0])

    version = version_pair(generation_fields, "each generation")
    if "min_global_steps" in trajectory_fields or "max_global_steps" in trajectory_fields:
        if version_pair(trajectory_fields, "trajectory") != version:
            raise ValueError("one trajectory crossed a policy update between assistant turns")
    trajectory_fields.update(min_global_steps=version, max_global_steps=version)
    return version


def budget_prompt(messages: list[dict], budget: Any) -> list[dict]:
    """Use the same budget instruction in preflight and rollout to keep token counts and model-visible constraints aligned."""
    result = [dict(message) for message in messages]
    from src.data_prep import canonical_prompt as source_format
    for message in result:
        if message.get("role") != "user":
            continue
        text = str(message["content"])
        for name in ("_CODE_INSTRUCTION", "_LEGACY_CODE_SUFFIX", "_ANSWER_INSTRUCTION", "_MATH_INSTRUCTION", "_MATH_PREFIX"):
            wrapper = getattr(source_format, name, "")
            if wrapper:
                text = text.replace(wrapper, "")
        if "Aline tore a sheet of paper into three pieces" in text:
            text = text.replace(
                "Can she, at some point, end up with a total of 2020 pieces? 2023 pieces?",
                "Which of the candidate totals, 2020 or 2023 pieces, can she reach? Give the achievable candidate total as your final answer.",
            )
        message["content"] = text
    instruction = (
        f"Interaction budget: at most {budget.max_assistant_turns} assistant turns and {budget.max_tool_calls} tool calls. "
        f"Each assistant turn has at most {budget.max_new_tokens_per_turn} generated tokens. "
        f"The trajectory has at most {budget.max_generated_tokens} generated tokens total; "
        f"the final-answer phase has at most {budget.final_answer_max_new_tokens} tokens. "
        "Tools are optional. Answer immediately when the solution is established. "
        "Do not repeatedly recheck an established result. At most one tool call per turn. "
        "After the last tool result, submit the final answer without calling tools. "
        "For math, put the final answer in \\boxed{} using the answer type requested "
        "by the problem; do not include measurement units. For programming tasks, "
        "submit one complete executable Python program in a closed Markdown Python code block."
    )
    if result and result[0].get("role") == "system":
        result[0]["content"] = str(result[0]["content"]) + "\n\n" + instruction
    else:
        result.insert(0, {"role": "system", "content": instruction})
    return result


class ProjectToolAgentLoop(ToolAgentLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget = self.data_config.interaction_budget
        if self.max_assistant_turns != self.budget.max_assistant_turns or not 0 <= self.budget.max_tool_calls < self.max_assistant_turns:
            raise ValueError("project requires matching turn limits and a reserved final assistant turn")
        if self.max_parallel_calls != 1 or not self.tools:
            raise ValueError("project requires one call per tool action")
        self.tool_parser = Qwen3ToolParser(self.tokenizer, 1)
        self.eos_ids = {int(self.tokenizer.convert_tokens_to_ids(s)) for s in ("<|im_end|>", "<|endoftext|>")}
        self.im_end_id = int(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        self.max_tool_response_length = 2**63 - 1

    async def run(self, sampling_params: dict[str, Any], **kwargs):
        self.initial_token_cache = kwargs.pop("initial_token_cache", {})
        if "session_id" not in kwargs or "privileged_context" in kwargs:
            raise ValueError("public rollout requires session_id and no privileged context")
        params = {**sampling_params, "sampling_seed": int(self.rollout_config.seed) + int(kwargs["session_id"])}
        return await super().run(params, **kwargs)

    async def _handle_pending_state(self, data, sampling_params):
        data.messages = budget_prompt(data.messages, self.budget)
        data.extra_fields["rollout_messages"] = data.messages
        data.extra_fields.update(tool_calls_used=0, generated_tokens=0, observation_tokens=0,
                                 final_phase=False, protocol_errors=[], turn_token_counts=[])
        schemas = getattr(data, "_active_tool_schemas", self.tool_schemas)
        key = (json.dumps(data.messages, ensure_ascii=False, sort_keys=True), repr(schemas))
        cache = self.initial_token_cache
        if key not in cache:
            if len(cache) >= 64:
                cache.pop(next(iter(cache)))
            cache[key] = self.loop.run_in_executor(
                None, lambda: self.continuous_token_builder.build_initial_tokens(data.messages, tools=schemas)
            )
        data.prompt_ids = list(await asyncio.shield(cache[key]))
        if len(data.prompt_ids) > self.prompt_length:
            raise ValueError("public budget prompt exceeds prompt capacity")
        return AgentState.GENERATING

    def _control(self, data: AgentData, detail: str = "") -> dict:
        remaining = self.budget.max_tool_calls - data.extra_fields["tool_calls_used"]
        text = ("Final-answer phase: no further tool calls are allowed. Submit the final answer now."
                if data.extra_fields["final_phase"] else
                f"Budget remaining: {remaining} tool calls; {self.max_assistant_turns-data.assistant_turns} assistant turns. "
                "Answer now if ready; tools are optional.")
        tokens_left = self.budget.max_generated_tokens - data.extra_fields["generated_tokens"]
        text += f" Remaining generated-token budget: {tokens_left}; final-answer allowance: {self.budget.final_answer_max_new_tokens}."
        return {"role": "user", "content": f"{detail}\n{text}".strip()}

    async def _append_context(self, data: AgentData, messages: list[dict]) -> None:
        if data.prompt_ids[-1] != self.im_end_id:
            data.prompt_ids.append(self.im_end_id)
            data.response_mask.append(0)
            if data.response_logprobs:
                data.response_logprobs.append(0.0)
        previous = list(data.messages)
        data.messages.extend(messages)
        merged, mask, logprobs = await self.ct_merge_non_assistant_msg(
            previous, data.messages, data.prompt_ids, data.response_mask,
            data.response_logprobs or None,
            tools=getattr(data, "_active_tool_schemas", self.tool_schemas),
        )
        data.prompt_ids, data.response_mask = merged.token_ids, mask
        if data.response_logprobs:
            data.response_logprobs = logprobs or []
        overhead = len(mask)-data.extra_fields["generated_tokens"]-data.extra_fields["observation_tokens"]
        if overhead > self.budget.max_protocol_tokens or len(mask)+self.budget.final_answer_max_new_tokens > self.response_length:
            raise ValueError("control context exhausted reserved final-answer capacity")

    async def _recover_action(self, data, reason):
        data.extra_fields["protocol_errors"].append(reason)
        if data.extra_fields["final_phase"] or data.assistant_turns >= self.max_assistant_turns:
            data.extra_fields.update(termination_reason=reason, final_answer_text="")
            return AgentState.TERMINATED
        remaining = self.budget.max_generated_tokens - data.extra_fields["generated_tokens"]
        data.extra_fields["final_phase"] = (
            data.extra_fields["tool_calls_used"] >= self.budget.max_tool_calls
            or data.assistant_turns >= self.max_assistant_turns - 1
            or remaining <= self.budget.final_answer_max_new_tokens
        )
        detail = (
            "The previous assistant turn reached its token limit. No tool call from "
            "that turn was executed. Continue from the existing work without restarting; "
            "submit the answer if established."
            if reason == "turn_length_cutoff" else
            f"Previous action was not executed: {reason}. Correct the action or submit the final answer."
        )
        await self._append_context(data, [self._control(data, detail)])
        return AgentState.GENERATING

    async def _handle_generating_state(self, data, sampling_params, ignore_termination=False):
        if data.image_data or data.video_data or data.audio_data:
            raise ValueError("project interaction is text-only")
        final = bool(data.extra_fields["final_phase"])
        remaining = self.budget.max_generated_tokens-data.extra_fields["generated_tokens"]
        reserve = 0 if final else self.budget.final_answer_max_new_tokens
        capacity = min(self.response_length - len(data.response_mask),
                       int(self.rollout_config.max_model_len) - len(data.prompt_ids))
        if not final:
            protocol_used = len(data.response_mask) - data.extra_fields["generated_tokens"] - data.extra_fields["observation_tokens"]
            capacity -= self.budget.max_tool_observation_tokens + max(0, self.budget.max_protocol_tokens - protocol_used)
        allowance = min(self.budget.final_answer_max_new_tokens if final else self.budget.max_new_tokens_per_turn,
                        remaining - reserve, capacity - reserve)
        if allowance < 1 or data.assistant_turns >= self.max_assistant_turns:
            raise ValueError("generation requested without reserved turn/token capacity")
        start = len(data.response_mask)
        initial_prompt = len(data.prompt_ids)-start
        with simple_timer("generate_sequences", data.metrics):
            output = await self.server_manager.generate(
                request_id=data.request_id, prompt_ids=data.prompt_ids,
                sampling_params={**sampling_params, "max_new_tokens": allowance},
                image_data=None, video_data=None, audio_data=None,
                mm_processor_kwargs=data.mm_processor_kwargs,
            )
        if not output.token_ids or len(output.token_ids) > allowance:
            raise ValueError("generation violated its per-turn token budget")
        if output.stop_reason in {"abort", "aborted"}:
            raise RuntimeError("inference aborted; not a trainable task failure")
        data.metrics["num_preempted"] = max(0, data.metrics.get("num_preempted", 0)) + max(0, output.num_preempted or 0)
        merge_generation_policy_version(data.extra_fields, output.extra_fields)
        data.assistant_turns += 1
        data.response_ids = output.token_ids
        merged, mask, logprobs = await self.ct_merge_assistant_token(
            data.prompt_ids, output.token_ids, data.response_mask,
            data.response_logprobs if output.log_probs else None, assistant_logprobs=output.log_probs,
        )
        data.prompt_ids, data.response_mask = merged.token_ids, mask
        if logprobs is not None:
            data.response_logprobs = logprobs
        data.extra_fields["generated_tokens"] += len(output.token_ids)
        data.extra_fields["turn_token_counts"].append(len(output.token_ids))
        snapshots = data.extra_fields.setdefault(DECISION_PREFIX_SNAPSHOTS, [])
        snap = dict(decision_index=len(snapshots), initial_prompt_length=initial_prompt,
                    response_start=start, response_stop=len(mask))
        DecisionPrefixSnapshot.create_from_mapping(snap)
        snapshots.append(snap)
        active = getattr(data, "_active_tools", self.tools)
        content, data.tool_calls = await self.tool_parser.extract_tool_calls(output.token_ids, [t.tool_schema for t in active.values()])
        data.messages.append(self._build_assistant_message(content, data))
        text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
        data.extra_fields["wait_count"] = data.extra_fields.get("wait_count", 0) + len(re.findall(r"\bwait\b", text, re.I))
        cutoff = output.token_ids[-1] not in self.eos_ids and (
            len(output.token_ids) >= allowance or output.stop_reason == "length"
        )
        if output.token_ids[-1] not in self.eos_ids and not cutoff:
            raise RuntimeError("generation stopped without EOS or an explicit length boundary")
        if cutoff:
            data.extra_fields["turn_length_cutoffs"] = data.extra_fields.get("turn_length_cutoffs", 0)+1
            if final:
                data.extra_fields.update(termination_reason="final_length_cutoff", final_answer_text=text)
                return AgentState.TERMINATED
            return await self._recover_action(data, "turn_length_cutoff")
        if self.tool_parser.failure:
            return await self._recover_action(data, self.tool_parser.failure)
        if data.tool_calls:
            if final or data.extra_fields["tool_calls_used"] >= self.budget.max_tool_calls or data.assistant_turns >= self.max_assistant_turns:
                return await self._recover_action(data, "final_phase_tool_call")
            return AgentState.PROCESSING_TOOLS
        data.extra_fields.update(termination_reason="completed", final_answer_text=text)
        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, data):
        if len(data.tool_calls) != 1:
            raise ValueError("tool execution requires exactly one parsed call")
        data.extra_fields["tool_calls_used"] += 1
        with simple_timer("tool_calls", data.metrics):
            response, reward, metrics = await super()._call_tool(data.tool_calls[0], data.tools_kwargs, data)
        data.extra_fields.setdefault("tool_events", []).append({
            "assistant_turn": data.assistant_turns,
            "raw_response_text": response.text or "",
            "metrics": metrics, "reward": reward,
        })
        status = metrics.get("tool_status", "execution_failure")
        data.extra_fields.setdefault("tool_statuses", []).append(status)
        if status == "service_failure":
            raise RuntimeError(
                f"tool infrastructure failed; not a student task error: {metrics['tool_error']}"
            )
        if status != "success":
            data.extra_fields["protocol_errors"].append(str(status))
        if response.image or response.video:
            raise ValueError("tool response must be text-only")
        limit = self.budget.max_tool_observation_tokens
        backend = self.tokenizer.backend_tokenizer
        if backend.truncation is not None or backend.padding is not None:
            raise RuntimeError("tool observation requires an unpadded, untruncated tokenizer")
        encoding = backend.encode(response.text or "", add_special_tokens=False)
        if len(encoding) > limit:
            marker = self.tokenizer.encode("\n[tool output truncated]\n", add_special_tokens=False)
            if len(marker) >= limit:
                raise ValueError("tool observation budget must exceed truncation marker")
            side = (limit-len(marker))//2
            tail_size = limit-len(marker)-side
            encoding.truncate(limit, stride=0, direction="right")
            chunks = [encoding, *encoding.overflowing]
            tail_parts = []
            remaining = tail_size
            for chunk in reversed(chunks):
                part = chunk.ids[-remaining:]
                tail_parts.append(part)
                remaining -= len(part)
                if remaining == 0:
                    break
            ids = encoding.ids[:side]+marker+[token for part in reversed(tail_parts) for token in part]
        else:
            ids = encoding.ids
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        count = len(backend.encode(text, add_special_tokens=False))
        while count > limit:
            ids = ids[:-1]
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            count = len(backend.encode(text, add_special_tokens=False))
        data.extra_fields["observation_tokens"] += count
        data.tool_rewards.append(reward or 0.0)
        data.user_turns += 1
        data.extra_fields["final_phase"] = (
            data.extra_fields["tool_calls_used"] >= self.budget.max_tool_calls or data.assistant_turns >= self.max_assistant_turns - 1
            or self.budget.max_generated_tokens - data.extra_fields["generated_tokens"]
            <= self.budget.final_answer_max_new_tokens
        )
        message = {"role": "tool", "content": text}
        if data.tool_calls[0].tool_call_id is not None:
            message["tool_call_id"] = data.tool_calls[0].tool_call_id
        await self._append_context(data, [message, self._control(data)])
        return AgentState.GENERATING
