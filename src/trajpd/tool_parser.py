"""Qwen3 tool-call parser."""

from __future__ import annotations

import json
import re
from contextvars import ContextVar

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser


class Qwen3ToolParser(ToolParser):
    start = "<tool_call>"
    end = "</tool_call>"
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def __init__(self, tokenizer, max_calls: int = 1) -> None:
        super().__init__(tokenizer)
        self.max_calls = max_calls
        
        
        
        self._failure: ContextVar[str | None] = ContextVar(
            f"qwen3_tool_parse_failure_{id(self)}", default=None
        )

    @property
    def failure(self) -> str | None:
        return self._failure.get()

    @failure.setter
    def failure(self, value: str | None) -> None:
        self._failure.set(value)

    async def extract_tool_calls(self, response_ids, tools=None):
        self.failure = None
        
        
        
        text = self.tokenizer.decode(
            response_ids, skip_special_tokens=False
        )
        starts, ends = text.count(self.start), text.count(self.end)
        if starts == ends == 0:
            return text, []
        if starts != ends:
            self.failure = "incomplete_tool_call"
            return text, []

        matches = self.pattern.findall(text)
        if len(matches) != starts:
            self.failure = "invalid_tool_boundary"
            return text, []
        if len(matches) > self.max_calls:
            self.failure = "too_many_tool_calls"
            return text, []

        available = {tool.function.name for tool in tools or []}
        calls: list[FunctionCall] = []
        for raw_call in matches:
            try:
                value = json.loads(raw_call)
            except json.JSONDecodeError:
                self.failure = "invalid_tool_json"
                return text, []
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                self.failure = "invalid_tool_call"
                return text, []
            if value["name"] not in available:
                self.failure = "unknown_tool"
                return text, []
            arguments = value.get("arguments")
            if not isinstance(arguments, dict):
                self.failure = "invalid_tool_arguments"
                return text, []
            calls.append(
                FunctionCall(
                    name=value["name"],
                    arguments=json.dumps(arguments, ensure_ascii=False),
                )
            )
        return self.pattern.sub("", text), calls
