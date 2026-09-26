"""SandboxFusion execution tool used by multi-turn rollouts."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import aiohttp
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import ToolResponse
from verl.utils.rollout_trace import rollout_trace_op


class SandboxFusionTool(BaseTool):

    def __init__(self, config: dict[str, Any], tool_schema):
        super().__init__(config, tool_schema)
        self.url = str(config["sandbox_fusion_url"])
        self.timeout = int(config["timeout"])
        self.max_attempts = int(config.get("max_attempts", 3))
        self.retry_backoff_seconds = float(config.get("retry_backoff_seconds", 1.0))
        self.max_inflight_requests = int(config.get("max_inflight_requests", 2))
        self.language = str(config["language"])
        self.memory_limit_mb = int(config["memory_limit_mb"])
        if (
            self.timeout < 1
            or self.max_attempts < 1
            or self.retry_backoff_seconds < 0
            or self.max_inflight_requests < 1
        ):
            raise ValueError("sandbox timeout/retry settings are invalid")
        self._request_semaphore = asyncio.Semaphore(self.max_inflight_requests)
        self._session: aiohttp.ClientSession | None = None

    def get_http_session(self) -> aiohttp.ClientSession:
        """Close connections after each request to avoid reusing connections already closed by the sandbox service."""
        session = getattr(self, "_session", None)
        if session is None or getattr(session, "closed", False):
            timeout = aiohttp.ClientTimeout(
                total=3 * self.timeout + 30,
                connect=10,
                sock_connect=10,
            )
            connector = aiohttp.TCPConnector(
                force_close=True,
                limit=self.max_inflight_requests,
                limit_per_host=self.max_inflight_requests,
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        """Close the shared pool only after this worker's requests have settled."""
        session = getattr(self, "_session", None)
        self._session = None
        if session is not None and not getattr(session, "closed", False):
            await session.close()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **kwargs,
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        del instance_id, kwargs
        if not isinstance(parameters, dict) or not isinstance(
            parameters.get("code"), str
        ):
            return (
                ToolResponse(text="Tool argument error: 'code' must be a string."),
                0.0,
                {"tool_status": "argument_failure"},
            )
        code = parameters["code"].strip()
        if not code:
            return (
                ToolResponse(text="Tool argument error: 'code' cannot be empty."),
                0.0,
                {"tool_status": "argument_failure"},
            )
        payload = {
            "compile_timeout": self.timeout,
            "run_timeout": self.timeout,
            "code": code,
            "stdin": None,
            "memory_limit_MB": self.memory_limit_mb,
            "language": self.language,
            "files": {},
            "fetch_files": [],
        }
        started = time.perf_counter()
        attempts_used = 0
        execution_id = uuid.uuid4().hex
        epoch = None
        submitted = False
        try:
            for attempt in range(self.max_attempts):
                attempts_used = attempt + 1
                try:
                    async with self._request_semaphore:
                        session = self.get_http_session()
                        if epoch is None:
                            async with session.get(self.url.rstrip("/") + "_session") as handshake:
                                handshake.raise_for_status()
                                epoch = (await handshake.json())["epoch"]
                        headers = {"X-Execution-Epoch": epoch, "X-Execution-ID": execution_id,
                                   "X-Execution-Retry": "1" if submitted else "0"}
                        submitted = True
                        async with session.post(self.url, json=payload, headers=headers) as result:
                            if result.status in {429, 503}:
                                submitted = False
                            result.raise_for_status()
                            response = await result.json()
                    if not isinstance(response, dict):
                        raise TypeError("sandbox response must be an object")
                    if response.get("status") == "SandboxError":
                        raise TypeError(
                            f"sandbox execution service returned SandboxError: "
                            f"{response.get('message', '')}"
                        )
                    run = response.get("run_result")
                    if not isinstance(run, dict):
                        raise TypeError("sandbox run_result must be an object")
                    if response.get("status") not in {"Success", "Failed"} or run.get("status") not in {"Finished", "TimeLimitExceeded"}:
                        raise RuntimeError(f"sandbox execution infrastructure error: {response.get('status')}/{run.get('status')}: {run.get('stderr', '')}")
                    try:
                        async with session.delete(
                            self.url.rstrip("/").rsplit("/", 1)[0] + "/run_code_result/" + execution_id,
                            headers={"X-Execution-Epoch": epoch},
                        ) as ack:
                            ack.raise_for_status()
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        pass
                    break
                except (
                    aiohttp.ClientError,
                    asyncio.TimeoutError,
                ) as current_error:
                    status = getattr(current_error, "status", None)
                    retryable = status is None or int(status) in {
                        408,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }
                    if not retryable or attempt + 1 >= self.max_attempts:
                        raise
                    await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            status = getattr(error, "status", None)
            error_name = type(error).__name__
            detail = str(error).strip()
            error_text = error_name
            if status is not None:
                error_text += f" (HTTP {status})"
            if detail:
                error_text += f": {detail}"
            if attempts_used > 1:
                error_text += f" after {attempts_used} attempts"
            return (
                ToolResponse(text=f"Tool service failure: {error_text}"),
                0.0,
                {
                    "tool_status": "service_failure",
                    "tool_error": error_text,
                    "tool_latency_s": time.perf_counter() - started,
                },
            )

        output = str(run.get("stdout") or "") + str(run.get("stderr") or "")
        if not output:
            output = (
                f"Sandbox status: {response.get('status', 'unknown')}; "
                f"run status: {run.get('status', 'unknown')}"
            )
        status = (
            "success"
            if response.get("status") == "Success" and run.get("return_code") == 0
            else "program_failure"
        )
        return (
            ToolResponse(text=output),
            0.0,
            {
                "tool_status": status,
                "sandbox_status": response.get("status"),
                "run_status": run.get("status"),
                "tool_latency_s": time.perf_counter() - started,
            },
        )
