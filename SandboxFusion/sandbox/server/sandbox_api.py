# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import asyncio
import time
import uuid
import traceback
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from sandbox.runners import (
    CODE_RUNNERS,
    CellRunResult,
    CodeRunArgs,
    CodeRunResult,
    CommandRunResult,
    CommandRunStatus,
    Language,
    RunJupyterRequest,
    run_jupyter,
)

sandbox_router = APIRouter()
logger = structlog.stdlib.get_logger()


class RunCodeRequest(BaseModel):
    compile_timeout: float = Field(10, description='compile timeout for compiled languages')
    run_timeout: float = Field(10, description='code run timeout')
    memory_limit_MB: int = Field(-1, description='maximum memory allowed in megabytes')
    code: str = Field(..., examples=['print("hello")'], description='the code to run')
    stdin: Optional[str] = Field(None, examples=[''], description='optional string to pass into stdin')
    language: Language = Field(..., examples=['python'], description='the language or execution mode to run the code')
    files: Dict[str, Optional[str]] = Field({}, description='a dict from file path to base64 encoded file content')
    fetch_files: List[str] = Field([], description='a list of file paths to fetch after code execution')


class RunStatus(str, Enum):
    # all command finished successfully
    Success = 'Success'
    # one of the process has non-zero return code
    Failed = 'Failed'
    # error on sandbox side
    SandboxError = 'SandboxError'


class RunCodeResponse(BaseModel):
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}


class RunJupyterResponse(BaseModel):
    status: RunStatus
    message: str
    driver: Optional[CommandRunResult] = None
    cells: List[CellRunResult] = []
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}


def parse_run_status(result: CodeRunResult) -> Tuple[RunStatus, str]:
    outcomes = []
    retcodes = []
    err_msgs = []
    if result.compile_result is not None:
        outcomes.append(result.compile_result.status)
        err_msgs.append(result.compile_result.stderr or '')
        if result.compile_result.return_code is not None:
            retcodes.append(result.compile_result.return_code)
    if result.run_result is not None:
        outcomes.append(result.run_result.status)
        err_msgs.append(result.run_result.stderr or '')
        if result.run_result.return_code is not None:
            retcodes.append(result.run_result.return_code)

    for o, m in zip(outcomes, err_msgs):
        if o == CommandRunStatus.Error:
            return RunStatus.SandboxError, m
    if any([o == CommandRunStatus.TimeLimitExceeded for o in outcomes]):
        return RunStatus.Failed, ''
    if any([r != 0 for r in retcodes]):
        return RunStatus.Failed, ''
    # no error, no tle and no non-zero return codes -> success
    return RunStatus.Success, ''


async def _execute_code_unbounded(request: RunCodeRequest):
    resp = RunCodeResponse(status=RunStatus.Success, message='', executor_pod_name=os.environ.get('MY_POD_NAME'))
    try:
        logger.debug(
            f'start processing {request.language} request with code ```\n{request.code[:100]}\n``` and files {list(request.files.keys())}...(memory_limit: {request.memory_limit_MB}MB)'
        )
        result = await CODE_RUNNERS[request.language](CodeRunArgs(**request.model_dump()))

        resp.compile_result = result.compile_result
        resp.run_result = result.run_result
        resp.files = result.files
        resp.status, message = parse_run_status(result)
        if resp.status == RunStatus.SandboxError:
            resp.message = message
    except Exception as e:
        message = f'exception on running code {request.code}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp


# Request identity is scoped to this server lifetime. A restarted server must
# reject old identities rather than execute an uncertain request twice.
_execution_slots = asyncio.Semaphore(32)


async def _execute_code(request: RunCodeRequest):
    async with _execution_slots:
        return await _execute_code_unbounded(request)


_execution_epoch = uuid.uuid4().hex
_executions = {}
_execution_ttl = 180.0
_execution_limit = 256


def _expire_executions():
    now = time.monotonic()
    for key, entry in list(_executions.items()):
        if entry[1].done() and now - entry[2] > _execution_ttl:
            del _executions[key]


@sandbox_router.get("/run_code_session")
async def execution_session():
    return {"epoch": _execution_epoch}


@sandbox_router.delete("/run_code_result/{request_id}")
async def acknowledge_execution(request_id: str, x_execution_epoch: str = Header(...)):
    if x_execution_epoch != _execution_epoch:
        raise HTTPException(409, "Sandbox restarted; execution outcome is uncertain")
    entry = _executions.get(request_id)
    if entry is not None and entry[1].done():
        del _executions[request_id]
    return {"acknowledged": True}


@sandbox_router.post("/run_code", response_model=RunCodeResponse, tags=['sandbox'])
async def run_code(request: RunCodeRequest, x_execution_epoch: Optional[str] = Header(None),
                   x_execution_id: Optional[str] = Header(None),
                   x_execution_retry: Optional[str] = Header(None)):
    # Existing evaluator calls retain their original interface.
    if x_execution_id is None and x_execution_epoch is None:
        return await _execute_code(request)
    if not x_execution_id or x_execution_epoch != _execution_epoch:
        raise HTTPException(409, "Sandbox session mismatch; refusing uncertain re-execution")
    _expire_executions()
    entry = _executions.get(x_execution_id)
    payload = request.model_dump()
    if entry is None:
        if x_execution_retry == "1":
            raise HTTPException(410, "Execution record unavailable; refusing re-execution")
        if len(_executions) >= _execution_limit:
            raise HTTPException(503, "Sandbox execution queue is full")
        task = asyncio.create_task(_execute_code(request))
        entry = (payload, task, time.monotonic())
        _executions[x_execution_id] = entry
    elif entry[0] != payload:
        raise HTTPException(409, "Execution ID reused with different input")
    # A dropped HTTP connection must not cancel the underlying execution.
    return await asyncio.shield(entry[1])


@sandbox_router.post("/run_jupyter", name='Run Code in Jupyter', response_model=RunJupyterResponse, tags=['sandbox'])
async def run_jupyter_handler(request: RunJupyterRequest):
    resp = RunJupyterResponse(status=RunStatus.Success, message='', executor_pod_name=os.environ.get('MY_POD_NAME'))
    code_repr = "\n".join(request.cells)[:100]
    try:
        logger.debug(
            f'start processing jupyter request with code ```\n{code_repr}\n``` and files {list(request.files.keys())}...'
        )
        result = await run_jupyter(request)
        resp.driver = result.driver
        if result.status != CommandRunStatus.Finished:
            resp.status = RunStatus.Failed
        else:
            resp.status = RunStatus.Success
            resp.cells = result.cells
            resp.files = result.files
    except Exception as e:
        message = f'exception on running jupyter {code_repr}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError

    return resp
