"""Task rewards shared by training and evaluation."""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import re
import time
import threading
from typing import Any

import requests
from verl.utils.reward_score import math_dapo


_CODE_BLOCK = re.compile(
    r"^[ \t]*```([^\r\n`]*)\r?\n(.*?)^[ \t]*```[ \t]*\r?$",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)
logger = logging.getLogger(__name__)
_SANDBOX_SESSIONS = threading.local()


def get_sandbox_session() -> requests.Session:
    """Keep each connection pool thread-local because reward calls run in separate executor threads."""
    session = getattr(_SANDBOX_SESSIONS, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        _SANDBOX_SESSIONS.session = session
    return session


def reset_sandbox_session() -> None:
    """Discard the failed thread-local pool so the next attempt does not reuse a broken connection."""
    session = getattr(_SANDBOX_SESSIONS, "session", None)
    _SANDBOX_SESSIONS.session = None
    if session is not None:
        session.close()


def normalize_reward_prediction(value: Any) -> str:
    if value is None:
        return ""
    return math_dapo.normalize_final_answer(str(value))


def extract_code_answer(solution: str) -> str | None:
    # Prefer the last explicitly labelled Python submission, not a later
    # unlabelled example-output block. Do not select by syntax or test success:
    # an invalid final submission must not fall back to an earlier draft.
    blocks = _CODE_BLOCK.findall(solution)
    python_blocks = [
        code for language, code in blocks
        if language.strip().lower() in {"python", "python3", "py"}
    ]
    candidates = python_blocks or [
        code for language, code in blocks if not language.strip()
    ]
    return candidates[-1].strip() if candidates else None


def call_sandbox_once(
    program: str, sandbox_url: str
) -> tuple[dict[str, Any] | None, str]:
    payload: dict[str, Any] = {
        "compile_timeout": 15,
        "run_timeout": 30,
        "code": program,
        "stdin": None,
        "memory_limit_MB": 1024,
        "language": "python",
        "files": {},
        "fetch_files": [],
    }
    max_attempts = 3
    retry_statuses = {408, 429, 500, 502, 503, 504}
    for attempt in range(max_attempts):
        try:
            result = get_sandbox_session().post(
                sandbox_url,
                json=payload,
                timeout=(10, 180),
            )
            result.raise_for_status()
            response = result.json()
            if not isinstance(response, dict):
                raise TypeError("sandbox response must be an object")
            if response.get("status") == "SandboxError":
                raise TypeError(
                    "sandbox execution service returned SandboxError: "
                    f"{response.get('message', '')}"
                )
            if not isinstance(response.get("run_result"), dict):
                raise TypeError("sandbox run_result must be an object")
            run = response["run_result"]
            if response.get("status") not in {"Success", "Failed"} or run.get("status") not in {"Finished", "TimeLimitExceeded"}:
                return None, f"sandbox execution infrastructure error: {response.get('status')}/{run.get('status')}: {run.get('stderr', '')}"
            return response, ""
        except (
            requests.RequestException,
            requests.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            reset_sandbox_session()
            status = getattr(getattr(error, "response", None), "status_code", None)
            retryable = status is None or int(status) in retry_statuses
            if not retryable or attempt + 1 >= max_attempts:
                logger.error(
                    "sandbox reward evaluation failed after %d attempt(s): %s",
                    attempt + 1,
                    error,
                )
                return None, f"{type(error).__name__}: {error}"
            time.sleep(2**attempt)
    raise AssertionError("unreachable sandbox retry state")


def compute_sandbox_test_results(
    program: str, count: int, sandbox_url: str
) -> tuple[list[bool], str, str]:
    response, sandbox_error = call_sandbox_once(program, sandbox_url)
    if response is None:
        return [False] * count, "evaluation_error", sandbox_error
    run = response["run_result"]
    if response.get("status") != "Success" or run.get("return_code") != 0:
        return [False] * count, "evaluated", ""
    final_line = next((line.strip() for line in reversed(str(run.get("stdout", "")).splitlines()) if line.strip()), None)
    if final_line is None:
        return [False] * count, "evaluation_error", "sandbox returned no result line"
    try:
        results = json.loads(final_line)
    except (json.JSONDecodeError, TypeError) as error:
        return [False] * count, "evaluation_error", str(error)
    if not isinstance(results, list) or len(results) != count or not all(type(result) is bool for result in results):
        return [False] * count, "evaluation_error", "sandbox result shape does not match test cases"
    return results, "evaluated", ""


def _leetcode_results(student_source, check_source, entry_point, count):
    import json

    namespace = {}
    try:
        exec(compile(student_source, "<student>", "exec"), namespace)
        exec(compile(check_source, "<tests>", "exec"), namespace)
        results = eval("check(" + entry_point + ")", namespace)
    except BaseException:
        results = [False] * count
    print(json.dumps(results))


def compute_leetcode_test_results(
    code: str, tests: dict[str, Any], sandbox_url: str
) -> tuple[list[bool], str, str]:
    tree = ast.parse(tests["test_code"])
    check = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check"
    )
    assertions = [node for node in check.body if isinstance(node, ast.Assert)]
    result_name = "_test_outcomes"
    body: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id=result_name, ctx=ast.Store())],
            value=ast.List(elts=[], ctx=ast.Load()),
        )
    ]
    for assertion in assertions:
        body.append(
            ast.Try(
                body=[assertion],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="BaseException", ctx=ast.Load()),
                        name=None,
                        body=[
                            ast.Expr(
                                value=ast.Call(
                                    func=ast.Attribute(
                                        value=ast.Name(id=result_name, ctx=ast.Load()),
                                        attr="append",
                                        ctx=ast.Load(),
                                    ),
                                    args=[ast.Constant(False)],
                                    keywords=[],
                                )
                            )
                        ],
                    )
                ],
                orelse=[
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=result_name, ctx=ast.Load()),
                                attr="append",
                                ctx=ast.Load(),
                            ),
                            args=[ast.Constant(True)],
                            keywords=[],
                        )
                    )
                ],
                finalbody=[],
            )
        )
    body.append(ast.Return(value=ast.Name(id=result_name, ctx=ast.Load())))
    check.body = body
    harness = ast.unparse(ast.fix_missing_locations(tree))
    student_source = tests["import_prefix"] + "\n" + code
    program = _LEETCODE_SOURCE + "\n_leetcode_results(" + ", ".join(map(repr, (student_source, harness, tests["entry_point"], len(assertions)))) + ")\n"
    return compute_sandbox_test_results(program, len(assertions), sandbox_url)


def _standard_code_results(code, tests):
    import io
    import json
    import sys

    _prelude = 'from bisect import *\nfrom collections import *\nfrom functools import *\nfrom heapq import *\nfrom itertools import *\nfrom math import *\nfrom statistics import *\nfrom typing import *\n'
    _source = _prelude + code
    _tests = tests
    _results = []

    if "fn_name" in _tests:
        try:
            _compiled = compile(_source, "<student>", "exec")
        except BaseException:
            _compiled = None
        for _raw, _expected in zip(_tests["inputs"], _tests["outputs"]):
            try:


                _namespace = {}
                exec(_compiled, _namespace)
                if _tests["fn_name"] in _namespace:
                    _candidate = _namespace[_tests["fn_name"]]
                else:
                    _candidate = getattr(_namespace["Solution"](), _tests["fn_name"])
                _args = [json.loads(_line) for _line in _raw.split("\n")]
                _actual = _candidate(*_args)
                _actual = (
                    json.dumps(_actual)
                    if isinstance(_actual, (dict, list, tuple, bool)) or _actual is None
                    else str(_actual)
                )
                _results.append(_actual.strip() == str(_expected).strip())
            except BaseException:
                _results.append(False)
    else:
        for _raw, _expected in zip(_tests["inputs"], _tests["outputs"]):
            _namespace = {"__name__": "__main__"}
            _stdin, _stdout = sys.stdin, sys.stdout
            _captured = io.StringIO()
            _execution_failed = False
            try:
                sys.stdin = io.StringIO(str(_raw))
                sys.stdout = _captured
                exec(compile(_source, "<student>", "exec"), _namespace)
            except SystemExit as _error:



                _execution_failed = _error.code not in (None, 0)
            except BaseException:
                _execution_failed = True
            finally:
                sys.stdin, sys.stdout = _stdin, _stdout
            _results.append(
                not _execution_failed
                and _captured.getvalue().strip() == str(_expected).strip()
            )

    print(json.dumps(_results))


def compute_standard_code_test_results(
    code: str, tests: dict[str, Any], sandbox_url: str
) -> tuple[list[bool], str, str]:
    program = _STANDARD_SOURCE + "\n_standard_code_results(" + repr(code) + ", " + repr(tests) + ")\n"
    return compute_sandbox_test_results(program, len(tests["inputs"]), sandbox_url)


_LEETCODE_SOURCE = inspect.getsource(_leetcode_results)
_STANDARD_SOURCE = inspect.getsource(_standard_code_results)


def compute_code_test_results(
    solution: str, ground_truth: str
) -> tuple[str, list[bool], str, str]:
    tests = json.loads(ground_truth)
    code = extract_code_answer(solution)
    if "test_code" in tests:
        tree = ast.parse(tests["test_code"])
        check = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "check"
        )
        case_count = sum(isinstance(node, ast.Assert) for node in check.body)
    else:
        case_count = len(tests["inputs"])
    if code is None:
        return "", [False] * case_count, "missing_code", ""

    sandbox_url = os.environ["OQ_SANDBOX_URL"]
    if "test_code" in tests:
        results, status, evaluation_error = compute_leetcode_test_results(
            code, tests, sandbox_url
        )
    else:
        results, status, evaluation_error = compute_standard_code_test_results(
            code, tests, sandbox_url
        )
    return code, results, status, evaluation_error


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, Any]:
    """Keep reward fields consistent for aggregation, while marking evaluator failures separately from wrong answers."""
    extra_info = dict(extra_info or {})
    
    if "final_answer_text" in extra_info:
        solution_str = str(extra_info["final_answer_text"])
    if "code" not in data_source.lower():
        try:
            result: Any = math_dapo.compute_score(
                solution_str, ground_truth, strict_box_verify=True
            )
            prediction = normalize_reward_prediction(result["pred"])
            return build_reward_score_record(
                data_source=data_source,
                score=result["score"],
                acc=result["acc"],
                pred=prediction,
                trajectory_status="evaluated",
                extra_info=extra_info,
            )
        except Exception:
            logger.exception(
                "math reward evaluation failed for data_source=%s", data_source
            )
            return build_reward_evaluation_error(data_source, extra_info)

    try:
        code, results, evaluation_status, evaluation_error = compute_code_test_results(
            solution_str, ground_truth
        )
    except Exception as error:
        logger.exception(
            "code reward evaluation failed for data_source=%s", data_source
        )
        return build_reward_evaluation_error(
            data_source,
            extra_info,
            detail=f"{type(error).__name__}: {error}",
        )
    if not results:
        return build_reward_evaluation_error(
            data_source,
            extra_info,
            detail="code evaluator returned no test outcomes",
        )
    if evaluation_error:
        extra_info["reward_evaluation_error_detail"] = evaluation_error
    passed = sum(results)
    score = passed / len(results)
    return build_reward_score_record(
        data_source=data_source,
        score=score,
        acc=passed == len(results),
        pred=code,
        trajectory_status=evaluation_status,
        extra_info=extra_info,
    )


def build_reward_score_record(
    *,
    data_source: str,
    score: float,
    acc: bool,
    pred: str,
    trajectory_status: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra_info = dict(extra_info or {})
    
    
    trajectory_return = float(score) if "code" in data_source.lower() else float(acc)
    if not 0.0 <= trajectory_return <= 1.0:
        raise ValueError("normalized trajectory return must be in [0, 1]")
    source_index = extra_info.get("source_index", -1)
    try:
        source_index = int(source_index)
    except (TypeError, ValueError):
        source_index = -1
    termination_reason = str(extra_info.get("termination_reason", "unknown"))
    tool_parse_failure = str(extra_info.get("tool_parse_failure", ""))
    raw_tool_errors = extra_info.get("tool_errors", ())
    reward_evaluation_error = str(
        extra_info.get("reward_evaluation_error_detail", "")
    ).strip()
    if reward_evaluation_error:
        infrastructure_error = reward_evaluation_error
    elif isinstance(raw_tool_errors, (list, tuple)) and raw_tool_errors:
        infrastructure_error = str(raw_tool_errors[-1])
    elif isinstance(raw_tool_errors, str):
        infrastructure_error = raw_tool_errors
    else:
        infrastructure_error = ""
    response_token_count = extra_info.get("response_token_count", -1)
    try:
        response_token_count = int(response_token_count)
    except (TypeError, ValueError):
        response_token_count = -1
    return {
        "score": float(score),
        "acc": bool(acc),
        "trajectory_return": trajectory_return,
        "pred": str(pred),
        "data_source": str(data_source),
        "trajectory_status": str(trajectory_status),
        "reward_evaluation_error": trajectory_status == "evaluation_error",
        
        
        "benchmark": str(extra_info.get("benchmark", data_source)),
        "source_index": source_index,
        "sample_id": str(extra_info.get("sample_id", "")),
        "termination_reason": termination_reason,
        "tool_parse_failure": tool_parse_failure,
        "task_score": float(score),
        "answer_parse_status": "parsed" if str(pred).strip() else "missing",
        "assistant_turns": len(extra_info.get("turn_token_counts", [])),
        "tool_calls_used": int(extra_info.get("tool_calls_used", 0)),
        "generated_tokens": int(extra_info.get("generated_tokens", 0)),
        "wait_count": int(extra_info.get("wait_count", 0)),
        "turn_length_cutoffs": int(extra_info.get("turn_length_cutoffs", 0)),
        "protocol_failure": bool(
            extra_info.get("protocol_errors") or tool_parse_failure
            or trajectory_status.startswith("parse_failure")
            or trajectory_status
            in {"argument_failure", "repeated_failed_tool_call", "tool_timeout"}
        ),
        
        
        "infrastructure_failure": bool(
            trajectory_status in {"service_failure", "evaluation_error"}
        ),
        "infrastructure_error": infrastructure_error,
        "response_token_count": response_token_count,
    }


def build_reward_evaluation_error(
    data_source: str,
    extra_info: dict[str, Any] | None = None,
    *,
    detail: str = "",
) -> dict[str, Any]:
    extra_info = dict(extra_info or {})
    if detail:
        extra_info["reward_evaluation_error_detail"] = detail
    return build_reward_score_record(
        data_source=data_source,
        score=0.0,
        acc=False,
        pred="",
        trajectory_status="evaluation_error",
        extra_info=extra_info,
    )
