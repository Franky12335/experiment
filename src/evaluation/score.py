"""Evaluation graders: Strict-box for GPQA/math and standard code judging for LCB."""
import os
import threading
from src.trajpd.score import compute_score as math_score, build_reward_score_record, build_reward_evaluation_error
from src.evaluation import code_score
from src.evaluation.sandbox_judge import check_correctness as native_check
_state = threading.local()
_request_limit = threading.Semaphore(8)
def check_correctness(**kwargs):
    kwargs["sandbox_fusion_url"] = os.environ["OQ_SANDBOX_URL"]
    kwargs["concurrent_semaphore"] = _request_limit
    result = native_check(**kwargs)
    for meta in result[1]:
        if meta.get("api_request_error") or meta.get("status") in {
            "api_error","sandbox_error","unknown_failure","unknown_failure_state","unexpected_success_state","unknown_api_state","internal_error"
        } or str(meta.get("status","")).startswith("unknown_api_status"):
            _state.errors.append(str(meta.get("api_request_error") or meta.get("status")))
    return result
code_score.check_correctness = check_correctness
def compute_score(data_source, solution_str, ground_truth, extra_info):
    extra_info = dict(extra_info or {})
    if "code" not in data_source.lower():
        # Explicit binary acc from the strict strict-box math grader.
        return math_score(data_source, solution_str, ground_truth, extra_info)
    import json
    tests = json.loads(ground_truth) if isinstance(ground_truth,str) else ground_truth
    if not isinstance(tests,dict) or not {"inputs","outputs"}.issubset(tests):
        raise ValueError("This aligned LCB dataset requires standard input/output tests")
    if not tests["inputs"] or len(tests["inputs"]) != len(tests["outputs"]):
        raise ValueError("LCB test inputs/outputs must be nonempty and paired")
    _state.errors=[]
    text = str(extra_info.get("final_answer_text",solution_str))
    result=code_score.compute_score(text,json.dumps(tests),timeout=30)
    if _state.errors:
        return build_reward_evaluation_error(data_source,extra_info,detail="; ".join(_state.errors))
    record=build_reward_score_record(data_source=data_source,score=max(0.0,float(result["score"])),
        acc=bool(result["acc"]),pred=result["pred"] or "",trajectory_status="evaluated",extra_info=extra_info)
    record["grader"]="standard_io"
    record["raw_score"]=float(result["score"])
    return record
