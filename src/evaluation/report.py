"""Aggregate the method's held-out generation metrics."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from pyarrow import parquet
from tqdm.auto import tqdm
from omegaconf import OmegaConf

from src.evaluation.metrics import summarize_benchmark_metrics


def mean_or_none(values) -> float | None:
    materialized = list(values)
    return mean(materialized) if materialized else None


def format_percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}"


def read_generation_files(root: Path) -> list[dict[str, Any]]:
    
    
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL generation dump found in {root}")
    rows = []
    for path in tqdm(
        files, desc="Read generation files", unit="file", dynamic_ncols=True
    ):
        with path.open() as stream:
            for line_number, line in enumerate(
                tqdm(
                    stream,
                    desc=path.name,
                    unit="row",
                    leave=False,
                    dynamic_ncols=True,
                ),
                1,
            ):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid JSON in {path}:{line_number}"
                        ) from error
    return rows


def extract_rendered_user_content(rendered_prompt: Any) -> str | None:
    if not isinstance(rendered_prompt, str):
        return None
    marker = "\nuser\n"
    if marker not in rendered_prompt:
        return None
    content = rendered_prompt.rsplit(marker, 1)[1]
    suffix = "\nassistant\n"
    return content[: -len(suffix)] if content.endswith(suffix) else content


def build_evaluation_identity_catalog(
    data_files: list[str],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, set[int]]]:
    catalog: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    expected_indices: dict[str, set[int]] = defaultdict(set)
    for data_file in data_files:
        for item in parquet.read_table(
            data_file, columns=["data_source", "prompt", "extra_info"]
        ).to_pylist():
            benchmark = str(item["data_source"]).lower()
            prompt = item["prompt"]
            if not isinstance(prompt, list) or not prompt:
                raise ValueError(
                    f"prepared evaluation row in {data_file} has no prompt"
                )
            user_messages = [
                message.get("content")
                for message in prompt
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            if not user_messages or not isinstance(user_messages[-1], str):
                raise ValueError(
                    f"prepared evaluation row in {data_file} has no user content"
                )
            extra_info = dict(item.get("extra_info") or {})
            source_index = int(extra_info["source_index"])
            identity = {
                "source_index": source_index,
                "sample_id": str(extra_info.get("sample_id", "")),
            }
            content = user_messages[-1]
            if content in catalog[benchmark]:
                raise ValueError(
                    f"duplicate prepared prompt for {benchmark}[{source_index}]"
                )
            if source_index in expected_indices[benchmark]:
                raise ValueError(
                    f"duplicate source_index for {benchmark}[{source_index}]"
                )
            catalog[benchmark][content] = identity
            expected_indices[benchmark].add(source_index)
    return dict(catalog), dict(expected_indices)


def normalize_generation_row(
    row: dict[str, Any], identity_catalog: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    benchmark = row.get("benchmark") or row.get("data_source")
    if benchmark is None:
        raise ValueError("every generation must contain benchmark/data_source")
    benchmark = str(benchmark).lower()
    source_index = row.get("source_index")
    sample_id = row.get("sample_id")
    if source_index is None or int(source_index) < 0:
        content = extract_rendered_user_content(row.get("input"))
        identity = identity_catalog.get(benchmark, {}).get(content or "")
        if identity is None:
            raise ValueError(
                f"cannot recover source_index for a {benchmark} generation from prepared data"
            )
        source_index = identity["source_index"]
        sample_id = sample_id or identity["sample_id"]
    acc = row.get("acc", row.get("task_acc"))
    if type(acc) not in (bool, int, float) or acc not in (0, 1):
        raise ValueError("every generation must contain binary acc (0 or 1)")
    trajectory_status = str(row.get("trajectory_status", "unknown"))
    tool_parse_failure = str(row.get("tool_parse_failure", ""))
    reward_evaluation_error = bool(row.get("reward_evaluation_error", False))
    infrastructure_failure = bool(
        row.get(
            "infrastructure_failure",
            reward_evaluation_error
            or trajectory_status in {"service_failure", "evaluation_error"},
        )
    )
    response_token_count = row.get("response_token_count")
    if response_token_count is not None:
        response_token_count = int(response_token_count)
        if response_token_count < 0:
            response_token_count = None
    return {
        "benchmark": benchmark,
        "source_index": int(source_index),
        "sample_id": sample_id,
        "uid": row.get("uid"),
        "acc": bool(acc),
        "score": float(row.get("score", row.get("reward", float(acc)))),
        "pred": str(row.get("pred", "")),
        "trajectory_status": trajectory_status,
        "termination_reason": str(row.get("termination_reason", "unknown")),
        "tool_parse_failure": tool_parse_failure,
        "protocol_failure": bool(
            row.get(
                "protocol_failure",
                tool_parse_failure
                or trajectory_status.startswith("parse_failure")
                or trajectory_status == "argument_failure",
            )
        ),
        "infrastructure_failure": infrastructure_failure,
        "reward_evaluation_error": reward_evaluation_error,
        "response_token_count": response_token_count,
        "overlong": row.get("overlong"),
        "overlong_reward": row.get("overlong_reward"),
        "evaluation_method": str(row.get("evaluation_method", "")),
        "generation_protocol": row.get("generation_protocol"),
    }


def compute_evaluation_protocol_audit(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(rows)
    response_lengths = [
        int(row["response_token_count"])
        for row in rows
        if row["response_token_count"] is not None
    ]
    overlong_values = [
        bool(row["overlong"]) for row in rows if row["overlong"] is not None
    ]
    protocol_valid_rows = [row for row in rows if not row["protocol_failure"]]
    prediction_present = [bool(str(row["pred"]).strip()) for row in rows]
    parse_failures = Counter(
        row["tool_parse_failure"] for row in rows if row["tool_parse_failure"]
    )
    code_rows = [row for row in rows if "code" in row["benchmark"]]
    return {
        "samples": total,
        "trajectory_status_counts": dict(
            sorted(Counter(row["trajectory_status"] for row in rows).items())
        ),
        "termination_reason_counts": dict(
            sorted(Counter(row["termination_reason"] for row in rows).items())
        ),
        "tool_parse_failure_counts": dict(sorted(parse_failures.items())),
        "tool_parse_failure_rates": {
            name: count / total for name, count in sorted(parse_failures.items())
        },
        "protocol_failure_rate": sum(bool(row["protocol_failure"]) for row in rows)
        / total,
        "conditional_valid_accuracy": (
            mean(bool(row["acc"]) for row in protocol_valid_rows)
            if protocol_valid_rows
            else None
        ),
        "prediction_present_rate": mean(prediction_present),
        "code_submission_rate": (
            mean(bool(str(row["pred"]).strip()) for row in code_rows)
            if code_rows
            else None
        ),
        "infrastructure_failure_rate": sum(
            bool(row["infrastructure_failure"]) for row in rows
        )
        / total,
        "response_token_coverage": len(response_lengths) / total,
        "response_tokens": (
            {
                "mean": mean(response_lengths),
                "min": min(response_lengths),
                "max": max(response_lengths),
            }
            if response_lengths
            else None
        ),
        "overlong_coverage": len(overlong_values) / total,
        "overlong_rate": mean(overlong_values) if overlong_values else None,
    }


def aggregate_evaluation_metrics(
    method: str,
    generation_dir: Path,
    evaluation_config: Path,
    output: Path,
) -> dict[str, Any]:
    config = OmegaConf.load(evaluation_config)
    from src.evaluation.identity import validate_identity
    validate_identity(method, config.evaluation.get("method", method))
    protocol = OmegaConf.to_container(config.evaluation, resolve=True)
    expected = int(protocol["samples_per_problem"])
    sampling = config.actor_rollout_ref.rollout.val_kwargs
    expected_generation_protocol = {
        "temperature": float(sampling.temperature),
        "top_p": float(sampling.top_p),
        "top_k": int(sampling.top_k),
        "do_sample": bool(sampling.do_sample),
        "samples_per_problem": int(sampling.n),
    }
    if expected_generation_protocol["samples_per_problem"] != expected:
        raise ValueError("evaluation sample count and generation n must match")
    protocol["generation"] = expected_generation_protocol
    protocol["average_estimator"] = "all_actual_samples_per_problem"
    protocol["best_estimator"] = "bootstrap_with_replacement"
    protocol["metric_version"] = "avg_actualN_pass_bootstrap_replacement_v1"
    if int(protocol.get("within_problem_bootstrap_samples", 1000)) != 1000:
        raise ValueError("The evaluation protocol Pass@k requires exactly 1000 within-problem bootstrap resamples")
    protocol["within_problem_bootstrap_samples"] = 1000
    protocol["confidence_interval_unit"] = "problem"
    best_ks = [int(value) for value in protocol.get("pass_k", [8,16,32,64,96,128])]
    best_ks = [k for k in best_ks if k <= expected]
    for obsolete in ("avg_k", "best_k", "pass_estimator", "subset_avg_role", "accuracy_estimator"):
        protocol.pop(obsolete, None)
    protocol["pass_k"] = best_ks
    protocol["pass_estimator"] = "bootstrap_with_replacement_1000"
    protocol["correctness_input"] = "binary acc, not reward/score"

    protocol["primary_metric"] = f"acc/mean@{expected}"
    protocol["secondary_metrics"] = [f"pass@{k}" for k in best_ks]
    expected_benchmarks = [str(value).lower() for value in protocol["benchmarks"]]
    data_files = [
        str(value)
        for value in OmegaConf.to_container(config.data.val_files, resolve=True)
    ]
    identity_catalog, expected_indices = build_evaluation_identity_catalog(data_files)
    rows = [
        normalize_generation_row(row, identity_catalog)
        for row in tqdm(
            read_generation_files(generation_dir),
            desc="Check generation rows",
            unit="row",
            dynamic_ncols=True,
        )
    ]
    for row in rows:
        if row["evaluation_method"] != method:
            raise ValueError(
                "generation method label mismatch: "
                f"found {row['evaluation_method']!r}, expected {method!r}"
            )
        if row["generation_protocol"] != expected_generation_protocol:
            raise ValueError(
                "generation sampling protocol mismatch: "
                f"found {row['generation_protocol']!r}, "
                f"expected {expected_generation_protocol!r}"
            )
    groups: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_uids = set()
    for row in tqdm(rows, desc="Group evaluation rows", unit="row", dynamic_ncols=True):
        if row["uid"] is not None:
            key = str(row["uid"])
            if key in seen_uids:
                raise ValueError(f"duplicate rollout uid: {key}")
            seen_uids.add(key)
        groups[row["benchmark"]][row["source_index"]].append(row)
    if set(groups) != set(expected_benchmarks):
        raise ValueError(
            f"benchmark mismatch: found {sorted(groups)}, expected {sorted(expected_benchmarks)}"
        )
    for benchmark in expected_benchmarks:
        found_indices = set(groups[benchmark])
        if found_indices != expected_indices.get(benchmark, set()):
            raise ValueError(
                f"{benchmark} problem identity mismatch: found {sorted(found_indices)}, "
                f"expected {sorted(expected_indices.get(benchmark, set()))}"
            )
    infrastructure_failures = [row for row in rows if row["infrastructure_failure"]]
    if protocol.get("fail_on_infrastructure_error", True) and infrastructure_failures:
        counts = Counter(row["trajectory_status"] for row in infrastructure_failures)
        raise RuntimeError(
            "formal aggregation refused because infrastructure failures were observed: "
            f"{dict(sorted(counts.items()))}"
        )
    summaries = {}
    for benchmark in tqdm(
        expected_benchmarks,
        desc="Compute benchmark metrics",
        unit="benchmark",
        dynamic_ncols=True,
    ):
        summaries[benchmark] = summarize_benchmark_metrics(
            benchmark,
            groups[benchmark],
            expected,
            best_ks,
            int(protocol["within_problem_bootstrap_samples"]),
            int(protocol["bootstrap_samples"]),
            int(protocol["seed"]),
        )
        summaries[benchmark]["protocol_audit"] = compute_evaluation_protocol_audit(
            [row for problem_rows in groups[benchmark].values() for row in problem_rows]
        )
    primary_key = f"acc/mean@{expected}"
    result = {
        "method": method,
        "protocol": protocol,
        "primary_metric": primary_key,
        "macro_average": mean(summary[primary_key] for summary in summaries.values()),
        "benchmarks": summaries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(output.stem + "_at_k.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method","benchmark","N","avg_at_N","k","pass_at_k","bootstrap_repetitions"])
        writer.writeheader()
        for benchmark, summary in summaries.items():
            for k in best_ks or [None]:
                writer.writerow({"method":method,"benchmark":benchmark,"N":expected,"avg_at_N":summary[primary_key],
                    "k":k,"pass_at_k":summary[f"pass@{k}"] if k is not None else "","bootstrap_repetitions":1000})
    lines = ["# Evaluation results", "", f"Avg@{expected}: all {expected} actual samples per problem. Pass@k: The evaluation protocol best@k/mean on binary acc, using 1000 bootstrap resamples with replacement per problem and then averaging across problems.", "",
             f"| Dataset | N | Avg@{expected} (%) | k | Pass@k (%) |", "|:--|--:|--:|--:|--:|"]
    for benchmark, summary in summaries.items():
        if not best_ks:
            lines.append(f"| {benchmark} | {expected} | {format_percent(summary[primary_key])} | — | — |")
        for k in best_ks:
            lines.append(f"| {benchmark} | {expected} | {format_percent(summary[primary_key])} | {k} | {format_percent(summary[f'pass@{k}'])} |")
    output.with_suffix(".md").write_text("\n".join(lines)+"\n")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result

