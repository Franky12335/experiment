"""Prepare training and held-out evaluation datasets."""

from __future__ import annotations

import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow
from pyarrow import parquet
from tqdm.auto import tqdm

from src.data_prep.canonical_prompt import build_canonical_prompt
from src.seed import configured_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_SFT = PROJECT_ROOT / "data/sft-3k.parquet"
CLEAN_SFT = PROJECT_ROOT / "data/sft-clean.parquet"
RAW_GRPO = PROJECT_ROOT / "data/grpo-30k.parquet"
CANONICAL_INDICES = PROJECT_ROOT / "data/canonical_source_indices.txt"
EVAL_ROOT = PROJECT_ROOT / "data/eval"
OUTPUT_ROOT = PROJECT_ROOT / "data/prepared"
_AIME_PROMPT = (
    "Analyze and solve the following problem step by step. You may use the Python tool "
    "for calculation or verification. When the answer is final, stop calling tools and "
    "write it as \\boxed{answer}. Do not put units inside the box.\n\n"
)
_GPQA_PROMPT = (
    "Solve the multiple-choice science problem carefully. You may use the Python tool "
    "for calculation or verification. End with exactly one of \\boxed{A}, \\boxed{B}, "
    "\\boxed{C}, or \\boxed{D}.\n\n"
)


def select_validation_indices(
    rows: list[dict[str, Any]], count: int, seed: int
) -> set[int]:
    if not 0 < count < len(rows):
        raise ValueError(
            f"validation count must be within (0, {len(rows)}), got {count}"
        )
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return set(indices[:count])


def split_dataset_rows(
    rows: list[dict[str, Any]], validation_count: int, seed: int
) -> tuple[list, list]:
    val_indices = select_validation_indices(rows, validation_count, seed)
    train = [row for index, row in enumerate(rows) if index not in val_indices]
    val = [row for index, row in enumerate(rows) if index in val_indices]
    return train, val


def build_distillation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy deployment-visible task records without constructing hidden prompt fields."""
    prepared = []
    for index, row in enumerate(rows):
        source_index = row["extra_info"]["source_index"]
        sample_id = str(row["extra_info"]["sample_id"])
        prepared.append(
            {
                **row,
                "extra_info": {
                    "index": index,
                    "source_index": source_index,
                    "sample_id": sample_id,
                },
            }
        )
    return prepared


def split_teacher_training_rows(
    prepared: list[dict[str, Any]], val_count: int = 512
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split Teacher training data before copying the public distillation partition."""
    train, validation = split_dataset_rows(
        prepared, validation_count=val_count, seed=configured_seed() + 1
    )
    return train, validation, build_distillation_rows(train)


def write_parquet_rows(rows: Iterable[dict[str, Any]], path: Path) -> dict[str, Any]:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet.write_table(
        pyarrow.Table.from_pylist(materialized), path, compression="zstd"
    )
    return {"path": str(path.relative_to(PROJECT_ROOT)), "rows": len(materialized)}


def join_message_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def normalize_task_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def prepare_evaluation_datasets() -> tuple[dict[str, dict[str, Any]], set[str]]:
    outputs: dict[str, dict[str, Any]] = {}
    eval_texts: set[str] = set()
    destination = OUTPUT_ROOT / "eval"

    specs = (
        ("aime2024", "aime2024.parquet"),
        ("aime2025", "aime2025.parquet"),
        ("gpqa_diamond", "gpqa_diamond.parquet"),
        ("livecodebench_v6", "livecodebench_v6.parquet"),
    )
    for benchmark, filename in tqdm(
        specs, desc="Evaluation datasets", unit="dataset", dynamic_ncols=True
    ):
        source = EVAL_ROOT / filename
        prepared = []
        source_rows = parquet.read_table(source).to_pylist()
        for source_index, row in enumerate(
            tqdm(
                source_rows,
                desc=f"Prepare {benchmark}",
                unit="row",
                leave=False,
                dynamic_ncols=True,
            )
        ):
            if benchmark == "aime2024":
                problem, answer = row["Problem"], str(row["Answer"])
                prompt = [{"role": "user", "content": _AIME_PROMPT + problem}]
                ability = "math"
            elif benchmark == "aime2025":
                problem, answer = row["problem"], str(row["answer"])
                prompt = [{"role": "user", "content": _AIME_PROMPT + problem}]
                ability = "math"
            elif benchmark == "gpqa_diamond":
                problem, answer = row["problem"], str(row["solution"])
                from src.data_prep.evaluation_prompts import gpqa_prompt
                prompt = gpqa_prompt(row)
                ability = "science"
            else:
                problem = row["messages"][-1]["content"]
                answer = str(row["reward_model"]["ground_truth"])
                from src.data_prep.evaluation_prompts import lcb_prompt
                prompt = lcb_prompt(row)
                ability = "code"
            sample_id = f"{benchmark}-{source_index:05d}"
            eval_texts.add(normalize_task_text(problem))
            prepared.append(
                {
                    "data_source": benchmark,
                    "prompt": prompt,
                    "ability": ability,
                    "reward_model": {"ground_truth": answer, "style": "rule"},
                    "extra_info": {
                        "benchmark": benchmark,
                        "sample_id": sample_id,
                        "source_index": source_index,
                    },
                    "agent_name": "tool_agent",
                }
            )
        outputs[benchmark] = write_parquet_rows(prepared, destination / filename)
        outputs[benchmark]["source_path"] = str(source.relative_to(PROJECT_ROOT))
    return outputs, eval_texts


def prepare_sft_datasets() -> dict[str, dict[str, Any]]:
    stable_clean = OUTPUT_ROOT / "teacher_sft_clean.parquet"
    if CLEAN_SFT.is_file():
        rows = parquet.read_table(CLEAN_SFT).to_pylist()
    else:
        rows = clean_sft_rows(parquet.read_table(RAW_SFT).to_pylist())
    rows.sort(
        key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    )
    clean_record = write_parquet_rows(rows, stable_clean)
    # The VeRL SFT validator drops incomplete global batches. Keep this
    # holdout at exactly one global batch so every declared validation row is used.
    train, validation = split_dataset_rows(rows, validation_count=128, seed=configured_seed())
    return {
        "clean": clean_record,
        "train": write_parquet_rows(train, OUTPUT_ROOT / "teacher_sft_train.parquet"),
        "validation": write_parquet_rows(
            validation, OUTPUT_ROOT / "teacher_sft_val.parquet"
        ),
    }


def clean_message_text(value: str) -> str:
    value = value.replace("de\x0cne", "define")
    return "".join(
        character
        for character in value
        if unicodedata.category(character) != "Cc" or character in "\n\r\t"
    )


def clean_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repair the message/tool pairing required by multi-turn SFT."""
    cleaned = []
    for row in tqdm(rows, desc="Clean Teacher SFT", unit="row", dynamic_ncols=True):
        messages: list[dict[str, Any]] = []
        for raw_message in row["messages"]:
            message = dict(raw_message)
            message["content"] = clean_message_text(str(message.get("content") or ""))
            if message["role"] == "tool":
                if not messages or messages[-1]["role"] != "assistant":
                    continue
                if "Failed to get code" in message["content"]:
                    continue
            messages.append(message)
        if (
            len(messages) >= 2
            and messages[0]["role"] == "user"
            and messages[-1]["role"] == "assistant"
            and all(has_message_payload(message) for message in messages)
        ):
            cleaned.append({"messages": messages, "tools": row["tools"]})
    return cleaned


def has_message_payload(message: dict[str, Any]) -> bool:
    """Accept tool-call-only assistant turns without accepting empty messages."""
    if str(message.get("content") or "").strip():
        return True
    tool_calls = message.get("tool_calls")
    return (
        message.get("role") == "assistant"
        and isinstance(tool_calls, list)
        and bool(tool_calls)
    )


def derive_canonical_source_indices(
    source_rows: list[dict[str, Any]],
) -> list[int]:
    """Recover the canonical selection or rebuild it by the same prompt rules."""
    if CANONICAL_INDICES.is_file():
        return [int(value) for value in CANONICAL_INDICES.read_text().splitlines()]
    groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for source_index, row in enumerate(
        tqdm(source_rows, desc="Group GRPO prompts", unit="row", dynamic_ncols=True)
    ):
        is_code = "code" in row["data_source"].lower()
        prompt = build_canonical_prompt(row["prompt"], row["data_source"], is_code)
        text = unicodedata.normalize("NFKC", join_message_text(prompt))
        normalized_text = re.sub(r"\s+", " ", text).strip()
        ground_truth = json.dumps(row["reward_model"]["ground_truth"], sort_keys=True)
        groups[normalized_text].append((source_index, ground_truth))
    selected = []
    for occurrences in tqdm(
        groups.values(), desc="Select GRPO prompts", unit="group", dynamic_ncols=True
    ):
        if len({ground_truth for _, ground_truth in occurrences}) == 1:
            selected.append(occurrences[0][0])
    return sorted(selected)


def prepare_grpo_datasets(
    eval_texts: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    source_rows = parquet.read_table(RAW_GRPO).to_pylist()
    selected = derive_canonical_source_indices(source_rows)
    prepared = []
    removed_test_partition = 0
    removed_exact_eval_match = 0
    for source_index in tqdm(
        selected, desc="Prepare Teacher GRPO", unit="row", dynamic_ncols=True
    ):
        row = source_rows[source_index]
        if row["data_source"].startswith("test-code-leetcode-"):
            removed_test_partition += 1
            continue
        problem_text = join_message_text(row["prompt"])
        if normalize_task_text(problem_text) in eval_texts:
            removed_exact_eval_match += 1
            continue
        is_code = "code" in row["data_source"].lower()
        prepared.append(
            {
                "data_source": row["data_source"],
                "prompt": build_canonical_prompt(
                    row["prompt"], row["data_source"], is_code
                ),
                "ability": row["ability"],
                "reward_model": row["reward_model"],
                "extra_info": {
                    "source_index": source_index,
                    "sample_id": f"teacher-grpo-{source_index:05d}",
                },
                "agent_name": "tool_agent",
            }
        )
    train, validation, distillation_rows = split_teacher_training_rows(prepared)
    distillation_output = write_parquet_rows(
        tqdm(
            distillation_rows,
            desc="Prepare distillation data",
            unit="row",
            dynamic_ncols=True,
        ),
        OUTPUT_ROOT / "decision_distillation_train.parquet",
    )
    outputs = {
        "train": write_parquet_rows(train, OUTPUT_ROOT / "teacher_grpo_train.parquet"),
        "validation": write_parquet_rows(
            validation, OUTPUT_ROOT / "teacher_grpo_val.parquet"
        ),
    }
    exclusions = {
        "not_in_canonical_selection": len(source_rows) - len(selected),
        "held_out_code_partition": removed_test_partition,
        "exact_eval_text_match": removed_exact_eval_match,
    }
    outputs["distillation"] = distillation_output
    return outputs, exclusions


def prepare_experiment_datasets() -> dict[str, Any]:
    """Materialize the datasets required by the three research stages."""
    eval_outputs, eval_texts = prepare_evaluation_datasets()
    sft_outputs = prepare_sft_datasets()
    grpo_outputs, grpo_exclusions = prepare_grpo_datasets(eval_texts)
    return {
        "teacher_sft": sft_outputs,
        "teacher_grpo": grpo_outputs,
        "teacher_grpo_exclusions": grpo_exclusions,
        "evaluation": eval_outputs,
    }


def main() -> None:
    print(json.dumps(prepare_experiment_datasets(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
