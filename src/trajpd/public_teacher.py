"""Construct and score the exact public Teacher prefix for one Student trajectory."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def validate_tokenizer_and_canonical_prompt_compatibility(
    student_path: str,
    teacher_path: str,
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
    tool_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> PreTrainedTokenizerBase:
    student = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    teacher = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
    if student.get_vocab() != teacher.get_vocab():
        raise ValueError("Student and Teacher vocabularies differ")
    if student.special_tokens_map != teacher.special_tokens_map:
        raise ValueError("Student and Teacher special tokens differ")
    for token, expected_id in {"<|im_end|>": 151645, "<|endoftext|>": 151643}.items():
        if student.convert_tokens_to_ids(token) != expected_id:
            raise ValueError(f"TrajPD Qwen protocol requires {token}={expected_id}")
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if getattr(student, attribute) != getattr(teacher, attribute):
            raise ValueError(f"Student and Teacher {attribute} differ")
    if json.loads(student.backend_tokenizer.to_str()) != json.loads(
        teacher.backend_tokenizer.to_str()
    ):
        raise ValueError("Student and Teacher tokenizer graphs differ")
    probe_messages = [
        {"role": "system", "content": "Follow the tool protocol."},
        {"role": "user", "content": "Return one verified result."},
    ]
    probe_tools = list(
        tool_schemas
        or [
            {
                "type": "function",
                "function": {
                    "name": "python",
                    "description": "Execute Python code.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    },
                },
            }
        ]
    )
    template_kwargs = dict(apply_chat_template_kwargs or {})
    rendered = student.apply_chat_template(
        probe_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=probe_tools,
        **template_kwargs,
    )
    student_ids = student.encode(rendered, add_special_tokens=False)
    teacher_ids = teacher.encode(rendered, add_special_tokens=False)
    direct_student_ids = encode_chat_prefix(
        student, probe_messages, template_kwargs, probe_tools
    )
    if student_ids != direct_student_ids or student_ids != teacher_ids:
        raise ValueError(
            "Student and Teacher tokenize the canonical Student chat/tool rendering "
            "differently"
        )
    return student


def encode_chat_prefix(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, Any]],
    apply_chat_template_kwargs: Mapping[str, Any] | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        tools=tools,
        **dict(apply_chat_template_kwargs or {}),
    )
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    return [int(token_id) for token_id in token_ids]


def extract_response_teacher_topk_logprobs(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    prompt_length: int,
    response_length: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The adapter row i predicts token i+1, so response scoring begins at prompt_length-1."""
    start = prompt_length - 1
    stop = start + response_length
    
    
    topk_ids = teacher_ids[start:stop, :].to(torch.long).cpu()
    topk_logprobs = teacher_logprobs[start:stop, :].to(torch.float32).detach().cpu()
    if topk_ids.ndim != 2 or topk_ids.shape != topk_logprobs.shape:
        raise ValueError("Teacher top-k ids and log-probabilities do not align")
    if topk_ids.shape[1] != topk:
        raise ValueError("Teacher probability width must match the configured top-k")
    if topk_ids.shape[0] != response_length:
        raise ValueError("Teacher top-k response slice has the wrong length")
    if not torch.isfinite(topk_logprobs).all():
        raise ValueError("Teacher top-k response scores contain NaN or Inf")
    return topk_ids, topk_logprobs
