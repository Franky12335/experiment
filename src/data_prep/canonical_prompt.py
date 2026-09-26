"""Canonical deployment-visible task prompts."""

from __future__ import annotations

_CODE_INSTRUCTION = (
    "\n\n**Note: You should first analyze the problem and form a high-level solution "
    "strategy whose time and memory complexity satisfy the stated constraints, then "
    "utilize the tools to help you solve the problem.**\n"
    "Every code_interpreter call must put only complete, valid Python source in "
    "arguments.code. Never put natural-language explanation, pseudocode, Markdown "
    "fences, or an unfinished string in arguments.code. Use finite loops and do not "
    "repeat an unchanged failed tool call. Before submitting, use the tool to verify "
    "the implementation. Then stop calling tools and submit exactly one complete, "
    "executable Python program in one complete Markdown Python code block, including "
    "both its opening and closing delimiters."
)
_LEGACY_CODE_SUFFIX = (
    "\n\nDevelop and verify the algorithm with the Python tool. After verification, "
    "stop calling tools and submit the complete program in one final ```python code "
    "fence."
)
_ANSWER_INSTRUCTION = (
    "\nDo not put units of the final answer inside \\boxed{}. The content of \\boxed{} "
    "should be the final numerical value of the answer only, without any units."
)
_MATH_PREFIX = "Analyze and solve the following math problem step by step. \n\n"
_MATH_INSTRUCTION = (
    "\n\nThe tool could be used for more precise and efficient calculation and could help "
    "you to verify your result before you reach the final answer.\n\n"
    "**Note: You should first analyze the problem and form a high-level solution "
    "strategy, then utilize the tools to help you solve the problem.**\n"
    "Remember once you make sure the current answer is your final answer, do not call "
    "the tools again and directly output the final answer in the following text format, "
    "the answer format must be: \\boxed{'The final answer goes here.'}."
)


def build_canonical_prompt(
    prompt: list[dict], data_source: str, is_code: bool
) -> list[dict]:
    messages = [dict(message) for message in prompt]
    messages[-1]["content"] = messages[-1]["content"].replace("\x0c", "\\f")
    if is_code:
        if messages[-1]["content"].endswith(_LEGACY_CODE_SUFFIX):
            messages[-1]["content"] = messages[-1]["content"][
                : -len(_LEGACY_CODE_SUFFIX)
            ]
        messages[-1]["content"] += _CODE_INSTRUCTION
    else:
        if data_source not in {"math_dapo", "mega-science"}:
            messages[-1]["content"] = (
                _MATH_PREFIX + messages[-1]["content"] + _MATH_INSTRUCTION
            )
        messages[-1]["content"] += _ANSWER_INSTRUCTION
    return messages
