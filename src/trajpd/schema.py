"""TrajPD tensor fields and decision boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DECISION_PREFIX_SNAPSHOTS = "decision_prefix_snapshots"
TRAJECTORY_RETURN = "trajectory_return"
OUTCOME_ADVANTAGE = "outcome_advantage"
REWARD_EXTRA_INFO = "reward_extra_info"
TEACHER_TOPK_TOKEN_IDS = "teacher_topk_token_ids"
TEACHER_TOPK_LOG_PROBS = "teacher_topk_log_probs"
DECISION_INDEX = "decision_index"


@dataclass(frozen=True, slots=True)
class DecisionPrefixSnapshot:
    """Store boundaries, avoiding quadratic storage of growing prefixes."""

    decision_index: int
    initial_prompt_length: int
    response_start: int
    response_stop: int

    @classmethod
    def create_from_mapping(cls, value: dict[str, Any]) -> "DecisionPrefixSnapshot":
        snapshot = cls(
            decision_index=int(value["decision_index"]),
            initial_prompt_length=int(value["initial_prompt_length"]),
            response_start=int(value["response_start"]),
            response_stop=int(value["response_stop"]),
        )
        snapshot.validate()
        return snapshot

    def validate(self) -> None:
        if self.decision_index < 0:
            raise ValueError("decision_index must be nonnegative")
        if self.initial_prompt_length < 1:
            raise ValueError("initial_prompt_length must be positive")
        if self.response_start < 0 or self.response_stop <= self.response_start:
            raise ValueError("decision response span must be nonempty and ordered")
