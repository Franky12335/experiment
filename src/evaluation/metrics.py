"""The evaluation protocol metrics: mean over each actual N-sample pool, then across problems.

Pass@k uses binary-accuracy best@k/mean:
1000 with-replacement resamples per problem, using a reset NumPy seed per call.
It is not an indicator of any success in the original sample pool.
"""

from __future__ import annotations

import random
from collections import Counter
from statistics import mean
from typing import Any

import numpy
from tqdm.auto import tqdm
from src.evaluation.bootstrap import bootstrap_metric


def compute_majority_correct(rows: list[dict[str, Any]]) -> bool | None:
    predictions = [str(row.get("pred", "")) for row in rows if row.get("pred")]
    if not predictions:
        return None
    counts = Counter(predictions)
    top_count = max(counts.values())
    winners = [prediction for prediction, count in counts.items() if count == top_count]
    if len(winners) != 1:
        return False
    winner = winners[0]
    return any(bool(row["acc"]) for row in rows if str(row.get("pred", "")) == winner)


def compute_bootstrap_mean_confidence_interval(
    values: list[float], samples: int, seed: int, confidence: float = 0.95
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty list")
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in range(len(values)))
        for _ in tqdm(
            range(samples),
            desc="Problem bootstrap",
            unit="sample",
            leave=False,
            dynamic_ncols=True,
        )
    )
    tail = (1.0 - confidence) / 2.0
    low = estimates[int(tail * samples)]
    high = estimates[min(samples - 1, int((1.0 - tail) * samples))]
    return [low, high]


def compute_bootstrap_best(accuracies, k, seed=42):
    """The evaluation protocol best@k/mean and best@k/std on binary acc (1000 resamples)."""
    if not accuracies or k < 1 or any(type(x) not in (bool, int, float) or x not in (0, 1) for x in accuracies):
        raise ValueError("Need nonempty binary outcomes and positive k")
    [(value, std)] = bootstrap_metric(
        data=accuracies, subset_size=k, reduce_fns=[numpy.max],
        n_bootstrap=1000, seed=seed,
    )
    return float(value), float(std)



def summarize_benchmark_metrics(
    benchmark, groups, expected_samples, best_ks,
    within_problem_bootstrap_samples, bootstrap_samples, seed,
):
    if expected_samples < 1 or not groups:
        raise ValueError("Nonempty problems and positive N required")
    if within_problem_bootstrap_samples != 1000:
        raise ValueError("Bootstrap Best requires exactly 1000 repetitions")
    if any(k < 1 or k > expected_samples for k in best_ks):
        raise ValueError("Pass k must be between 1 and actual N")
    problem_rows = []
    for source_index in sorted(groups):
        rows = list(groups[source_index])
        if len(rows) != expected_samples:
            raise ValueError(f"{benchmark}[{source_index}] has {len(rows)} samples, expected {expected_samples}")
        values = [row["acc"] for row in rows]
        if any(type(x) not in (bool, int, float) or x not in (0, 1) for x in values):
            raise ValueError("Accuracy must be binary")
        item = {"source_index": source_index, "samples": len(rows),
                "correct": sum(values), "mean": mean(values),
                f"acc/mean@{expected_samples}": mean(values)}
        for k in best_ks:
            value, std = compute_bootstrap_best(values, k, seed)
            item[f"pass@{k}"] = value
            item[f"pass@{k}/bootstrap_std"] = std
        problem_rows.append(item)
    means = [row["mean"] for row in problem_rows]
    result = {"benchmark": benchmark, "problems": len(problem_rows), "samples_per_problem": expected_samples,
              f"acc/mean@{expected_samples}": mean(means),
              f"acc/mean@{expected_samples}/problem_bootstrap_ci95":
                  compute_bootstrap_mean_confidence_interval(means, bootstrap_samples, seed),
              "per_problem": problem_rows}
    for k in best_ks:
        result[f"pass@{k}"] = mean(row[f"pass@{k}"] for row in problem_rows)
    return result
