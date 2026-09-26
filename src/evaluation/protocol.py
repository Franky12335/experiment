"""Validate the common evaluation resource limits before allocating GPUs."""
def validate_protocol(config):
    rollout = config.actor_rollout_ref.rollout
    n = int(config.evaluation.samples_per_problem)
    ks = [int(k) for k in config.evaluation.pass_k]
    if n < 1 or any(k < 1 or k > n for k in ks):
        raise ValueError("Pass k cannot exceed actual generated N")
    if int(config.evaluation.within_problem_bootstrap_samples) != 1000:
        raise ValueError("The evaluation protocol Pass@k requires exactly 1000 bootstrap resamples")
    budget = config.data.interaction_budget
    turns = int(rollout.multi_turn.max_assistant_turns)
    calls = int(budget.max_tool_calls)
    if turns < 2 or not 1 <= calls < turns:
        raise ValueError("Reserve an assistant turn for the final answer")
    if rollout.multi_turn.max_parallel_calls != 1:
        raise ValueError("Common evaluation permits one tool call per turn")
    if not 0 < float(rollout.val_kwargs.top_p) <= 1:
        raise ValueError("Invalid top_p")
    response_needed = budget.max_generated_tokens + calls * budget.max_tool_observation_tokens + budget.max_protocol_tokens
    if response_needed > config.data.max_response_length:
        raise ValueError("Response capacity cannot cover generated tokens, tool observations and control messages")
    if config.data.max_prompt_length + config.data.max_response_length > rollout.max_model_len:
        raise ValueError("Model context cannot cover prompt and response")
    if budget.max_new_tokens_per_turn + budget.final_answer_max_new_tokens > budget.max_generated_tokens:
        raise ValueError("Generation budget must reserve a final answer")
