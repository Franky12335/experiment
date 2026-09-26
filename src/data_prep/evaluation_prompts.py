"""Evaluation prompts; LCB examples come only from the public prompt."""

import json

answer_format =  "\nRemember once you make sure the current answer is your final answer, do not call the tools again and directly output the final answer in the following text format, the answer format must be: \\boxed{'The final answer goes here.'}."

math_prompt_2 = "\n\nThe tool could be used for more precise and efficient calculation and could help you to verify your result before you reach the final answer."

agent_prompt= "\n\n**Note: You should first analyze the problem and form a high-level solution strategy, then utilize the tools to help you solve the problem.**"

def map_fn_gpqa(row: dict, *, data_source: str=None):
    problem, answer, domain = (row['problem'], row['solution'], row['domain'])
    prompt_1 = f'Analyze and solve the following {domain} problem step by step. \n\n'
    prompt = prompt_1 + problem + math_prompt_2 + agent_prompt + answer_format + '\n Here you need to put the final uppercase letter option of this problem into \\boxed{}'
    data = {'data_source': row['data_source'], 'prompt': [{'role': 'user', 'content': prompt}], 'ability': domain, 'reward_model': {'ground_truth': str(answer)}, 'agent_name': 'tool_agent'}
    return data

def map_lcb(row: dict, *, data_source: str=None):
    problem = row['problem']
    start_prompt = 'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.\n\n'
    code_prompt = '\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n```python\n# YOUR CODE HERE\n"'
    reward_model = json.loads(row['reward_model'])
    inputs = reward_model['ground_truth']['inputs']
    outputs = reward_model['ground_truth']['outputs']
    public_examples = '\nHere are some input and output examples of the expected code:\nInput:' + str(inputs) + '\nOutput:\n' + str(outputs)
    prompt = start_prompt + problem + public_examples + agent_prompt + code_prompt + '\nBefore sumbit your code, you can utilize tools to check the correctness of your code, once you make sure the current code is correct, do not call the tools again and submit your code within ```python\n# YOUR CODE HERE\n```.'
    data = {'data_source': row['data_source'], 'prompt': [{'role': 'user', 'content': prompt}], 'ability': row['ability'], 'reward_model': json.loads(row['reward_model']), 'agent_name': 'tool_agent', 'extra_info': json.loads(row['extra_info'])}
    return data


def gpqa_prompt(row):
    if str(row["solution"]) not in "ABCD" or len(str(row["solution"])) != 1:
        raise ValueError("GPQA requires one uppercase option")
    return map_fn_gpqa(row)["prompt"]

def lcb_prompt(row):
    from src.data_prep.canonical_prompt import _LEGACY_CODE_SUFFIX
    text=row["messages"][-1]["content"]
    prefix="Write a correct Python program for the specification below. It must read standard input and write standard output.\n\n"
    if not text.startswith(prefix) or not text.endswith(_LEGACY_CODE_SUFFIX):
        raise ValueError("Unexpected LCB source prompt")
    body=text[len(prefix):-len(_LEGACY_CODE_SUFFIX)]
    statement,sep,examples=body.rpartition("\n\nPublic input/output cases:\n")
    if not sep:
        raise ValueError("Missing public examples")
    public=json.loads(examples)
    adapted=dict(problem=statement,data_source=row["data_source"],ability=row["ability"],
                 reward_model=json.dumps({"ground_truth":public}),extra_info="{}")
    return map_lcb(adapted)["prompt"]

