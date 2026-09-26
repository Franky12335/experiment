# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ast
import json
import re
import traceback
import numpy as np
from src.evaluation.sandbox_judge import check_correctness

def compute_score(completion, test_cases, timeout=30):
    """Score the final Python submission against standard input/output tests."""
    if '</think>' in completion:
        solution_str = completion.split('</think>')[1]
    else:
        solution_str = completion
    test_cases = str(test_cases)
    try:
        solutions = re.findall('```python\\n(.*?)```', solution_str, re.DOTALL)
        if len(solutions) == 0:
            return {'score': -1.0, 'acc': False, 'pred': None}
        else:
            solution = solutions[-1]
            try:
                solution = solution.replace('\\n', '\n').replace('\\t', '\t')
                tree = ast.parse(solution)
            except:
                traceback.print_exc(10)
                return {'score': -1.0, 'acc': False, 'pred': None}
        if isinstance(test_cases, str):
            try:
                input_output = json.loads(test_cases)
            except:
                tmp = json.dumps(test_cases)
                input_output = json.loads(tmp)
        elif isinstance(test_cases, dict):
            input_output = test_cases
            test_cases = json.dumps(test_cases)
        else:
            assert False
        metrics = check_correctness(sandbox_fusion_url='<your_sandbox_fusion_url>', in_outs=json.loads(json.dumps(test_cases)), generation=solution, timeout=timeout)
        metrics = list(metrics)
        fixed = []
        for e in metrics[0]:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        metrics[0] = fixed
        return {'score': sum((x if x in [False, True] else False for x in metrics[0])) / len(metrics[0]), 'acc': sum(metrics[0]) == len(metrics[0]), 'pred': solution}
    except Exception as e:
        traceback.print_exc(10)
        return {'score': -1.0, 'acc': False, 'pred': solution}
