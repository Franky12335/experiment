# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Local Ray environment for the fixed FSDP training setting."""
import json
import os
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}

def get_ppo_ray_runtime_env(config=None):
    working_dir = json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir")
    result = {"env_vars": {k: v for k, v in PPO_RAY_RUNTIME_ENV["env_vars"].items() if k not in os.environ},
              **({"working_dir": None} if working_dir is None else {})}
    for key in ("VERL_FULL_DETERMINISM", "PYTHONHASHSEED", "RUN_SEED", "CUBLAS_WORKSPACE_CONFIG",
                "FLASH_ATTENTION_DETERMINISTIC", "NCCL_DETERMINISTIC", "NCCL_ALGO"):
        if key in os.environ:
            result["env_vars"][key] = os.environ[key]
    return result
