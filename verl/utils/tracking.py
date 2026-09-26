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
"""
A unified tracking interface that supports logging data to different backend
"""

"""Local console reporting for training metrics."""
from contextlib import contextmanager

class Tracking:
    def __init__(self, project_name, experiment_name, default_backend="console", config=None):
        backends = [default_backend] if isinstance(default_backend, str) else default_backend
        if any(name != "console" for name in (backends or [])):
            raise ValueError("Only console reporting is supported")
    def log(self, data, step, backend=None):
        print(f"step={step} " + " ".join(f"{key}={value}" for key, value in sorted(data.items())), flush=True)
    def finish(self, exit_code=0):
        pass

class ValidationGenerationsLogger:
    def __init__(self, project_name=None, experiment_name=None):
        pass
    def log(self, loggers, samples, step):
        for sample in samples:
            print(f"validation step={step}: {sample}", flush=True)

class DapoFilteredRewardTableLogger:
    def __init__(self, project_name=None, experiment_name=None):
        pass
    def log(self, loggers, reward_counts, step):
        print(f"reward counts step={step}: {reward_counts}", flush=True)

class RLInsightLogger:
    @staticmethod
    def enabled():
        return False
    @staticmethod
    @contextmanager
    def trace_state(name, **kwargs):
        yield
