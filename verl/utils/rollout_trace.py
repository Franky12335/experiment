# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Compatibility contexts for local execution with no remote telemetry."""
from contextlib import contextmanager

class RolloutTraceConfig:
    _instance = None
    backend = None
    client = None
    token2text = False
    max_samples_per_step_per_worker = None
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    @classmethod
    def init(cls, project_name, experiment_name, backend, token2text=False, max_samples_per_step_per_worker=None):
        if backend is not None:
            raise ValueError("Remote telemetry is not included")
        cls.get_instance()
    @classmethod
    def get_backend(cls):
        return None
    @classmethod
    def get_client(cls):
        return None
    @classmethod
    def enable_token2text(cls):
        return False
    @classmethod
    def reset(cls):
        cls._instance = None

@contextmanager
def rollout_trace_attr(sample_index=None, step=None, rollout_n=None, name="rollout_trace", validate=False, trace=True):
    yield

def rollout_trace_op(func):
    return func
