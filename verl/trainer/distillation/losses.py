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

from dataclasses import dataclass
from verl.base_config import BaseConfig

@dataclass
class DistillationLossSettings(BaseConfig):
    use_topk: bool = True
    use_estimator: bool = False

def is_distillation_enabled(config):
    return config is not None and config.enabled

def get_distillation_loss_settings(loss_name):
    if loss_name != "forward_kl_topk":
        raise ValueError("The Teacher transport requires top-k log probabilities")
    return DistillationLossSettings()
