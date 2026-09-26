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
"""Continuous Token builder factory and model-family resolution."""
"""Qwen token construction used by the paper setting."""
from .continuous_token import QwenContinuousTokenBuilder

def create_continuous_token_builder(tokenizer, *, model_family="auto", hf_model_type=None,
                                    chat_template_kwargs=None, mm_processor_kwargs=None,
                                    processor=None, **builder_kwargs):
    if hf_model_type not in ("qwen3", "qwen2"):
        raise ValueError("This package requires the paper's Qwen model family")
    if processor is not None and getattr(processor, "image_processor", None) is not None:
        raise ValueError("The paper setting uses text inputs")
    return QwenContinuousTokenBuilder(tokenizer, chat_template_kwargs=chat_template_kwargs, **builder_kwargs)
