# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""LLMBoost APIs."""

from mindspore.common import Tensor

class LLMBoost():
    r"""
    Implements an LLM in a single kernel.
    it forwards the python function to the C++ binded object
    """
    def __init__(self, config):
        r"""
        initialize the parameters of the llm binder.
        config is simply the config object of the model
        """
        from mindspore._c_expression import LlmBoostBinder
        self.config = config
        self.binder = LlmBoostBinder("AscendNative", config.model_type)
        self.binder.init_model(config.to_dict())

    def init(self):
        """
        Initialize the object
        returns True if object needs input manipulation by mindformers
        """
        return False

    def set_kvcache(self, k_caches=None, v_caches=None):
        return

    def forward(self, input_ids, batch_valid_length, position_ids=None):
        ret = self.binder.forward([input_ids, batch_valid_length], "nothing really")
        return Tensor(ret[0])

    def set_weights(self, ckpt_dict):
        self.binder.set_weights_map(ckpt_dict)

    def add_flags(self, is_first_iteration=False):
        self.binder.add_flags(is_first_iteration=is_first_iteration)
