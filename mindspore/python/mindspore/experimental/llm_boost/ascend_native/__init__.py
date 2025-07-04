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
"""
Provide llm boost for inference, such as LlamaBoost.
"""
from __future__ import absolute_import

from mindspore.experimental.llm_boost.ascend_native.llama_boost_ascend_native import LlamaBoostAscendNative

__all__ = ['LlamaBoostAscendNative']
