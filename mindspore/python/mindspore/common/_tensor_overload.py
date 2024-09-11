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

"""Mint adaptor."""

from __future__ import absolute_import
import os
from mindspore.common._register_for_tensor import tensor_operator_registry_for_mint

def log_mint(log):
    def wrapper(self, *args, **kwargs):
        if os.environ.get('MS_TENSOR_API_ENABLE_MINT') == '1':
            return  tensor_operator_registry_for_mint.get('log')(self, *args, **kwargs)
        return log(self, *args, **kwargs)
    return wrapper
