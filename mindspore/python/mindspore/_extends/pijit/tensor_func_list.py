# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Store and get tensor method"""
from typing import Optional, Dict

from mindspore.common.tensor import _TENSOR_ATTRIBUTES
from mindspore._c_expression import function_id


_tensor_method_id_to_name: Dict[int, str] = {
    function_id(attr): name for name, attr in _TENSOR_ATTRIBUTES.items() if callable(attr)
}


def get_tensor_method_name(id: int) -> Optional[str]:
    """Get method name by function id"""
    return _tensor_method_id_to_name.get(id, None)
