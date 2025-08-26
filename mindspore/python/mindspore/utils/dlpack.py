
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""dlpack for tensor."""
from mindspore._c_expression import TensorPy as TensorPy_

def from_dlpack(dlpack):
    r"""
    Convert dlpack to Tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.
    """
    return TensorPy_.from_dlpack(dlpack)


def to_dlpack(tensor):
    r"""
    Convert tensor to dlpack.

    .. warning::
        This is an experimental API that is subject to change or deletion.
    """
    if tensor.has_init:
        tensor.init_data()
    return TensorPy_.to_dlpack(tensor)
