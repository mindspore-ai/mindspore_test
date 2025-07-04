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
"""mint module."""
from __future__ import absolute_import

from mindspore.ops.function.math_func import inverse_ext as inv
from mindspore.ops.function.math_func import vector_norm_ext as vector_norm
from mindspore.ops.function.math_func import matrix_norm_ext as matrix_norm
from mindspore.ops.function.math_func import linalg_norm as norm
from mindspore.ops.auto_generate import linalg_qr as qr

__all__ = [
    'inv',
    'vector_norm',
    'matrix_norm',
    'norm',
    'qr',
]
