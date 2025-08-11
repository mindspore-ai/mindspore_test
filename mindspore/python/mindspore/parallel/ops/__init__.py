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
"""Interfaces for parallel-related functionality"""
from __future__ import absolute_import

from .parallel_matmul import MatMulDistributedOp
from .parallel_elementwise import ElementWiseDistributedOp

_matmul_dist_op = MatMulDistributedOp()
_add_ext_dist_op = ElementWiseDistributedOp("AddExt")
_add_dist_op = ElementWiseDistributedOp("Add")
_relu_dist_op = ElementWiseDistributedOp("ReLU")
_sub_ext_dist_op = ElementWiseDistributedOp("SubExt")
_sub_dist_op = ElementWiseDistributedOp("Sub")
_mul_dist_op = ElementWiseDistributedOp("Mul")
_div_dist_op = ElementWiseDistributedOp("Div")
_floor_div_dist_op = ElementWiseDistributedOp("FloorDiv")
_real_div_dist_op = ElementWiseDistributedOp("RealDiv")
_add_scalar_dist_op = ElementWiseDistributedOp("AddScalar")
_sub_scalar_dist_op = ElementWiseDistributedOp("SubScalar")
