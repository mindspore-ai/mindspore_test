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
"""storeattrgrad op"""
from mindspore.ops.op_info_register import op_info_register, CpuRegOp, DataType

raise_op_info = CpuRegOp("raise") \
    .input(0, "x", "dynamic") \
    .output(0, "y", "dynamic") \
    .dtype_format(DataType.I32_Default, DataType.I32_Default) \
    .get_op_info()


@op_info_register(raise_op_info)
def _raise_cpu():
    """_getattrgrad_cpu cpu register"""
    return
