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

"""Tensor Func Proto module for defining tensor_py function prototypes and their arguments."""
from collections import defaultdict

from op_proto import OpProto


class TensorFuncProto:

    def __init__(self,
                 func_name,
                 op_proto,
                 ascend, gpu, cpu):
        self.func_name = func_name
        self.op_proto = op_proto
        self.ascend = ascend
        self.gpu = gpu
        self.cpu = cpu


def load_func_protos_from_yaml(tensor_func_yaml_data, op_protos):
    op_protos_dict = {}
    for op_proto in op_protos:
        op_protos_dict[op_proto.op_name] = op_proto
    func_protos = defaultdict(list)
    for func_name, tensor_func_data in tensor_func_yaml_data.items():
        func_data_list = [tensor_func_data] if isinstance(tensor_func_data, dict) else tensor_func_data
        for func_data in func_data_list:
            op_name = func_data.get('op_name', '')
            if op_name == '':
                raise TypeError('For generating tensor functions, op name should not be empty')
            op_proto = op_protos_dict.get(op_name, None)
            if op_proto is None:
                raise TypeError("For generating tensor functions, op_proto should not be empty")
            ascend = func_data.get('Ascend', 'aclnn')
            gpu = func_data.get('GPU', 'aclnn')
            cpu = func_data.get('CPU', 'aclnn')
            tensor_func_proto = TensorFuncProto(func_name=func_name,
                                                op_proto=op_proto,
                                                ascend=ascend, gpu=gpu, cpu=cpu)
            func_protos[func_name].append(tensor_func_proto)
    return func_protos
