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


class TensorFuncProto:
    """
    Represents a tensor function prototype with associated function name, operation prototype, and target devices.
    """

    def __init__(self,
                 func_name,
                 op_proto,
                 py_method,
                 ascend,
                 gpu,
                 cpu):
        self.func_name = func_name
        self.op_proto = op_proto
        self.py_method = py_method
        self.ascend = ascend
        self.gpu = gpu
        self.cpu = cpu


def load_func_protos_from_yaml(tensor_func_yaml_data, op_protos, deprecated_op_protos):
    """
    Loads tensor function prototypes from YAML data and returns them as a dictionary.
    """
    op_protos_dict = {}
    for op_proto in op_protos:
        op_protos_dict[op_proto.op_name] = op_proto
    for deprecated_op_proto in deprecated_op_protos:
        op_protos_dict[deprecated_op_proto.op_name] = deprecated_op_proto
    func_protos = defaultdict(list)
    alias_func_mapping = defaultdict(list)
    for func_name, tensor_func_data in tensor_func_yaml_data.items():
        func_data_list = [tensor_func_data] if isinstance(tensor_func_data, dict) else tensor_func_data
        for func_data in func_data_list:
            if 'alias' in func_data:
                alias_func_mapping[func_data['alias']].append(func_name)
                continue
            op_name = _get_op_name_from_op_yaml(func_data)
            op_proto = op_protos_dict.get(op_name, None)
            if op_proto is None:
                raise TypeError("For generating tensor functions, op_proto should not be empty")
            py_method = func_data.get('py_method', '')
            if py_method == '':
                raise TypeError('For generating tensor functions, py method should not be empty')
            ascend = func_data.get('Ascend', 'aclnn')
            gpu = func_data.get('GPU', 'aclnn')
            cpu = func_data.get('CPU', 'aclnn')
            tensor_func_proto = TensorFuncProto(func_name=func_name,
                                                op_proto=op_proto,
                                                py_method=py_method,
                                                ascend=ascend,
                                                gpu=gpu,
                                                cpu=cpu)
            func_protos[func_name].append(tensor_func_proto)
    return func_protos, alias_func_mapping


def _get_op_name_from_op_yaml(func_data: dict) -> str:
    """Extracts the operation name from the given YAML function data."""
    op_yaml = func_data.get('op_yaml', '')
    if op_yaml == '':
        raise TypeError('For generating tensor functions, op yaml should not be empty')
    if 'deprecated' in op_yaml:
        op_name = op_yaml.replace('/', '_').replace('_op.yaml', '')
    else:
        op_name = op_yaml.replace('_op.yaml', '')
    if op_name == '':
        raise TypeError('For generating tensor functions, op name should not be empty')
    return op_name
