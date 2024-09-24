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

from op_proto import OpProto


class TensorFuncProto:

    def __init__(self,
                 func_name,
                 op_name,
                 class_name,
                 ascend, gpu, cpu):
        self.func_name = func_name
        self.op_name = op_name
        self.class_name = class_name
        self.ascend = ascend
        self.gpu = gpu
        self.cpu = cpu

    @staticmethod
    def load_from_yaml(tensor_func_yaml_data, ops_yaml_str):
        op_protos = {}
        for operator_name, operator_data in ops_yaml_str.items():
            op_proto = OpProto.load_from_yaml(operator_name, operator_data)
            op_protos[op_proto.op_name] = op_proto
        func_protos = []
        for func_name, func_data in tensor_func_yaml_data.items():
            op_name = func_data.get('op_name', '')
            if op_name == '':
                raise TypeError('For generating tensor functions, op name should not be empty')
            op_proto = op_protos.get(op_name, None)
            if op_proto is None:
                raise TypeError("For generating tensor functions, op_proto should not be empty")
            ascend = func_data.get('Ascend', 'aclnn')
            gpu = func_data.get('GPU', 'python')
            cpu = func_data.get('CPU', 'python')
            tensor_func_proto = TensorFuncProto(func_name=func_name, op_name=op_proto.op_name,
                                                class_name=op_proto.op_class.name,
                                                ascend=ascend, gpu=gpu, cpu=cpu)
            func_protos.append(tensor_func_proto)
        return func_protos
