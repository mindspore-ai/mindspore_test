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
This module defines the PyboostInnerPrimGenerator class, which is responsible for generating Python primitive
wrappers for Pyboost operations. The generator constructs Python function definitions based on operator prototypes,
generates necessary import statements, and writes the generated content into Python source files.

The primary functionality is to take operator prototypes, extract relevant fields, and create Python function wrappers
that can be used to call the Pyboost primitive implementations.
"""

import os
import template
import gen_constants as K
from gen_utils import save_file
from template import Template
from base_generator import BaseGenerator


class TensorFuncRegCppGenerator(BaseGenerator):

    def __init__(self):
        self.func_def_reg = Template("tensor_class->def(\"${func_name}\", Tensor${class_name});\n")

        self.TENSOR_FUNC_CC_REG = template.TENSOR_FUNC_CC_REG
        self.TENSOR_FUNC_HEADER_REG = template.TENSOR_FUNC_HEADER_REG
        self.TENSOR_FUNC_HEADER_BODY = template.TENSOR_FUNC_HEADER_BODY
        self.TENSOR_FUNC_CALL_BODY = template.TENSOR_FUNC_CALL_BODY
        self.TENSOR_FUNC_CLASS_REG = template.TENSOR_FUNC_CLASS_REG

    def generate(self, work_path, func_protos):
        func_header_body_str = ''
        func_call_body_str = ''
        func_def_body_str = ''
        for func_proto in func_protos:
            func_header_body_str += self.TENSOR_FUNC_HEADER_BODY.replace(
                class_name=func_proto.class_name)
            func_call_body_str += self.TENSOR_FUNC_CALL_BODY.replace(
                class_name=func_proto.class_name)
            func_def_body_str += self.func_def_reg.replace(func_name=func_proto.func_name,
                                                           class_name=func_proto.class_name)

        func_cc_reg = self.TENSOR_FUNC_CC_REG.replace(func_call_body=func_call_body_str,
                                                      func_def_body=func_def_body_str)
        func_header_reg = self.TENSOR_FUNC_HEADER_REG.replace(func_header_body=func_header_body_str)
        save_file(os.path.join(work_path, K.TENSOR_FUNC_REGISTER_PATH), "tensor_func_reg.h", func_header_reg)
        save_file(os.path.join(work_path, K.TENSOR_FUNC_REGISTER_PATH), "tensor_func_reg.cc", func_cc_reg)
