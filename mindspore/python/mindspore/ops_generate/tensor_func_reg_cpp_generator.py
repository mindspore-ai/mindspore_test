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
from op_proto import OpProto
from tensor_func_proto import TensorFuncProto


class TensorFuncRegCppGenerator(BaseGenerator):

    def __init__(self):
        self.func_def_reg = Template("tensor_class->def(\"${func_name}\", Tensor${class_name});\n")

        self.TENSOR_FUNC_CC_REG = template.TENSOR_FUNC_CC_REG
        self.TENSOR_FUNC_HEADER_REG = template.TENSOR_FUNC_HEADER_REG
        self.TENSOR_FUNC_HEADER_BODY = template.TENSOR_FUNC_HEADER_BODY
        self.TENSOR_FUNC_CALL_BODY = template.TENSOR_FUNC_CALL_BODY
        self.TENSOR_FUNC_OVERLOAD_CALL_BODY_REG = template.TENSOR_FUNC_OVERLOAD_CALL_BODY_REG

    def generate(self, work_path, func_protos_data: dict[str, list[TensorFuncProto]], op_protos):
        func_header_body_str = ''
        func_call_body_str = ''
        func_def_body_str = ''
        for func_name, func_protos in func_protos_data.items():
            if len(func_protos) == 1:
                func_proto = func_protos[0]
                class_name = func_proto.op_proto.op_class.name
                func_header_body_str += self.TENSOR_FUNC_HEADER_BODY.replace(
                    class_name=class_name)
                func_call_body_str += self.TENSOR_FUNC_CALL_BODY.replace(
                    class_name=class_name)
                func_def_body_str += self.func_def_reg.replace(func_name=func_proto.func_name,
                                                               class_name=class_name)
            else:
                func_header_body_str += self.TENSOR_FUNC_HEADER_BODY.replace(
                    class_name=func_name.capitalize())
                func_def_body_str += self.func_def_reg.replace(func_name=func_name,
                                                               class_name=func_name.capitalize())
                # Process overload cases
                func_call_body_str += self._get_overload_func_call_body(func_name, func_protos)
                func_call_body_str += self.TENSOR_FUNC_CALL_BODY.replace(
                    class_name=func_name.capitalize())

        func_cc_reg = self.TENSOR_FUNC_CC_REG.replace(func_call_body=func_call_body_str,
                                                      func_def_body=func_def_body_str)
        func_header_reg = self.TENSOR_FUNC_HEADER_REG.replace(func_header_body=func_header_body_str)
        save_file(os.path.join(work_path, K.TENSOR_FUNC_REGISTER_PATH), "tensor_func_reg.h", func_header_reg)
        save_file(os.path.join(work_path, K.TENSOR_FUNC_REGISTER_PATH), "tensor_func_reg.cc", func_cc_reg)

    def _get_overload_func_call_body(self, func_name, func_protos):
        self.func_signatures_template = Template(
            'static PythonArgParser parser({\n'
            '  ${signatures}});')
        self.signature_template = Template("\"$func_name(${args})\"")

        self.func_dispatch_template = Template(
            'switch (converter.index_){\n'
            '  ${dispatch}\n'
            '}\n')

        self.dispatch_case_template = Template(
            'case ${index}: {\n'
            '  MS_LOG(INFO) << "Call Tensor${class_name}";'
            '  return ToPython(Tensor${class_name}Register::GetOp()(args));'
            '  break;}'
        )
        op_protos = [func_proto.op_proto for func_proto in func_protos]
        overload_args = [op_proto.op_args for op_proto in func_protos]

    def _generate_func_signature_str(self, op_proto: OpProto) -> str:
        args_str = ''
        first_arg = True
        for index, arg in enumerate(op_proto.op_args):
            single_arg = ''
            if not first_arg:
                single_arg = ', '
            first_arg = False
            arg_dtype = arg.arg_dtype
            arg_name = arg.arg_name
            single_arg += f"{arg_dtype} {arg_name}"
            if arg.as_init_arg:
                arg_default = str(arg.default)
                single_arg += '='
                single_arg += arg_default
            args_str += single_arg
        sig_str = self.signature_template.replace(func_name=op_proto.op_name, args=args_str)
        return self.func_signatures_template.replace(signatures=sig_str)

    def _generate_dispatch_str(self, op_proto: OpProto) -> str:
        index_num = '0'
        func_name = op_proto.op_class.name + 'Dispatcher'
        args_str = ''
        for index, arg in enumerate(op_proto.op_args):
            args_str += f"{arg.arg_name}, "
        dispatch_str = self.dispatch_case_template.replace(index=index_num,
                                                           class_name=op_proto.op_class.name)
        return self.func_dispatch_template.replace(dispatch=dispatch_str)
