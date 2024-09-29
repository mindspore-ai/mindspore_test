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
        self.single_case_template = Template(
            'case ${case_id}:\n'
            '  ${device_dispatcher}'
            '  break;\n'
        )
        self.device_dispatcher_template = Template(
            'if (backend == kAscendDevice || backend == kDavinciDevice) {\n'
            '  ${ascend_dispatcher}\n'
            '} else if (backend == kCPUDevice) {\n'
            '  ${cpu_dispatcher}\n'
            '} else if (backend == kGPUDevice) {\n'
            '  ${gpu_dispatcher}\n'
            '} else {'
            '  MS_LOG(ERROR) << "Device target is not supported!";\n'
            '  return py::none();\n'
            '}'
        )
        self.aclnn_return_template = Template(
            '${arg_handler_processor}\n'
            'MS_LOG(INFO) << "Call Tensor${class_name}";\n'
            'return ToPython(Tensor${class_name}Register::GetOp()(arg_list));\n'
        )

        self.TENSOR_FUNC_CC_REG = template.TENSOR_FUNC_CC_REG
        self.TENSOR_FUNC_HEADER_REG = template.TENSOR_FUNC_HEADER_REG
        self.TENSOR_FUNC_HEADER_BODY = template.TENSOR_FUNC_HEADER_BODY
        self.TENSOR_FUNC_CALL_BODY = template.TENSOR_FUNC_CALL_BODY
        self.TENSOR_FUNC_OVERLOAD_CALL_BODY_REG = template.TENSOR_FUNC_OVERLOAD_CALL_BODY_REG

    def generate(self, work_path, func_protos_data: dict[str, list[TensorFuncProto]]):
        func_header_body_str = ''
        func_call_body_str = ''
        func_def_body_str = ''

        func_header_body_str, func_call_body_str, func_def_body_str = (
            self._get_single_op_str(func_protos_data, func_header_body_str, func_call_body_str, func_def_body_str))

        func_header_body_str, func_call_body_str, func_def_body_str = (
            self._get_overload_op_str(func_protos_data, func_header_body_str, func_call_body_str, func_def_body_str))

        func_cc_reg = self.TENSOR_FUNC_CC_REG.replace(func_call_body=func_call_body_str,
                                                      func_def_body=func_def_body_str)
        func_header_reg = self.TENSOR_FUNC_HEADER_REG.replace(func_header_body=func_header_body_str)
        save_file(os.path.join(work_path, K.TENSOR_FUNC_REGISTER_PATH), "tensor_func_reg.h", func_header_reg)
        save_file(os.path.join(work_path, K.TENSOR_FUNC_REGISTER_PATH), "tensor_func_reg.cc", func_cc_reg)

    def _get_single_op_str(self, func_protos_data,
                           func_header_body_str,
                           func_call_body_str,
                           func_def_body_str):
        single_op_func_data = {}
        for func_api_name, func_protos in func_protos_data.items():
            if len(func_protos) == 1:
                func_name = func_protos[0].func_name
                if func_name not in single_op_func_data:
                    single_op_func_data[func_name] = func_protos[0]

        for func_name, func_proto in single_op_func_data.items():
            func_name = func_proto.func_name
            class_name = func_proto.op_proto.op_class.name
            func_header_body_str += self.TENSOR_FUNC_HEADER_BODY.replace(class_name=class_name)
            device_dispatcher_str = self._get_device_dispatchers_str(func_proto)
            signature_str = self._generate_single_signature_str(func_proto.op_proto)
            func_call_body_str += self.TENSOR_FUNC_CALL_BODY.replace(class_name=class_name,
                                                                     device_dispatcher=device_dispatcher_str,
                                                                     signatures=signature_str)
            func_def_body_str += self.func_def_reg.replace(func_name=func_name,
                                                           class_name=class_name)
        return func_header_body_str, func_call_body_str, func_def_body_str

    def _get_overload_op_str(self, func_protos_data,
                             func_header_body_str,
                             func_call_body_str,
                             func_def_body_str):
        overload_op_func_data = {}
        for func_api_name, func_protos in func_protos_data.items():
            if len(func_protos) > 1:
                overload_op_func_data[func_api_name] = func_protos

        for func_api_name, func_protos in overload_op_func_data.items():
            func_header_body_str += self._get_overload_tensor_func_header_body_str(func_protos)
            func_call_body_str += self._get_overload_func_call_str(func_api_name, func_protos)
            func_def_body_str += self.func_def_reg.replace(func_name=func_api_name,
                                                           class_name=func_api_name.capitalize())
        return func_header_body_str, func_call_body_str, func_def_body_str

    def _get_overload_tensor_func_header_body_str(self, func_protos):
        overload_tensor_func_header_body_str = ''
        for tensor_proto in func_protos:
            overload_tensor_func_header_body_str += self.TENSOR_FUNC_HEADER_BODY.replace(
                class_name=tensor_proto.op_proto.op_class.name)
        return overload_tensor_func_header_body_str

    def _get_overload_func_call_str(self, func_api_name, func_protos):
        signatures_str = self._generate_func_signatures_str(func_protos)
        dispatch_cases_str = self._get_dispatch_cases(func_protos)
        overload_func_call_str = self.TENSOR_FUNC_OVERLOAD_CALL_BODY_REG.replace(class_name=func_api_name.capitalize(),
                                                                                 signatures=signatures_str,
                                                                                 dispatch_cases=dispatch_cases_str)
        return overload_func_call_str

    def _generate_func_signatures_str(self, func_protos) -> str:
        sig_str = ''
        first_sig = True
        for tensor_proto in func_protos:
            op_proto = tensor_proto.op_proto
            if not first_sig:
                sig_str += ',\n'
            first_sig = False
            sig_str += self._generate_single_signature_str(op_proto)
        return sig_str

    def _generate_single_signature_str(self, op_proto: OpProto) -> str:
        args_str = f'"{op_proto.op_class.name}('
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
        return args_str + ')"'

    def _get_dispatch_cases(self, func_protos):
        dispatch_cases_str = ''
        for idx, func_proto in enumerate(func_protos):
            device_dispatcher_str = self._get_device_dispatchers_str(func_proto)
            dispatch_cases_str += self.single_case_template.replace(case_id=idx,
                                                                    device_dispatcher=device_dispatcher_str)
        dispatch_cases_str += 'default:\n'
        dispatch_cases_str += '  return py::none();'
        return dispatch_cases_str

    def _get_device_dispatchers_str(self, func_proto):
        ascend_dispatcher_str = self._get_single_device_dispatcher_str(func_proto, 'ascend')
        cpu_dispatcher_str = self._get_single_device_dispatcher_str(func_proto, 'cpu')
        gpu_dispatcher_str = self._get_single_device_dispatcher_str(func_proto, 'gpu')
        device_dispatcher_str = self.device_dispatcher_template.replace(ascend_dispatcher=ascend_dispatcher_str,
                                                                        cpu_dispatcher=cpu_dispatcher_str,
                                                                        gpu_dispatcher=gpu_dispatcher_str)
        return device_dispatcher_str

    def _get_single_device_dispatcher_str(self, func_proto, device):

        callback_python_template = Template(
            'MS_LOG(INFO) << "${info}";\n'
            'py::function fn = python_adapter::GetPyFn(\"${python_module}\", \"${python_func}\");\n'
            'py::object res = fn(*args);\n'
            'return res;\n'
        )
        if getattr(func_proto, device) == 'aclnn':
            arg_handler_processor_str = self._get_arg_handler_processor(func_proto)
            return self.aclnn_return_template.replace(arg_handler_processor=arg_handler_processor_str,
                                                      class_name=func_proto.op_proto.op_class.name)
        else:
            python_module_and_func = getattr(func_proto, device)
            if '.' not in python_module_and_func:
                return (f'MS_LOG(ERROR) << "Callback python module and func is: {python_module_and_func}";\n'
                        f'return py::none();')
            last_doc_index = python_module_and_func.rindex('.')
            python_module = python_module_and_func[:last_doc_index]
            python_func = python_module_and_func[last_doc_index + 1:]
            return callback_python_template.replace(info=python_module_and_func,
                                                    python_module=python_module,
                                                    python_func=python_func)

    def _get_arg_handler_processor(self, func_proto):
        arg_handler_processor = ''
        op_proto = func_proto.op_proto
        op_args = op_proto.op_args
        for idx, op_arg in enumerate(op_args):
            arg_handler = op_arg.arg_handler
            if arg_handler:
                cc_arg_handler = ''.join(word.capitalize() for word in arg_handler.split('_'))
                arg_handler_processor += f"args[{idx}] = (*pynative::{cc_arg_handler}(args, kIndex{idx}))->value();\n"
        return arg_handler_processor
