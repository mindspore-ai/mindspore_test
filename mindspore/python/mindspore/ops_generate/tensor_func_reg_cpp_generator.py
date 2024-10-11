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
from template import Template
import gen_constants as K
from gen_utils import save_file
from base_generator import BaseGenerator
from op_proto import OpProto


class TensorFuncRegCppGenerator(BaseGenerator):
    """
    Generates C++ tensor function registration code for different backends (Ascend, CPU, GPU).

    This class is responsible for generating header and implementation files required to register
    tensor functions, including device-specific dispatchers and function definitions.
    """

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
        self.pyboost_return_template = Template(
            '${arg_handler_processor}\n'
            'MS_LOG(INFO) << "Call Tensor${class_name}";\n'
            'return ToPython(Tensor${class_name}Register::GetOp()(arg_list));\n'
        )
        self.callback_python_template = Template(
            'MS_LOG(INFO) << "Callback python method: ${py_method}";\n'
            'py::function fn = python_adapter::GetPyFn(\"mindspore.ops.tensor_method\", \"${py_method}\");\n'
            'py::object res = fn(*arg_list);\n'
            'return res;\n'
        )

        self.TENSOR_FUNC_CC_REG = template.TENSOR_FUNC_CC_REG
        self.TENSOR_FUNC_HEADER_REG = template.TENSOR_FUNC_HEADER_REG
        self.TENSOR_FUNC_HEADER_BODY = template.TENSOR_FUNC_HEADER_BODY
        self.TENSOR_FUNC_CALL_BODY = template.TENSOR_FUNC_CALL_BODY
        self.TENSOR_FUNC_OVERLOAD_CALL_BODY_REG = template.TENSOR_FUNC_OVERLOAD_CALL_BODY_REG
        # The format of arg_handler_map is {arg_handler_name : list of supported types}.
        # The first one of type list is the target dtype. Types corresponds to type_str_map.
        self.arg_handler_map = {"to_2d_paddings": "tuple[int]|list[int]|int",
                                "dtype_to_type_id": "int|type",
                                "to_kernel_size": "tuple[int]|list[int]|int",
                                "to_strides": "tuple[int]|list[int]|int",
                                "str_to_enum": "int|str",
                                "to_pair": "tuple[int]|list[int]|int|float",
                                "to_dilations": "tuple[int]|list[int]|int",
                                "to_output_padding": "tuple[int]|list[int]|int",
                                "to_rates": "tuple[int]|list[int]|int"}

    def generate(self, work_path, func_protos_data):
        """
        Generates C++ header and source files for tensor function registrations.

        Args:
            work_path (str): The directory where the generated files will be saved.
            func_protos_data (dict): Dictionary mapping function names to lists of TensorFuncProto objects.

        The function constructs C++ registration strings from the provided tensor function prototypes
        and writes the output to the specified work path.
        """
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
        """
        Generates C++ strings for single operation function registrations.

        Args:
            func_protos_data (dict): Dictionary of tensor function prototypes.
            func_header_body_str (str): Header body string to append generated code to.
            func_call_body_str (str): Function call body string to append generated code to.
            func_def_body_str (str): Function definition body string to append generated code to.

        Returns:
            tuple: Updated header body, call body, and definition body strings.
        """
        single_op_func_data = {}
        for _, func_protos in func_protos_data.items():
            if len(func_protos) == 1:
                func_name = func_protos[0].func_name
                if func_name not in single_op_func_data:
                    single_op_func_data[func_name] = func_protos[0]

        cls_names = set()
        for func_name, func_proto in single_op_func_data.items():
            func_name = func_proto.func_name
            class_name = func_proto.op_proto.op_class.name
            device_dispatcher_str = self._get_device_dispatchers_str(func_proto)
            signature_str = self._generate_single_signature_str(func_proto.op_proto)
            if class_name not in cls_names:
                func_header_body_str += self.TENSOR_FUNC_HEADER_BODY.replace(class_name=class_name)
                cls_names.add(class_name)
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
        """
        Generates C++ strings for overloaded operation function registrations.

        Args:
            func_protos_data (dict): Dictionary of tensor function prototypes.
            func_header_body_str (str): Header body string to append generated code to.
            func_call_body_str (str): Function call body string to append generated code to.
            func_def_body_str (str): Function definition body string to append generated code to.

        Returns:
            tuple: Updated header body, call body, and definition body strings.
        """
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
        """
        Generates C++ header body string for overloaded tensor functions.

        Args:
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated header body string for the overloaded functions.
        """
        overload_tensor_func_header_body_str = ''
        for tensor_proto in func_protos:
            overload_tensor_func_header_body_str += self.TENSOR_FUNC_HEADER_BODY.replace(
                class_name=tensor_proto.op_proto.op_class.name)
        return overload_tensor_func_header_body_str

    def _get_overload_func_call_str(self, func_api_name, func_protos):
        """
        Generates C++ call body string for overloaded tensor functions.

        Args:
            func_api_name (str): Name of the function API.
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated call body string for the overloaded functions.
        """
        signatures_str = self._generate_func_signatures_str(func_protos)
        dispatch_cases_str = self._get_dispatch_cases(func_protos)
        overload_func_call_str = self.TENSOR_FUNC_OVERLOAD_CALL_BODY_REG.replace(class_name=func_api_name.capitalize(),
                                                                                 signatures=signatures_str,
                                                                                 dispatch_cases=dispatch_cases_str)
        return overload_func_call_str

    def _generate_func_signatures_str(self, func_protos) -> str:
        """
        Generates function signatures as a string from the given prototypes.

        Args:
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated function signatures string.
        """
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
        """
        Generates a single function signature string for the given operation prototype.

        Args:
            op_proto (OpProto): Operation prototype to generate the signature for.

        Returns:
            str: Generated function signature string.
        """
        args_str = f'"{op_proto.op_class.name}('
        first_arg = True
        for _, arg in enumerate(op_proto.op_args):
            single_arg = ''
            if not first_arg:
                single_arg = ', '
            first_arg = False
            arg_handler = arg.arg_handler
            if arg_handler != '':
                if arg_handler in self.arg_handler_map:
                    arg_dtype = self.arg_handler_map[arg_handler]
                else:
                    raise ValueError("Generate failed. Check if {} is registered in TensorFuncRegCppGenerator."
                                     .format(arg_handler))
            else:
                arg_dtype = arg.arg_dtype
                for cast_type in arg.type_cast:
                    arg_dtype += '|'
                    arg_dtype += cast_type
            arg_name = arg.arg_name
            single_arg += f"{arg_dtype} {arg_name}"
            if arg.as_init_arg:
                arg_default = str(arg.default)
                single_arg += '='
                single_arg += arg_default
            args_str += single_arg
        return args_str + ')"'

    def _get_dispatch_cases(self, func_protos):
        """
        Generates C++ switch-case statements for dispatching tensor function calls.

        Args:
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated switch-case dispatch statements.
        """
        dispatch_cases_str = ''
        for idx, func_proto in enumerate(func_protos):
            device_dispatcher_str = self._get_device_dispatchers_str(func_proto)
            dispatch_cases_str += self.single_case_template.replace(case_id=idx,
                                                                    device_dispatcher=device_dispatcher_str)
        dispatch_cases_str += 'default:\n'
        dispatch_cases_str += '  return py::none();'
        return dispatch_cases_str

    def _get_device_dispatchers_str(self, func_proto):
        """
        Generates device-specific dispatch strings for the given function prototype.

        Args:
            func_proto (TensorFuncProto): Function prototype to generate dispatch strings for.

        Returns:
            str: Generated device-specific dispatch string.
        """
        ascend_dispatcher_str = self._get_single_device_dispatcher_str(func_proto, 'ascend')
        cpu_dispatcher_str = self._get_single_device_dispatcher_str(func_proto, 'cpu')
        gpu_dispatcher_str = self._get_single_device_dispatcher_str(func_proto, 'gpu')
        device_dispatcher_str = self.device_dispatcher_template.replace(ascend_dispatcher=ascend_dispatcher_str,
                                                                        cpu_dispatcher=cpu_dispatcher_str,
                                                                        gpu_dispatcher=gpu_dispatcher_str)
        return device_dispatcher_str

    def _get_single_device_dispatcher_str(self, func_proto, device):
        """
        Generates the dispatch string for a specific device.

        Args:
            func_proto (TensorFuncProto): Function prototype to generate the dispatcher for.
            device (str): Device type ('ascend', 'cpu', 'gpu').

        Returns:
            str: Generated device dispatcher string.
        """
        func_proto_device = getattr(func_proto, device)
        if func_proto_device == 'pyboost':
            arg_handler_processor_str = self._get_arg_handler_processor(func_proto)
            return self.pyboost_return_template.replace(arg_handler_processor=arg_handler_processor_str,
                                                        class_name=func_proto.op_proto.op_class.name)

        if func_proto_device == 'py_method':
            return self.callback_python_template.replace(py_method=func_proto.py_method)

        raise TypeError("Only support pyboost or python_method.")


    def _get_arg_handler_processor(self, func_proto):
        """
        Generates argument handler processing code for the given function prototype.

        Args:
            func_proto (TensorFuncProto): Function prototype to generate argument processing for.

        Returns:
            str: Generated argument handler processing code.
        """
        arg_handler_processor = ''
        op_proto = func_proto.op_proto
        op_args = op_proto.op_args
        for idx, op_arg in enumerate(op_args):
            arg_handler = op_arg.arg_handler
            if arg_handler:
                func_str = ''.join(word.capitalize() for word in arg_handler.split('_'))
                arg_handler_processor += f"arg_list[{idx}] = (*pynative::{func_str}(arg_list, kIndex{idx}))->value();\n"
        return arg_handler_processor
