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
This module provides classes for generating C++ header and implementation files for functions based on op_protos.
"""

import os

import template
from template import Template
import gen_constants as K
from gen_utils import save_file
from base_generator import BaseGenerator
from pyboost_utils import is_optional_param, get_input_dtype, get_return_type


class FunctionsHeaderGenerator(BaseGenerator):
    """
    Generates C++ header files for backend functions based on operator prototypes.
    """

    def __init__(self):
        """
        Initialize the functions header generator with templates for code generation.
        """
        self.FUNCTIONS_H_TEMPLATE = template.FUNCTIONS_H_TEMPLATE
        self.function_interface_template = Template("${return_type} BACKEND_EXPORT ${op_name}(${input_args});")

    def generate(self, work_path, op_protos):
        """
        Generate the header file for backend functions.

        Args:
            work_path (str): The directory where the generated file should be saved.
            op_protos (list): A list of operator prototypes used to generate the header.
        """
        functions_list = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or op_proto.op_dispatch.is_comm_op:
                continue
            input_args_with_type_str = self._get_input_args(op_proto)
            return_type_str = _get_return_type_str(op_proto)
            functions = self.function_interface_template.replace(op_name=op_proto.op_name,
                                                                 input_args=input_args_with_type_str,
                                                                 return_type=return_type_str)
            functions_list.append(functions)
        pyboost_func_h_str = self.FUNCTIONS_H_TEMPLATE.replace(op_call_with_grad=functions_list)
        save_path = os.path.join(work_path, K.MS_OPS_KERNEL_FUNCTIONS_AUTO_GEN_PATH)
        file_name = "functions.h"
        save_file(save_path, file_name, pyboost_func_h_str)

    def _get_input_args(self, op_proto):
        """
        Get the input arguments with type information for the function interface.

        Args:
            op_proto: The operator prototype.

        Returns:
            str: A string of input arguments with types.
        """
        args_list = []
        for op_arg in op_proto.op_args:
            input_dtype = get_input_dtype(op_arg.arg_dtype, is_optional_param(op_arg))
            args_list.append("const " + input_dtype + " &" + op_arg.arg_name)
        return args_list


class FunctionsGenerator(BaseGenerator):
    """
    Generates C++ implementation files for backend functions based on operator prototypes.
    """

    def __init__(self):
        """
        Initialize the functions generator with templates for code generation.
        """
        self.FUNCTIONS_CC_TEMPLATE = template.FUNCTIONS_CC_TEMPLATE
        self.FUNCTION_BODY_TEMPLATE = template.FUNCTION_BODY_TEMPLATE
        self.pyboost_func_include_header_template = Template(
            f'#include "{K.MS_COMMON_PYBOOST_KERNEL_PATH}/auto_generate/${{operator_name}}.h"\n'
        )
        self.clone_inplace_input_template = Template(
            'GetCloneFunc()(op, prim::kPrim${class_name}, device_target, {${grad_args}});'
        )

    def generate(self, work_path, op_protos):
        """
        Generate the implementation file for backend functions.

        Args:
            work_path (str): The directory where the generated file should be saved.
            op_protos (list): A list of operator prototypes used to generate the implementation.
        """
        func_include_headers_list = []
        op_call_with_grad_list = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or op_proto.op_dispatch.is_comm_op:
                continue
            func_include_headers_list.append(
                self.pyboost_func_include_header_template.replace(operator_name=op_proto.op_name))
            op_call_with_grad_list.append(self._get_function_body(op_proto))
        pyboost_func_h_str = self.FUNCTIONS_CC_TEMPLATE.replace(op_call_with_grad=op_call_with_grad_list,
                                                                pyboost_op_header_include=func_include_headers_list)
        save_path = os.path.join(work_path, K.MS_OPS_KERNEL_FUNCTIONS_AUTO_GEN_PATH)
        file_name = "functions.cc"
        save_file(save_path, file_name, pyboost_func_h_str)

    def _get_function_body(self, op_proto):
        """
        Get the function body for a given operator prototype.

        Args:
            op_proto: The operator prototype.

        Returns:
            str: The generated function body.
        """
        input_args = self._get_input_args(op_proto, False)
        input_args_with_type = self._get_input_args(op_proto, True)
        inplace_clone_args = self._get_clone_input_args(op_proto, False, False)
        clone_func_str = self._get_clone_inplace_str(op_proto.op_inplace, op_proto.op_class.name, inplace_clone_args)
        return_type_str = _get_return_type_str(op_proto)
        return self.FUNCTION_BODY_TEMPLATE.replace(op_name=op_proto.op_name,
                                                   class_name=op_proto.op_class.name,
                                                   input_args=input_args,
                                                   clone_func=clone_func_str,
                                                   input_args_with_type=input_args_with_type,
                                                   return_type=return_type_str)

    def _get_input_args(self, op_proto, has_type):
        """
        Get the input arguments for the function body.

        Args:
            op_proto: The operator prototype.
            has_type (bool): Whether to include type information for the arguments.

        Returns:
            str: A string of input arguments, with or without types.
        """
        args_list = []
        for op_arg in op_proto.op_args:
            input_dtype = get_input_dtype(op_arg.arg_dtype, is_optional_param(op_arg))
            if has_type:
                args_list.append("const " + input_dtype + " &" + op_arg.arg_name)
            else:
                args_list.append(op_arg.arg_name)
        return args_list

    def _get_clone_inplace_str(self, is_inplace_op: bool, class_name: str, grad_args: list):
        """
        Generates the view base str of arguments for the operator.

        This method constructs a list of argument names that need to be cast to their corresponding types.

        Args:
            is_view_or_inplace (bool): Whether the op is view op or inplace op.
            grad_args (list): grad args

        Returns:
            str: Formatted view or inplace first argument names.
        """
        if not is_inplace_op:
            return ''
        return self.clone_inplace_input_template.replace(class_name=class_name, grad_args=grad_args)

    def _get_clone_input_args(self, op_proto, has_type, with_optional):
        """
        Get the input arguments for the DoGrad function.

        Args:
            op_proto: The operator prototype.
            has_type (bool): Whether to include type information for the arguments.

        Returns:
            list: A list of input arguments for the DoGrad function.
        """
        args_list = []
        for op_arg in op_proto.op_args:
            input_dtype = get_input_dtype(op_arg.arg_dtype, is_optional_param(op_arg))
            if has_type:
                args_list.append(f"const {input_dtype} &{op_arg.arg_name}")
            else:
                if not with_optional and is_optional_param(op_arg):
                    args_list.append(f"OptionalToValue({op_arg.arg_name})")
                else:
                    args_list.append(f"{op_arg.arg_name}")
        return args_list


def _get_return_type_str(op_proto):
    """
    Get the return type string for the function.

    Args:
        op_proto: The operator prototype.

    Returns:
        str: The return type as a string.
    """
    returns_type = []
    type_convert_to_base = {
        'std::vector<mindspore::tensor::TensorPtr>': 'std::vector<mindspore::tensor::BaseTensorPtr>',
        'mindspore::tensor::TensorPtr': 'mindspore::tensor::BaseTensorPtr'
    }
    for return_obj in op_proto.op_returns:
        temp_return = get_return_type(return_obj.arg_dtype)
        if temp_return in type_convert_to_base:
            returns_type.append(type_convert_to_base[temp_return])
        else:
            raise Exception("Not return found")
    if len(returns_type) == 1:
        cpp_func_return = returns_type[0]
    elif len(returns_type) > 1:
        cpp_func_return = "std::tuple<"
        cpp_func_return += ', '.join(s for s in returns_type)
        cpp_func_return += ">"
    else:
        raise Exception("Not return found")
    return cpp_func_return
