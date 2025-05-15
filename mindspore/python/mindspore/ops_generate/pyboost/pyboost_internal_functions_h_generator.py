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
"""
This module defines the `PyboostInternalFunctionsHeaderGenerator` class, which is used to generate the header file
(`functions.h`) that contains function declarations for internal op in Pyboost.

The class uses templates and operation prototypes to create function declarations based on the
operation's primitive and arguments. The generated file is saved to the specified path.
"""

import os

import common.template as template
import common.gen_constants as K
from common.template import Template
from common.gen_utils import save_file
from common.base_generator import BaseGenerator

from .pyboost_utils import is_optional_param
from .op_template_parser import OpTemplateParser


class PyboostInternalFunctionsHeaderGenerator(BaseGenerator):
    """
    A class to generate the `functions.h` header file, which contains internal op function declarations.
    """

    def __init__(self):
        """Initializes the PyboostInternalFunctionsHeaderGenerator with the necessary templates."""
        self.pyboost_internal_function_header_template = template.PYBOOST_INTERNAL_FUNCTION_HEADER_TEMPLATE

        self.pyboost_internal_func_template = Template(
            'void internal_${operator_name}(const std::shared_ptr<pyboost::OpRunner> &op, ${call_args_with_type});'
        )

    def generate(self, work_path, op_protos):
        """
        Generates the Pyboost internal function header file (`functions.h`).

        Args:
            work_path (str): The directory where the generated file will be saved.
            op_protos (list): A list of operation prototypes to parse and convert into function declarations.

        Returns:
            None: The method writes the generated header file to the specified directory.
        """
        func_list = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            if getattr(op_proto.op_dispatch, 'internal_op_ascend') == 'None':
                continue
            operator_name = op_proto.op_name
            call_args_with_types = self.get_call_args_with_type(op_proto)
            func_list.append(self.pyboost_internal_func_template.replace(operator_name=operator_name,
                                                                         call_args_with_type=call_args_with_types))

        if not func_list:
            return
        pyboost_internal_func_h_str = \
            self.pyboost_internal_function_header_template.replace(internal_func_list=func_list)
        save_path = os.path.join(work_path, K.MS_PYBOOST_INTERNAL_FUNCTIONS_AUTO_GEN_PATH)
        file_name = "functions.h"
        save_file(save_path, file_name, pyboost_internal_func_h_str)


    def _parse_call_args_types(self, op_args):
        call_args_types = []
        for op_arg in op_args:
            is_optional = is_optional_param(op_arg)
            if op_arg.is_type_id:
                call_args_types.append('TypeId')
                continue
            call_args_types.append(self._get_convert_dtype(op_arg.arg_dtype, is_optional))
        return call_args_types

    def _get_convert_dtype(self, arg_dtype, is_optional=False):
        """convert dtype to mindspore type"""
        type_convert = {
            'int': 'int64_t',
            'float': 'float',
            'bool': 'bool',
            'number': 'mindspore::ScalarPtr',
            'str': 'string',
            'tensor': 'mindspore::tensor::TensorPtr',
            'tuple[int]': 'std::vector<int64_t>',
            'tuple[float]': 'std::vector<float>',
            'tuple[bool]': 'std::vector<bool>',
            'tuple[tensor]': 'std::vector<TensorPtr>',
            'list[int]': 'std::vector<int64_t>',
            'list[float]': 'std::vector<float>',
            'list[bool]': 'std::vector<bool>',
            'list[tensor]': 'std::vector<TensorPtr>',
        }

        optional_tensor_type_convert = {
            'tensor': 'std::optional<mindspore::tensor::TensorPtr>',
            'tuple[tensor]': 'std::optional<mindspore::tensor::TensorPtr>',
            'list[tensor]': 'std::optional<mindspore::tensor::TensorPtr>'
        }

        if is_optional and arg_dtype in optional_tensor_type_convert:
            return optional_tensor_type_convert[arg_dtype]

        if arg_dtype in type_convert:
            return type_convert[arg_dtype]
        raise TypeError(f"""Unsupported dtype {arg_dtype} for args.""")

    def get_call_args_with_type(self, op_proto):
        """Get call args with cpp type according to op proto"""
        op_parser = OpTemplateParser(op_proto)
        call_args_after_convert, _, _ = op_parser.op_args_converter()
        call_args_type = self._parse_call_args_types(op_proto.op_args)
        call_args_with_types = []
        for call_arg, arg_dtype in zip(call_args_after_convert, call_args_type):
            call_args_with_types.append("const " + arg_dtype + " &" + call_arg)
        return call_args_with_types
