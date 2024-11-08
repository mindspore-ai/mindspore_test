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
This module defines the OpTemplateParser class for parsing operator prototypes.

The OpTemplateParser class is responsible for converting attributes of OpProto instances into strings
that can be directly used to fill in code templates. It handles the parsing of argument types,
function signatures, and generates conversion stubs for PyBoost operations.
"""

import re

import pyboost_utils
from pyboost_utils import get_input_dtype, tuple_input_to_cpp_type, get_return_type, \
    number_input_to_cpp_type, get_const_number_convert, get_tuple_input_convert, is_optional_param

from op_proto import OpProto


class OpTemplateParser:
    """
    Parses operator prototypes and generates template-compatible strings.

    This class converts the attributes of an OpProto instance into the strings needed for code generation
    in PyBoost operations.

    Attributes:
        op_proto (OpProto): The operator prototype containing the relevant information.
    """

    def __init__(self, op_proto: OpProto):
        """
        Initializes the OpTemplateParser with the given operator prototype.

        Args:
            op_proto (OpProto): The operator prototype to be parsed.
        """
        self.op_proto = op_proto

    def _parse_call_args_types(self, op_args):
        """
        Parses the data types of the call arguments for the operator.

        Args:
            op_args (list): A list of operator arguments.

        Returns:
            list: A list of data types for the call arguments.
        """
        call_args_types = []
        for op_arg in op_args:
            is_optional = is_optional_param(op_arg)
            call_args_types.append(get_input_dtype(op_arg.arg_dtype, is_optional))
        return call_args_types

    def parse_call_args_with_types(self):
        """
        Parses the original call arguments and their types for the operator.

        Returns:
            list: A list of formatted strings representing the call arguments with their types.
        """
        call_args = self.parse_original_call_args(self.op_proto.op_args)
        call_args_types = self._parse_call_args_types(self.op_proto.op_args)
        call_args_with_types = []
        for type_name, arg_name in zip(call_args_types, call_args):
            call_args_with_types.append("const " + type_name + " &" + arg_name)
        return call_args_with_types

    def parse_need_malloc_tensors(self):
        """
        Parses the operator arguments to identify which tensors require memory allocation.

        Returns:
            tuple: A tuple containing:
                - need_malloc_tensors (list): Arguments that need memory allocation.
                - tensor_list_convert (list): Conversions needed for tensor lists.
                - call_args_with_tensor (list): The call arguments formatted for tensors.
        """
        need_malloc_tensors = []
        tensor_list_convert = []
        call_args_with_tensor = []
        call_args = self.parse_original_call_args(self.op_proto.op_args)
        for op_arg, call_arg in zip(self.op_proto.op_args, call_args):
            if pyboost_utils.is_tensor(op_arg):
                call_arg = op_arg.arg_name + "_tensor"
                need_malloc_tensors.append(call_arg)
                call_args_with_tensor.append(call_arg)
            elif tuple_input_to_cpp_type(op_arg.arg_dtype) and pyboost_utils.is_tensor_list(op_arg):
                need_malloc_tensors.append(call_arg + "_vector")
                tensor_list_convert.append(get_tuple_input_convert(call_arg, op_arg.arg_dtype))
                call_args_with_tensor.append(call_arg + "_vector")
            else:
                call_args_with_tensor.append(call_arg)
        return need_malloc_tensors, tensor_list_convert, call_args_with_tensor

    def parse_original_call_args(self, op_args):
        """
        Parses the original call arguments from the operator prototype.

        Args:
            op_args (list): A list of operator arguments.

        Returns:
            list: A list of formatted strings representing the original call arguments.
        """
        call_args = []
        for op_arg in op_args:
            if pyboost_utils.is_tensor(op_arg):
                call_arg = op_arg.arg_name + "_tensor"
            elif pyboost_utils.is_tensor_list(op_arg):
                call_arg = op_arg.arg_name + "_tensor_list"
            else:
                call_arg = op_arg.arg_name
            call_args.append(call_arg)
        return call_args

    def op_args_converter(self):
        """
        Converts operator arguments to the corresponding C++ data types.

        Returns:
            tuple: A tuple containing:
                - call_args_after_convert (list): The converted call arguments.
                - value_tuple_convert (list): Conversions needed for value tuples.
                - const_number_convert (list): Conversions needed for constant numbers.
        """
        call_args_after_convert = []
        value_tuple_convert = []
        const_number_convert = []
        call_args = self.parse_original_call_args(self.op_proto.op_args)
        for op_arg, call_arg in zip(self.op_proto.op_args, call_args):
            if number_input_to_cpp_type(op_arg.arg_dtype):
                call_args_after_convert.append(call_arg + "_imm")
                const_number_convert.append(get_const_number_convert(call_arg, op_arg))
            elif tuple_input_to_cpp_type(op_arg.arg_dtype):
                call_args_after_convert.append(call_arg + "_vector")
                value_tuple_convert.append(get_tuple_input_convert(call_arg, op_arg.arg_dtype))
            else:
                call_args_after_convert.append(call_arg)
        if const_number_convert:
            const_number_convert.insert(0, '// Convert ValuePtr to c++ scalar\n')
        if value_tuple_convert:
            value_tuple_convert.insert(0, '// ValueTuple to std::vector\n')
        return call_args_after_convert, value_tuple_convert, const_number_convert

    def get_pyboost_func_name(self):
        """
        Gets the PyBoost function name based on the operator's class name.

        Returns:
            str: The generated PyBoost function name.
        """
        return "Pyboost_" + self.op_proto.op_class.name

    def get_pyboost_name(self):
        """
        Gets the PyBoost name for the operator.

        Returns:
            str: The generated PyBoost name for the operator.
        """
        return "pyboost_" + self.op_proto.op_name

    def get_op_def_name_str(self):
        """
        Gets the operator definition name string.

        Returns:
            str: The generated operator definition name string.
        """
        return "g" + self.op_proto.op_class.name

    def gen_signature_same_type_table(self):
        """
        Generates a signature table for arguments of the same type.

        Returns:
            tuple: A tuple containing:
                - type_num (int): The number of argument types.
                - signature_table (str): The generated signature table as a string.
        """
        signature_table = ''
        type_num = 0
        args_signature = self.op_proto.op_args_signature
        if args_signature is not None:
            dtype_group = args_signature.dtype_group
            indexes = {arg.arg_name: index for index, arg in enumerate(self.op_proto.op_args)}
            if dtype_group is not None:
                match = re.findall(r'\((.*?)\)', dtype_group)
                for item in match:
                    name_args = item.replace(' ', '').split(",")
                    signature_table += '{'
                    for arg in name_args:
                        arg_index = indexes[arg]
                        signature_table += f"""{arg_index}, """
                    signature_table = signature_table[:-2]
                    signature_table += '}, '
                    type_num += 1
                signature_table = signature_table[:-2]
        return type_num, signature_table

    def get_call_args_tensor(self):
        """
        Retrieves the call arguments that are of tensor type.

        Returns:
            list: A list of call arguments that are tensors.
        """
        call_args_tensor = []
        call_args_types = self._parse_call_args_types(self.op_proto.op_args)
        call_args = self.parse_original_call_args(self.op_proto.op_args)
        for type, arg_name in zip(call_args_types, call_args):
            if type in ("BaseTensorPtr", "std::optional<BaseTensorPtr>"):
                call_args_tensor.append(arg_name)
        return call_args_tensor

    def has_prim_init(self):
        """
        Checks if any arguments require primitive initialization.

        Returns:
            bool: True if any argument requires primitive initialization, otherwise False.
        """
        op_args = self.op_proto.op_args
        has_prim_init = False
        for op_arg in op_args:
            prim_init = op_arg.is_prim_init
            if prim_init:
                has_prim_init = True
                break
        return has_prim_init

    def generate_pyboost_op_func_return_type(self):
        """
        Generates the C++ return type for the PyBoost operator function.

        Returns:
            str: The generated C++ return type for the function.

        Raises:
            Exception: If no valid return type is found.
        """
        returns_type = []
        type_convert_to_base = {
            'std::vector<tensor::TensorPtr>': 'std::vector<tensor::BaseTensorPtr>',
            'tensor::TensorPtr': 'tensor::BaseTensorPtr'
        }
        for return_obj in self.op_proto.op_returns:
            temp_return = get_return_type(return_obj.arg_dtype)
            if temp_return in type_convert_to_base:
                returns_type.append(type_convert_to_base[temp_return])
            else:
                raise Exception("Not return found")
        if len(returns_type) == 1:
            cpp_func_return = returns_type[0]
        elif len(returns_type) > 1:
            cpp_func_return = "std::tuple<"
            cpp_func_return += ','.join(s for s in returns_type)
            cpp_func_return += ">"
        else:
            raise Exception("Not return found")
        return cpp_func_return

    def generate_pyboost_outputs(self):
        """
        Generates the output variables for the PyBoost operator function.

        Returns:
            tuple: A tuple containing:
                - op_outputs (str): The output variable representation for the operator.
                - call_outputs (str): The call output variable representation for the operator.
        """
        op_outputs = ''
        call_outputs = ''
        returns_type = []
        for return_obj in self.op_proto.op_returns:
            returns_type.append(get_return_type(return_obj.arg_dtype))

        if len(returns_type) == 1:
            if returns_type[0] == 'tensor::TensorPtr':
                op_outputs = 'outputs[0]'
                call_outputs = 'outputs_[0]'
            elif returns_type[0] == "std::vector<tensor::TensorPtr>":
                op_outputs = 'outputs'
                call_outputs = 'outputs_'
            else:
                raise Exception("Not support return type {}".format(returns_type[0]))
        elif len(returns_type) > 1:
            outputs_str = ''
            for i in range(len(returns_type)):
                outputs_str += 'outputs[{}],'.format(i)
            op_outputs = outputs_str[:-1]

            outputs_str = ''
            for i in range(len(returns_type)):
                outputs_str += 'outputs_[{}],'.format(i)
            outputs_str = outputs_str[:-1]
            call_outputs = "std::make_tuple(" + outputs_str + ")"

        return op_outputs, call_outputs
