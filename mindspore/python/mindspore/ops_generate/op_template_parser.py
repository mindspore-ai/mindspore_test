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
import gen_constants as K
from template import Template
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
        for _type, arg_name in zip(call_args_types, call_args):
            if _type in ("mindspore::tensor::BaseTensorPtr", "std::optional<mindspore::tensor::BaseTensorPtr>"):
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
            'std::vector<mindspore::tensor::TensorPtr>': 'std::vector<mindspore::tensor::BaseTensorPtr>',
            'mindspore::tensor::TensorPtr': 'mindspore::tensor::BaseTensorPtr'
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
            if returns_type[0] == 'mindspore::tensor::TensorPtr':
                op_outputs = 'outputs[0]'
                call_outputs = 'outputs_[0]'
            elif returns_type[0] == "std::vector<mindspore::tensor::TensorPtr>":
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

    def _is_input_arg(self, arg_name, op_name):
        res = False
        if op_name in K.INPUT_NAME_MAP and arg_name == K.INPUT_NAME_MAP[op_name]:
            res = True
        elif op_name not in K.INPUT_NAME_MAP and arg_name in K.INPUT_ARGS_NAME:
            res = True
        return res

    def generate_signature_str(self, kw_only_args=None, varargs=None, *, is_tensor_api: bool) -> str:
        """
        Generates a single function signature string for the given operation prototype.

        Args:
            kw_only_args (list[str]): List of keyword-only argument names.
            varargs (list[str]): List of variable args names.

        Kwargs:
            is_tensor_api (bool): Whether this function is used in the Tensor API scenario.

        Returns:
            str: Generated function signature string.
        """

        op_name = self.op_proto.op_class.name
        args_str = f'"{op_name}('
        first_arg = True
        kw_args_init_flag = False

        arg_index = 0
        for arg in self.op_proto.op_args:
            arg_name = arg.arg_name

            if is_tensor_api and self._is_input_arg(arg_name, op_name):
                continue

            single_arg = ''
            if not first_arg:
                single_arg = ', '

            arg_handler = arg.arg_handler
            if arg_handler:
                if arg_handler in K.ARG_HANDLER_MAP:
                    arg_dtype = K.ARG_HANDLER_MAP[arg_handler]
                else:
                    raise ValueError(
                        f"Generate failed. Check if {arg_handler} is registered in TensorFuncRegCppGenerator.")
            else:
                arg_dtype = arg.arg_dtype
                for cast_type in arg.type_cast:
                    arg_dtype += f'|{cast_type}'

            # handle varargs params
            if varargs and arg_name in varargs and arg_index == 0:
                single_arg += f"{arg_dtype} *{arg_name}"
            else:
                single_arg += f"{arg_dtype} {arg_name}"

            if arg.as_init_arg:
                single_arg += f"={arg.default}"

            # handle keyword-only params
            if kw_only_args and not kw_args_init_flag and arg_name == kw_only_args[0]:
                single_arg = ("*, " if first_arg else ", *") + single_arg
                kw_args_init_flag = True

            args_str += single_arg
            first_arg = False
            arg_index += 1

        return args_str + ')"'

    def get_arg_handler_processor(self, func_name, op_proto, *, is_tensor_api):
        """
        Generates argument handler processing code for the given function prototype.

        Args:
            func_name (str): The name of the function.
            op_proto (OpProto): Operator prototype instance to generate argument processing for.

        Returns:
            str: Generated argument handler processing code.
        """
        tensor_arg_handler_prt_template = Template(
            "arg_list[${idx}] = "
            "(*pynative::${func_str}(\"${func_name}\", \"${op_arg_name}\", arg_list[${idx}]))->value();\n"
        )
        function_arg_handler_prt_template = Template(
            "arg_list[${idx}] = "
            "(*${func_str}(\"${func_name}\", \"${op_arg_name}\", arg_list[${idx}]))->value();\n"
        )
        arg_handler_prt_template = (
            tensor_arg_handler_prt_template) if is_tensor_api else function_arg_handler_prt_template

        arg_handler_template = Template(
            "arg_list[${idx}] = "
            "pynative::${func_str}(\"${func_name}\", \"${op_arg_name}\", arg_list[${idx}]);\n"
        )
        arg_handler_optional_template = Template(
            'if (!py::isinstance<py::none>(arg_list[${idx}])) {\n'
            '  ${arg_handler_str}\n'
            '}\n'
        )

        arg_handler_processor = []
        op_args = op_proto.op_args
        for idx, op_arg in enumerate(op_args):
            arg_handler = op_arg.arg_handler
            func_str = ''.join(word.capitalize() for word in arg_handler.split('_'))
            if arg_handler:
                op_arg_name = op_arg.arg_name
                if func_str in ("StrToEnum", "DtypeToTypeId"):
                    arg_handler_str = arg_handler_prt_template.replace(func_str=func_str,
                                                                       func_name=func_name,
                                                                       op_arg_name=op_arg_name,
                                                                       idx=idx)
                else:
                    arg_handler_str = arg_handler_template.replace(func_str=func_str,
                                                                   func_name=func_name,
                                                                   op_arg_name=op_arg_name,
                                                                   idx=idx)

                if op_arg.default == "None":
                    arg_handler_str = arg_handler_optional_template.replace(idx=idx,
                                                                            arg_handler_str=arg_handler_str)
                arg_handler_processor.append(arg_handler_str)

        return arg_handler_processor
