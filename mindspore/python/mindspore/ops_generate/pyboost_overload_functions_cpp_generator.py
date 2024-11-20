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
This module defines the PyboostFunctionsGenerator class for generating C++ functions for PyBoost operations.

The generator processes operator prototypes and constructs the necessary function definitions, including
conversions for optional parameters and tensor arguments. It generates the registration code and includes
the necessary header files for the generated functions.
"""

import os
import template
from template import Template
import gen_constants as K
import pyboost_utils
import op_api_proto
from gen_utils import save_file
from op_proto import OpProto
from base_generator import BaseGenerator


class PyboostOverloadFunctionsGenerator(BaseGenerator):
    """
    Generates PyBoost overload functions cpp code based on operator prototypes.

    This class processes operator prototypes (`op_protos`) to create the necessary C++ function definitions for
    PyBoost operations. It constructs function bodies, handles optional value conversions, and generates
    registration code and header inclusions.
    """

    def __init__(self):
        self.PYBOOST_OVERLOAD_FUNCTIONS_TEMPLATE = template.PYBOOST_OVERLOAD_FUNCTIONS_CC_TEMPLATE
        self.PYBOOST_MINT_CLASS_DEF = template.PYBOOST_MINT_CLASS_DEF
        self.PYBOOST_OVERLOAD_MINT_CLASS_DEF = template.PYBOOST_OVERLOAD_MINT_CLASS_DEF
        self.TENSOR_FUNC_UT_BODY = template.TENSOR_FUNC_UT_BODY
        self.TENSOR_FUNC_UT_OVERLOAD_BODY = template.TENSOR_FUNC_UT_OVERLOAD_BODY

        self.single_case_template = Template(
            'case ${case_id}:\n'
            '  ${device_dispatcher}\n'
            '  break;\n'
        )
        self.device_dispatcher_template = Template(
            'if (backend == kAscendDevice || backend == kDavinciDevice) {\n'
            '  ${ascend_dispatcher}\n'
            '} else if (backend == kCPUDevice) {\n'
            '  ${cpu_dispatcher}\n'
            '} else if (backend == kGPUDevice) {\n'
            '  ${gpu_dispatcher}\n'
            '} else {\n'
            '  MS_LOG(ERROR) << "Device target is not supported!";\n'
            '  return py::none();\n'
            '}'
        )
        self.pyboost_return_template = Template(
            '${arg_handler_processor}\n'
            'MS_LOG(INFO) << "Call Tensor${class_name}";\n'
            'return mindspore::tensor::ToPython(tensor::Tensor${class_name}Register::GetOp()(arg_list));\n'
        )
        self.callback_python_template = Template(
            'MS_LOG(INFO) << "Callback python method: ${py_method}";\n'
            'py::function fn = python_adapter::GetPyFn(\"mindspore.ops.tensor_method\", \"${py_method}\");\n'
            'py::object res = fn(self, *args, **kwargs);\n'
            'return res;\n'
        )
        self.arg_handler_prt_template = Template(
            "arg_list[${idx}] = "
            "(*${func_str}(\"${func_name}\", \"${op_arg_name}\", arg_list[${idx}]))->value();\n"
        )
        self.arg_handler_template = Template(
            "arg_list[${idx}] = "
            "${func_str}(\"${func_name}\", \"${op_arg_name}\", arg_list[${idx}]);\n"
        )
        self.arg_handler_optional_template = Template(
            'if (!py::isinstance<py::none>(arg_list[${idx}])) {\n'
            '  ${arg_handler_str}\n'
            '}\n'
        )
        self.pybind_register_template = Template(
            '(void)py::class_ <${cpp_func_name}Functional, Functional, std::shared_ptr<${cpp_func_name}Functional>>(\n'
            '*m, "${mint_func_name}Functional_")\n'
            '.def(py::init<>())\n'
            '.def("__call__", &${cpp_func_name}Functional::Call, "Call ${mint_func_name} functional.");\n'
        )
        self.callback_python_in_ut_template = Template(
            'MS_LOG(INFO) << "Callback python method in UT: ${py_method}";\n'
            'fn = python_adapter::GetPyFn(\"mindspore.ops.tensor_method\", \"${py_method}\");\n'
            'res = fn(self, *args, **kwargs);\n'
            'break;\n'
        )
        self.single_case_in_ut_template = Template(
            'case ${case_id}:\n'
            '  ${device_dispatcher}\n'
        )
        # The format of arg_handler_map is {arg_handler_name : list of supported types}.
        # The first one of type list is the target dtype. Types corresponds to type_str_map.
        self.arg_handler_map = {"to_2d_paddings": "int|tuple[int]|list[int]",
                                "dtype_to_type_id": "type",
                                "to_kernel_size": "int|tuple[int]|list[int]",
                                "to_strides": "int|tuple[int]|list[int]",
                                "str_to_enum": "str",
                                "to_pair": "int|tuple[int]|list[int]|float",
                                "to_dilations": "tuple[int]|list[int]|int",
                                "to_output_padding": "int|tuple[int]|list[int]",
                                "to_rates": "int|tuple[int]|list[int]"}
        self.input_args_name = {"input", "x", "input_x"}

    def generate(self, work_path, op_protos, mint_func_protos_data, alias_func_mapping):
        """
        Generates the C++ PyBoost functions and writes them to the specified files.

        This method processes a list of operator prototypes (`op_protos`), extracting necessary information
        such as operator names, arguments, and conversion types. It constructs the function definitions, includes,
        and registration code. The generated content is saved to the specified path as a C++ source file.

        Args:
            work_path (str): The file path where the generated files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators to be processed.
            mint_func_protos_data(dict): A dict of tensor prototypes containing device-related information.

        Returns:
            None
        """

        mint_classes_def_list = []

        _, single_mint_func_data, overload_mint_func_data = op_api_proto.categorize_func_data(mint_func_protos_data)

        mint_classes_def_list.extend(self._get_single_func_call_body_list(single_mint_func_data))
        mint_classes_def_list.extend(self._get_overload_func_call_body_list(overload_mint_func_data))
        mint_classes_reg_list = self._get_mint_func_reg_list(mint_func_protos_data, alias_func_mapping)

        pyboost_overload_file_str = (
            self.PYBOOST_OVERLOAD_FUNCTIONS_TEMPLATE.replace(mint_func_classes_def=mint_classes_def_list,
                                                             pybind_register_code=mint_classes_reg_list))
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_FUNC_GEN_PATH)
        file_name = "pyboost_overload_functions.cc"
        save_file(save_path, file_name, pyboost_overload_file_str)

    def _get_single_func_call_body_list(self, single_op_func_data):
        """
        Generates the list of call body strings for single operation functions.

        Args:
            single_op_func_data (dict): Dictionary of tensor function prototypes with only one definition.

        Returns:
            list: Updated str list for generating C++ function call bodies.
        """
        func_call_body_list = []
        for _, func_proto in single_op_func_data.items():
            func_name = func_proto.func_name
            class_name = func_proto.op_proto.op_class.name
            device_dispatcher_str = self._get_device_dispatchers_str(func_proto)
            signature_str = self._generate_single_signature_str(func_proto.op_proto)
            op_args = func_proto.op_proto.op_args
            max_size = len(op_args)
            self_index = [i for i in range(len(op_args)) if op_args[i].arg_name in self.input_args_name]
            if len(self_index) != 1:
                raise ValueError(
                    f'There must be only one field named \'input\'. But got {len(self_index)} in {func_name}')
            ut_body = self.TENSOR_FUNC_UT_BODY.replace(py_method=func_proto.py_method)
            func_call_body_list.append(self.PYBOOST_MINT_CLASS_DEF.replace(
                class_name=class_name,
                func_name=func_name,
                device_dispatcher=device_dispatcher_str,
                signatures=signature_str,
                max_args=max_size,
                self_index=self_index,
                ut_body=ut_body))
        return func_call_body_list

    def _get_overload_func_call_body_list(self, overload_op_func_data):
        """
        Generates the list of call body strings for overloaded operation functions.

        Args:
            overload_op_func_data (dict): Dictionary of tensor function prototypes with overloaded definitions.

        Returns:
            list: Updated str list for generating C++ function call bodies.
        """
        func_call_body_list = []
        for func_api_name, func_protos in overload_op_func_data.items():
            func_call_body_list.append(self._get_overload_func_call_str(func_api_name, func_protos))
        return func_call_body_list

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
        dispatch_cases = self._get_dispatch_cases(func_protos)
        ut_dispatch_cases = self._get_ut_dispatch_cases(func_protos)
        ut_overload_body = self.TENSOR_FUNC_UT_OVERLOAD_BODY.replace(ut_dispatch_cases=ut_dispatch_cases)

        max_size = 0
        self_index = 0
        for tensor_proto in func_protos:
            op_proto = tensor_proto.op_proto
            op_args = op_proto.op_args
            max_size = max(len(op_args), max_size)
            self_index = [i for i in range(len(op_args)) if op_args[i].arg_name in self.input_args_name]
            if len(self_index) != 1:
                raise ValueError(
                    f'There must be only one field named \'input\'. But got {len(self_index)} in {func_api_name}')
        formatted_class_name = pyboost_utils.format_func_api_name(func_api_name)
        overload_func_call_str = self.PYBOOST_OVERLOAD_MINT_CLASS_DEF.replace(class_name=formatted_class_name,
                                                                              func_name=func_api_name,
                                                                              signatures=signatures_str,
                                                                              dispatch_cases=dispatch_cases,
                                                                              max_args=max_size,
                                                                              self_index=self_index,
                                                                              ut_overload_body=ut_overload_body)
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
            if arg.arg_name in self.input_args_name:
                continue
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

    def _get_ut_dispatch_cases(self, func_protos):
        """
        Generates C++ switch-case statements for dispatching tensor function calls.

        Args:
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated switch-case dispatch statements.
        """
        dispatch_cases_str = ''
        for idx, func_proto in enumerate(func_protos):
            device_dispatcher_str = self.callback_python_in_ut_template.replace(py_method=func_proto.py_method)
            dispatch_cases_str += self.single_case_in_ut_template.replace(case_id=idx,
                                                                          device_dispatcher=device_dispatcher_str)
        dispatch_cases_str += 'default:\n'
        dispatch_cases_str += '  res = py::none();'
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
        arg_handler_processor = []
        op_proto = func_proto.op_proto
        op_args = op_proto.op_args
        for idx, op_arg in enumerate(op_args):
            arg_handler = op_arg.arg_handler
            func_str = ''.join(word.capitalize() for word in arg_handler.split('_'))
            if arg_handler:
                func_name = func_proto.func_name
                op_arg_name = op_arg.arg_name
                if func_str in ("StrToEnum", "DtypeToTypeId"):
                    arg_handler_str = self.arg_handler_prt_template.replace(func_str=func_str,
                                                                            func_name=func_name,
                                                                            op_arg_name=op_arg_name,
                                                                            idx=idx)
                else:
                    arg_handler_str = self.arg_handler_template.replace(func_str=func_str,
                                                                        func_name=func_name,
                                                                        op_arg_name=op_arg_name,
                                                                        idx=idx)

                if op_arg.default == "None":
                    arg_handler_str = self.arg_handler_optional_template.replace(idx=idx,
                                                                                 arg_handler_str=arg_handler_str)
                arg_handler_processor.append(arg_handler_str)

        return arg_handler_processor

    def _get_mint_func_reg_list(self, mint_func_protos_data, alias_func_mapping):
        """
        Generates the list of pybind definition strings for mint functions.

        Args:
            mint_func_protos_data (dict): Dictionary of tensor function prototypes with only one definition.
            alias_func_mapping (dict): A dict of tensor prototypes containing device-related information.

        Returns:
            list: list of strs for generating pybind definitions of mint functions' API.
        """
        mint_func_reg_list = []
        for func_name in mint_func_protos_data.keys():
            formatted_class_name = pyboost_utils.format_func_api_name(func_name)
            mint_func_reg_list.append(self.pybind_register_template.replace(mint_func_name=formatted_class_name,
                                                                            cpp_func_name=formatted_class_name))
        return mint_func_reg_list
