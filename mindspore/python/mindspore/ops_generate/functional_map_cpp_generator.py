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
Generates C++ functional map header files for graph mode.
"""

import os
import template
import pyboost_utils
import gen_constants as K
from gen_utils import save_file
from base_generator import BaseGenerator


class FunctionalMapCppGenerator(BaseGenerator):
    """
    Generates C++ functional map header files for graph mode.
    """

    def __init__(self):
        """
        Initializes the generator with templates for the functional map.
        """
        self.function_map_cc_template = template.FUNCTIONAL_MAP_CC_TEMPLATE
        self.function_map_h_template = template.FUNCTIONAL_MAP_H_TEMPLATE
        self.class_to_method_template = template.Template("{\"${class_name}\", \"${method_name}\"}")
        self.functional_map_template = template.Template("{\"${func_api_name}\", {${class_to_method_str}}},")
        self.k_prim_op_template = template.Template("prim::kPrim${camel_op_name}")
        self.deprecated_method_decl_template = template.Template(
            "auto ${dep_op_name} = std::make_shared<prim::DeprecatedTensorMethod>(\"${dep_op_name}\", \"${op_name}\");")
        self.functional_method_map_template = template.Template("{\"${op_name}\", {${sort_func_method_list_str}}},")

        self.arg_handler_map = {"to_2d_paddings": ["tuple[int]", "list[int]", "int"],
                                "dtype_to_type_id": ["int", "type"],
                                "to_kernel_size": ["tuple[int]", "list[int]", "int"],
                                "to_strides": ["tuple[int]", "list[int]", "int"],
                                "str_to_enum": ["int", "str"],
                                "to_pair": ["tuple[int]", "list[int]", "int", "float"],
                                "to_dilations": ["tuple[int]", "list[int]", "int"],
                                "to_output_padding": ["tuple[int]", "list[int]", "int"],
                                "to_rates": ["tuple[int]", "list[int]", "int"]}
        self.prompt_type_map = {"int": "int",
                                "float": "float",
                                "str": "str",
                                "bool": "bool",
                                "number": "number",
                                "tensor": "Tensor",
                                "type": "mstype",
                                "None": "None"}

    def generate(self, work_path, func_protos_data, alias_func_mapping):
        """
        Generates the functional map header file.

        Args:
            work_path (str): The directory path to save the generated file.
            func_protos_data (dict): A dictionary mapping function API names to their prototype data.
            alias_func_mapping (dict): A dictionary mapping function name to its alias function names.

        Returns:
            None
        """
        dep_method_decl_list = self._get_dep_method_decl_list(func_protos_data)
        functional_method_map_list = self._get_functional_method_map(func_protos_data, alias_func_mapping)
        functional_map_list = self._get_functional_map_list(func_protos_data, alias_func_mapping)
        funcs_sig_map_list = self._get_func_sigs_list(func_protos_data, alias_func_mapping)
        functional_map_cc_code = self.function_map_cc_template.replace(deprecated_method_decl=dep_method_decl_list,
                                                                       functional_method_map=functional_method_map_list,
                                                                       functional_map=functional_map_list,
                                                                       func_sigs_map=funcs_sig_map_list)
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_FUNC_GEN_PATH)
        save_file(save_path, "functional_map.cc", functional_map_cc_code)
        save_file(save_path, "functional_map.h", self.function_map_h_template.replace())

    def _get_functional_map_list(self, func_protos_data, alias_func_mapping):
        """
        Generates a list of functional map strings needed for generating the function_map.cc file.

        Args:
            func_protos_data (dict): A dictionary mapping function API names to a list of function prototype data.
            Each prototype contains class names and corresponding Python methods.
            alias_func_mapping (dict): A dictionary mapping function name to its alias function names.

        Returns: list: A list of functional map strings, where each string represents the mapping of a function API
        name to its associated class-to-method pairs.
        """

        def get_class_to_method_list(func_protos):
            """
            Get a str representation of a list of class names and corresponding Python methods.
            """
            class_to_method_list = []
            for func_proto in func_protos:
                class_name = func_proto.op_proto.op_class.name
                class_to_method_list.append(
                    self.class_to_method_template.replace(class_name=class_name,
                                                          method_name=func_proto.py_method))
            return class_to_method_list

        functional_map_list = []
        for func_api_name, func_protos in func_protos_data.items():
            class_to_method_list = get_class_to_method_list(func_protos)
            functional_map_list.append(
                self.functional_map_template.replace(func_api_name=func_api_name,
                                                     class_to_method_str=class_to_method_list))
            if func_api_name in alias_func_mapping:
                class_to_method_list = get_class_to_method_list(func_protos)
                functional_map_list.append(
                    self.functional_map_template.replace(func_api_name=alias_func_mapping[func_api_name],
                                                         class_to_method_str=class_to_method_list))
        return functional_map_list

    def _get_func_sigs_list(self, func_protos_data, alias_func_mapping):
        """
        Generates a list of function signatures for each function API name based on the provided prototype data.

        Args:
            func_protos_data (dict): A dictionary mapping function API names to their corresponding prototype data.
                                     Each prototype contains information necessary to generate function signatures.

        Returns: list: A list of function signature strings for each function API, which are generated based on the
        prototype data.
        """
        funcs_list = []
        for func_api_name, func_protos in func_protos_data.items():
            func_signatures = self._generate_func_signatures_str(func_api_name, func_protos)
            funcs_list.append(func_signatures)
            if func_api_name in alias_func_mapping:
                for alias_api_name in alias_func_mapping[func_api_name]:
                    func_signatures = self._generate_func_signatures_str(alias_api_name, func_protos)
                    funcs_list.append(func_signatures)

        return funcs_list

    def _generate_func_signatures_str(self, func_api_name, func_protos) -> str:
        """
        Generates function signatures as a string from the given prototypes.

        Args:
            func_api_name (str): The name of the API to generate signatures for.
            func_protos (list): List of TensorFuncProto objects representing the function prototypes.

        Returns:
            str: Generated function signatures string.
        """
        sig_str = '{' + f'\"{func_api_name}\",\n ' + '{'
        first_sig = True
        for tensor_proto in func_protos:
            op_proto = tensor_proto.op_proto
            if not first_sig:
                sig_str += ',\n'
            first_sig = False
            sig_str += self._generate_single_signature_str(func_api_name, op_proto)
        sig_str += '}\n},'
        return sig_str

    def _generate_single_signature_str(self, func_api_name, op_proto) -> str:
        """
        Generates a single function signature string for the given operation prototype.

        Args:
            func_api_name (str): The name of the API to generate signatures for.
            op_proto (OpProto): Operation prototype to generate the signature for.

        Returns:
            str: Generated function signature string.
        """
        args_str = f'"{func_api_name}('
        first_arg = True
        arg_valid_types = []
        for _, arg in enumerate(op_proto.op_args):
            arg_handler = arg.arg_handler
            if arg_handler != '':
                if arg_handler in self.arg_handler_map:
                    arg_valid_types.extend(self.arg_handler_map[arg_handler])
                else:
                    raise ValueError("Generate failed. Check if {} is registered in TensorFuncRegCppGenerator."
                                     .format(arg_handler))
            else:
                arg_valid_types.append(arg.arg_dtype)
                for cast_type in arg.type_cast:
                    arg_valid_types.append(cast_type)
            arg_name = arg.arg_name
            if arg.as_init_arg and str(arg.default) == 'None':
                arg_valid_types.append('None')
            arg_valid_types = self._parse_arg_type_list(func_api_name, arg_name, arg_valid_types)
            single_arg = f'{arg_name}=<' + ','.join(arg_valid_types) + '>'
            if first_arg:
                args_str += single_arg
                first_arg = False
            else:
                args_str += ", " + single_arg
            arg_valid_types = []
        return args_str + ')"'

    def _parse_arg_type_list(self, func_api_name, arg_name, arg_valid_types):
        """
        Parses a list of argument types and maps them to generalized types.

        Args:
            func_api_name (str): The name of the function API for which the argument types are being parsed.
            arg_name (str): The name of the argument whose valid types are being generalized.
            arg_valid_types (list): A list of valid argument types that need to be generalized.

        Returns:
            set: A set of generalized argument types (e.g., 'List', 'Tuple') based on the input types.

        Raises:
            ValueError: If an unrecognized or invalid type is encountered in the argument types list.
        """
        generalized_type_list = set()
        for arg_type in arg_valid_types:
            if arg_type in self.prompt_type_map:
                generalized_type_list.add(self.prompt_type_map[arg_type])
            elif "list" in arg_type:
                generalized_type_list.add('List')
            elif "tuple" in arg_type:
                generalized_type_list.add('Tuple')
            else:
                raise ValueError(f"Invalid type {arg_type} in api: {func_api_name} {arg_name}.")
        return generalized_type_list

    def _get_dep_method_decl_list(self, func_protos_data):
        """
        Extracts and generates declarations for deprecated methods from the provided function prototypes.

        Args:
            func_protos_data (dict): A dictionary where keys are function API names and values are lists
                of function prototypes. Each prototype contains an operation name.

        Returns:
            list: A list of strings, each representing a declaration for a deprecated method.
        """
        deprecated_method_decl_list = []
        for func_api_name, func_protos in func_protos_data.items():
            for func_proto in func_protos:
                op_name = func_proto.op_proto.op_name
                if not op_name.startswith("deprecated"):
                    continue

                deprecated_method_name = ''.join(word.capitalize() for word in op_name.split('_'))
                deprecated_method_decl_list.append(
                    self.deprecated_method_decl_template.replace(dep_op_name=deprecated_method_name,
                                                                 op_name=func_api_name))

        return deprecated_method_decl_list

    def _get_functional_method_map(self, func_protos_data, alias_func_mapping):
        """
            Generates a list of functional method maps from the provided function prototypes and alias mappings.

            Args:
                func_protos_data (dict): A dictionary where keys are function API names and values are lists
                    of function prototypes.
                alias_func_mapping (dict): A dictionary mapping function API names to their aliases.

            Returns:
                list: A list of strings, each representing a functional method map.
        """
        def get_sort_func_method_list(func_protos):
            """
            Retrieves a sorted list of operator primitives, prioritizing deprecated operators.
            """
            func_method_list = []
            for func_proto in func_protos:
                k_op_name = pyboost_utils.get_op_name(func_proto.op_proto.op_name, func_proto.op_proto.op_class.name)
                if k_op_name.startswith("Deprecated"):
                    func_method_list.append(k_op_name)
                else:
                    func_method_list.append(self.k_prim_op_template.replace(camel_op_name=k_op_name))

            func_method_list.sort(key=lambda x: x.startswith("Deprecated"), reverse=True)
            return func_method_list

        deprecated_method_decl_list = []
        for func_api_name, func_protos in func_protos_data.items():
            sort_func_method_list = get_sort_func_method_list(func_protos)
            deprecated_method_decl_list.append(
                self.functional_method_map_template.replace(op_name=func_api_name,
                                                            sort_func_method_list_str=sort_func_method_list))

            if func_api_name in alias_func_mapping:
                deprecated_method_decl_list.append(
                    self.functional_method_map_template.replace(op_name=alias_func_mapping[func_api_name],
                                                                sort_func_method_list_str=sort_func_method_list))

        return deprecated_method_decl_list
