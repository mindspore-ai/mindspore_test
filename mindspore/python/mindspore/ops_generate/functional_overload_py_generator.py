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
Module for generating C++ header files with operator name definitions.

This module defines the `OpsNameHGenerator` class, which produces C++ code to define
constants for operator names based on given prototypes.
"""

import os

import gen_constants as K
import gen_utils
import pyboost_utils
import template
from template import Template

from base_generator import BaseGenerator


class FunctionalOverloadPyGenerator(BaseGenerator):
    """
    Class for generating C++ header files containing operator name constants.
    """

    def __init__(self):
        """
        Initializes the OpsNameHGenerator instance.
        """
        self.FUNCTIONAL_OVERLOAD_PY_TEMPLATE = template.FUNCTIONAL_OVERLOAD_PY_TEMPLATE

        self.import_mint_template = Template("from mindspore._c_expression import ${cpp_func_name}Functional_\n")
        self.mint_init_template = Template("${mint_func_name} = ${cpp_func_name}Functional_()\n")

    def generate(self, work_path, mint_func_protos_data, alias_api_mapping):
        """
        Generates python code for operator names and saves it to a header file.

        Args:
            mint_func_protos_data (dict): A dictionary mapping mint API names to their prototype data.
        """
        import_mint_list, mint_init_list, add_to_all_list = [], [], []
        for mint_api_name, _ in mint_func_protos_data.items():
            cpp_func_name = pyboost_utils.format_func_api_name(mint_api_name)
            import_mint_list.append(self.import_mint_template.replace(cpp_func_name=cpp_func_name))
            mint_init_list.append(self.mint_init_template.replace(mint_func_name=mint_api_name,
                                                                  cpp_func_name=cpp_func_name))
            add_to_all_list.append(f'"{mint_api_name}",\n')
            if mint_api_name in alias_api_mapping:
                for alias_api_name in alias_api_mapping[mint_api_name]:
                    mint_init_list.append(self.mint_init_template.replace(mint_func_name=alias_api_name,
                                                                          cpp_func_name=cpp_func_name))
                    add_to_all_list.append(f'"{alias_api_name}",\n')

        func_overload_py_file = self.FUNCTIONAL_OVERLOAD_PY_TEMPLATE.replace(import_mint_list=import_mint_list,
                                                                             mint_init_list=mint_init_list,
                                                                             add_to_all_list=add_to_all_list)
        save_path = os.path.join(work_path, K.MS_MINT_FUNC_PATH)
        file_name = "functional_overload.py"
        gen_utils.save_file(save_path, file_name, func_overload_py_file)
