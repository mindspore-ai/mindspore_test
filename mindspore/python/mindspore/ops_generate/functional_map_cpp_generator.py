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
        self.function_map_h_template = template.FUNCTIONAL_MAP_CPP_TEMPLATE
        self.class_to_method_template = template.Template("{\"${class_name}\", \"${method_name}\"}")
        self.functional_map_template = template.Template("{\"${func_api_name}\", {${class_to_method_str}}},")

    def generate(self, work_path, func_protos_data):
        """
        Generates the functional map header file.

        Args:
            work_path (str): The directory path to save the generated file.
            func_protos_data (dict): A dictionary mapping function API names to their protocol data.

        Returns:
            None
        """
        functional_map_list = []
        for func_api_name, func_protos in func_protos_data.items():
            class_to_method_list = []
            for func_proto in func_protos:
                class_name = func_proto.op_proto.op_class.name
                class_to_method_list.append(
                    self.class_to_method_template.replace(class_name=class_name,
                                                          method_name=func_proto.py_method))

            functional_map_list.append(self.functional_map_template.replace(func_api_name=func_api_name,
                                                                            class_to_method_str=class_to_method_list))

        functional_map_h_code = self.function_map_h_template.replace(functional_map=functional_map_list)
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_FUNC_GEN_PATH)
        save_file(save_path, "functional_map.h", functional_map_h_code)
