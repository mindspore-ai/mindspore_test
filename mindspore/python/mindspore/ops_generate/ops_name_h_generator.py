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

# refactored
import template

from base_generator import BaseGenerator


class OpsNameHGenerator(BaseGenerator):
    """
    Class for generating C++ header files containing operator name constants.
    """

    def __init__(self):
        """
        Initializes the OpsNameHGenerator instance.
        """
        self.op_name_op_def_template = template.Template(K.OP_NAME_OP_DEF)
        self.op_def_body_template = template.Template("""constexpr auto kName${k_name_op} = "${k_name_op}";\n""")

    def generate(self, work_path, op_protos):
        """
        Generates C++ code for operator names and saves it to a header file.

        Args:
            work_path (str): The directory to save the generated files.
            op_protos (list): A list of operator prototypes.

        Returns:
            None
        """
        op_name_gen_list = []
        for op_proto in op_protos:
            k_name_op = pyboost_utils.get_op_name(op_proto.op_name, op_proto.op_class.name)
            op_name_gen_list.append(self.op_def_body_template.replace(k_name_op=k_name_op))

        op_name_code = self.op_name_op_def_template.replace(ops_namespace_body=op_name_gen_list)

        op_name_code = template.CC_LICENSE_STR + op_name_code

        save_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_PATH)
        file_name = "gen_ops_name.h"
        gen_utils.save_file(save_path, file_name, op_name_code)
