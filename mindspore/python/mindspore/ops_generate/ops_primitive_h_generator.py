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
Module for generating C++ header files for operator primitives.

This module defines the `OpsPrimitiveHGenerator` class, which creates C++ header files
containing definitions for operator primitives based on provided operator prototypes.
"""

import os

import gen_constants as K
import gen_utils
import pyboost_utils

# refactored
import template

from base_generator import BaseGenerator


class OpsPrimitiveHGenerator(BaseGenerator):
    """
    This class generates the header file for operator primitives.
    """

    def __init__(self):
        """
        Initializes the generator with templates for operator primitive definitions.
        """
        self.op_prim_op_def_template = template.Template(K.OP_PRIM_OP_DEF)
        self.op_def_template = template.Template(
            "GVAR_DEF(PrimitivePtr, kPrim${k_name_op}, std::make_shared<Primitive>(ops::kName${k_name_op}))\n")

    def generate(self, work_path, op_protos):
        """
        Generates the header file content for operator primitives and saves it.

        Args:
            work_path (str): The directory to save the generated files.
            op_protos (list): A list of operator prototypes.

        Returns:
            None

        The method generates the content of the header file for each operator primitive
        defined in the 'op_protos' list and saves it to the specified work path.
        """
        ops_prim_gen_list = []
        for op_proto in op_protos:
            k_name_op = pyboost_utils.get_op_name(op_proto.op_name, op_proto.op_class.name)
            ops_prim_gen_list.append(self.op_def_template.replace(k_name_op=k_name_op))

        op_prim_op_def = self.op_prim_op_def_template.replace(auto_gen_path=K.MS_OP_DEF_AUTO_GENERATE_PATH,
                                                              ops_prim_gen=ops_prim_gen_list)

        res_str = template.CC_LICENSE_STR + op_prim_op_def

        save_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_PATH)
        file_name = "gen_ops_primitive.h"
        gen_utils.save_file(save_path, file_name, res_str)
