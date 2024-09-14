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
Generates C++ header and source files for lite operations based on YAML configurations.
"""

import os

import gen_constants as K
import gen_utils
import pyboost_utils

# refactored
import template

from base_generator import BaseGenerator


class LiteOpsHGenerator(BaseGenerator):
    """
    This class is responsible for generating the header file for lite operations.
    """

    def __init__(self):
        """
        Initializes the generator with the necessary templates for generating C++ header files.
        """
        self.lite_ops_h_template = template.Template(K.LITE_OPS_H)
        self.lite_ops_class_template = template.op_cc_template
        self.arg_prim_init_template = template.Template("\n"
                                                        "  void set_${arg_name}(const ${dtype} &${arg_name});\n"
                                                        "  ${dtype} get_${arg_name}() const;")

    def generate(self, work_path, op_protos):
        """
        Generates the header file content for lite operations and saves it to the specified path.

        Args:
            work_path (str): The directory where the generated files will be saved.
            op_protos (list): A list of operator prototypes containing information about the operators.

        Returns:
            None

        """
        lite_ops_h_code_list = []
        for op_proto in op_protos:
            op_name = pyboost_utils.get_op_name(op_proto.op_name, op_proto.op_class.name)
            op_args = op_proto.op_args
            arg_prim_init_str = ""
            for op_arg in op_args:
                if not op_arg.is_prim_init:
                    continue

                arg_name = op_arg.arg_name
                dtype = trans_dtype_for_lite(op_arg.arg_dtype)
                arg_prim_init_str += self.arg_prim_init_template.replace(arg_name=arg_name, dtype=dtype)

            temp = self.lite_ops_class_template.replace(op_name=op_name, arg_prim_init_list=arg_prim_init_str)
            lite_ops_h_code_list.append(temp)

        lite_ops_h = self.lite_ops_h_template.replace(auto_gen_path=K.OP_DEF_AUTO_GENERATE_PATH,
                                                      ops_namespace_body=lite_ops_h_code_list)

        res_str = template.CC_LICENSE_STR + lite_ops_h
        save_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_PATH)
        file_name = "gen_lite_ops.h"
        gen_utils.save_file(save_path, file_name, res_str)


class LiteOpsCcGenerator(BaseGenerator):
    """
    This class is responsible for generating the source file for lite operations.
    """

    def __init__(self):
        """
        Initializes the generator with the necessary templates for generating C++ source files.
        """
        self.lite_ops_cc_template = template.Template(K.LITE_OPS_CC)
        self.op_template = template.op_template
        self.register_primitive_c_template = template.Template("REGISTER_PRIMITIVE_C(kName${op_name}, ${op_name});\n"
                                                               "MIND_API_OPERATOR_IMPL(${op_name}, BaseOperator);\n\n")

    def generate(self, work_path, op_protos):
        """
        Generates the source file content for lite operations and saves it to the specified path.

        Args:
            work_path (str): The directory where the generated files will be saved.
            op_protos (list): A list of operation prototypes to generate content for.

        Returns:
            None
        """
        lite_ops_cc_gen_list = []
        for op_proto in op_protos:
            arg_prim_init_str = ""
            op_name = pyboost_utils.get_op_name(op_proto.op_name, op_proto.op_class.name)
            op_args = op_proto.op_args
            for op_arg in op_args:
                if not op_arg.is_prim_init:
                    continue

                arg_name = op_arg.arg_name
                dtype = trans_dtype_for_lite(op_arg.arg_dtype)
                arg_prim_init_str += self.op_template.replace(op_name=op_name, arg_name=arg_name, dtype=dtype)

            self.register_primitive_c_template.replace(op_name=op_name)
            lite_ops_cc_gen_list.append(arg_prim_init_str + self.register_primitive_c_template.replace(op_name=op_name))

        lite_ops_cc = self.lite_ops_cc_template.replace(auto_gen_path=K.OP_DEF_AUTO_GENERATE_PATH,
                                                        ops_namespace_body=lite_ops_cc_gen_list)

        res_str = template.CC_LICENSE_STR + lite_ops_cc
        save_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_PATH)
        file_name = "gen_lite_ops.cc"
        gen_utils.save_file(save_path, file_name, res_str)


def trans_dtype_for_lite(dtype):
    """
    Translate the data type for lite usage based on the argument information.

    Args:
        dtype (str): The original data type as a string.

    Returns:
        str: The translated data type suitable for lite usage.
    """
    type_mappings = {
        "str": "std::string",
        "tuple[str]": "std::vector<std::string>",
        "list[str]": "std::vector<std::string>",
        "tuple[int]": "std::vector<int64_t>",
        "list[int]": "std::vector<int64_t>",
        "tuple[float]": "std::vector<float>",
        "list[float]": "std::vector<float>",
        "tuple[bool]": "std::vector<bool>",
        "list[bool]": "std::vector<bool>",
        "int": "int64_t"
    }
    return type_mappings.get(dtype, dtype)
