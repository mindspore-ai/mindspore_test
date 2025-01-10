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
This module defines the `PyboostFunctionsHeaderGenerator` class, which is responsible for generating
the header file (`pyboost_functions.h`) for Pyboost function declarations.

The class uses templates and operation prototypes to create function declarations based on the
operation's primitive and arguments. The generated file is saved to the specified path.
"""

import os

import template

from template import Template
import gen_constants as K
from gen_utils import save_file
from op_template_parser import OpTemplateParser
from base_generator import BaseGenerator


class PyboostFunctionsHeaderGenerator(BaseGenerator):
    """
    A class to generate the `pyboost_functions.h` header file, which contains Pyboost function declarations.
    """

    def __init__(self):
        """Initializes the PyboostFunctionsHeaderGenerator with the necessary templates."""
        self.PYBOOST_FUNCTION_HEADER_TEMPLATE = template.PYBOOST_FUNCTION_HEADER_TEMPLATE

        self.pyboost_func_template = Template(
            'py::object ME_EXPORT ${func_name}_Base(const PrimitivePtr &prim, const py::list &args);'
        )

    def generate(self, work_path, op_protos):
        """
        Generates the Pyboost function header file (`pyboost_functions.h`).

        Args:
            work_path (str): The directory where the generated file will be saved.
            op_protos (list): A list of operation prototypes to parse and convert into Pyboost function declarations.

        Returns:
            None: The method writes the generated header file to the specified directory.
        """
        func_list = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or not op_proto.op_dispatch.enable:
                continue
            op_parser = OpTemplateParser(op_proto)
            op_pyboost_func_name = op_parser.get_pyboost_func_name()
            func_list.append(self.pyboost_func_template.replace(func_name=op_pyboost_func_name))
        pyboost_func_h_str = self.PYBOOST_FUNCTION_HEADER_TEMPLATE.replace(prim_func_list=func_list)
        save_path = os.path.join(work_path, K.PIPELINE_PYBOOST_FUNC_GEN_PATH)
        file_name = "pyboost_functions.h"
        save_file(save_path, file_name, pyboost_func_h_str)
