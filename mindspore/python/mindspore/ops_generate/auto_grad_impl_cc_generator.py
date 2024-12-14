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
This module provides a generator class for creating C++ implementation files for AutoGrad functionality.
"""

import os

import template
from template import Template
import gen_constants as K
from gen_utils import save_file
from base_generator import BaseGenerator
from pyboost_utils import is_optional_param, get_input_dtype, is_op_multi_output


class AutoGradImplGenerator(BaseGenerator):
    """
    Generates C++ implementation files for the AutoGrad functionality based on operator prototypes.
    """

    def __init__(self):
        """
        Initialize the AutoGrad implementation generator with templates for code generation.
        """
        self.AUTO_GRAD_IMPL_CC_TEMPLATE = template.AUTO_GRAD_IMPL_CC_TEMPLATE
        self.DO_GRAD_FUNCTION_BODY_TEMPLATE = template.DO_GRAD_FUNCTION_BODY_TEMPLATE
        self.auto_grad_reg_template = Template("const_cast<kernel::pyboost::${class_name}GradFunc&>(" + \
                                               "kernel::pyboost::AutoGradFactory::Get()." + \
                                               "ops_auto_grad_registers().${class_name}GradFuncObj) = " + \
                                               "kernel::pyboost::${class_name}GradFunc(DoGrad${class_name});")
        self.do_grad_op_args_with_type = Template(
            "const kernel::pyboost::OpPtr &op, ${input_args_with_type}"
        )

    def generate(self, work_path, op_protos):
        """
        Generate the AutoGrad implementation file.

        Args:
            work_path (str): The directory where the generated file should be saved.
            op_protos (list): A list of operator prototypes used to generate the implementation.
        """
        auto_grad_reg_list = []
        do_grad_op_list = []
        for op_proto in op_protos:
            if op_proto.op_dispatch is None or op_proto.op_dispatch.is_comm_op:
                continue
            auto_grad_reg_list.append(self.auto_grad_reg_template.replace(class_name=op_proto.op_class.name))
            do_grad_op_list.append(self._get_single_do_grad_op(op_proto))
        pyboost_func_h_str = self.AUTO_GRAD_IMPL_CC_TEMPLATE.replace(do_grad_op=do_grad_op_list,
                                                                     auto_grad_reg=auto_grad_reg_list)
        save_path = os.path.join(work_path, K.PYBOOST_AUTO_GRAD_FUNC_GEN_PATH)
        file_name = "auto_grad_impl.cc"
        save_file(save_path, file_name, pyboost_func_h_str)

    def _get_single_do_grad_op(self, op_proto):
        """
        Generate the DoGrad function for a single operator prototype.

        Args:
            op_proto: The operator prototype for which the DoGrad function is generated.

        Returns:
            str: The generated DoGrad function string.
        """
        input_args_str = self._get_input_args(op_proto, False, False)
        input_args_with_optional_str = self._get_input_args(op_proto, False, True)
        input_args_with_type_str = self._get_input_args(op_proto, True, False)
        multi_output_str = 'Multi' if is_op_multi_output(op_proto.op_returns) else ''
        view_arg_str = self._get_view_str(op_proto.op_view, input_args_str)
        grad_args_with_type_str = self.do_grad_op_args_with_type.replace(input_args_with_type=input_args_with_type_str)
        op_def_name_str = "g" + op_proto.op_class.name
        bprop_expander = "true" if op_proto.bprop_expander else "false"
        return self.DO_GRAD_FUNCTION_BODY_TEMPLATE.replace(class_name=op_proto.op_class.name,
                                                           grad_args_with_type=grad_args_with_type_str,
                                                           grad_input_args=input_args_str,
                                                           grad_input_args_with_optional=input_args_with_optional_str,
                                                           is_multi=multi_output_str,
                                                           view_arg=view_arg_str,
                                                           op_def_name=op_def_name_str,
                                                           bprop_expander=bprop_expander)

    def _get_input_args(self, op_proto, has_type, with_optional):
        """
        Get the input arguments for the DoGrad function.

        Args:
            op_proto: The operator prototype.
            has_type (bool): Whether to include type information for the arguments.

        Returns:
            list: A list of input arguments for the DoGrad function.
        """
        args_list = []
        for op_arg in op_proto.op_args:
            input_dtype = get_input_dtype(op_arg.arg_dtype, is_optional_param(op_arg))
            if has_type:
                args_list.append(f"const {input_dtype} &{op_arg.arg_name}_tensor")
            else:
                if not with_optional and is_optional_param(op_arg):
                    args_list.append(f"OptionalToValue({op_arg.arg_name}_tensor)")
                else:
                    args_list.append(f"{op_arg.arg_name}_tensor")
        return args_list

    def _get_view_str(self, is_view_op: bool, grad_args: list):
        """
        Get the view argument string for a DoGrad function.

        Args:
            is_view_op (bool): Whether the operator is a view operator.
            grad_args (list): A list of gradient arguments.

        Returns:
            str: The view argument string.
        """
        view_arg_str = ''
        for i, grad_arg in enumerate(grad_args):
            if is_view_op and i == 0:
                view_arg_str = ", " + grad_arg
                break
        return view_arg_str
