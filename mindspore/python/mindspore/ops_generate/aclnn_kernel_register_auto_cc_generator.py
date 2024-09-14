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
Generates C++ registration code for ACL NN kernels based on operator prototypes.
"""

import os

import gen_constants as K
import gen_utils
import pyboost_utils

import template

from base_generator import BaseGenerator
from gen_aclnn_implement import gen_aclnn_kernel


class AclnnKernelRegisterAutoCcGenerator(BaseGenerator):
    """Generates ACL NN kernel registration code for Ascend devices."""

    def __init__(self):
        self.aclnn_reg_code_template = template.Template(K.ACLNN_REG_CODE)

    def generate(self, work_path, op_protos):
        """
        Generates registration code for ACL NN kernels and saves it to a file.

        Args:
            work_path (str): The directory to save the generated file.
            op_protos (list): List of operator prototypes.

        Returns:
            None
        """
        aclnn_reg_code = []
        for op_proto in op_protos:
            if not op_proto.op_dispatch or not op_proto.op_dispatch.enable:
                continue
            if op_proto.op_dispatch.ascend != 'default':  # KernelMod is provided by yaml, don't auto generate it.
                continue
            _, _, none_tensor_exist = pyboost_utils.get_dtypes(op_proto)
            if none_tensor_exist:
                # gen operator aclnn kernel c++ files
                gen_aclnn_kernel(op_proto, auto=True)
                continue

            class_name = op_proto.op_class.name
            inputs_outputs_num = len(op_proto.op_args) + len(op_proto.op_returns)
            aclnn_name = pyboost_utils.AclnnUtils.get_aclnn_interface(class_name)
            aclnn_reg_code.append(
                f"MS_ACLNN_COMMON_KERNEL_FACTORY_REG({class_name}, {aclnn_name}, {inputs_outputs_num});\n")

        reg_code = self.aclnn_reg_code_template.replace(ops_gen_kernel_path=K.MS_OPS_KERNEL_PATH,
                                                        aclnn_reg_code=aclnn_reg_code)
        res_str = template.CC_LICENSE_STR + reg_code

        save_path = os.path.join(work_path, f"{K.MS_OPS_KERNEL_PATH}/ascend/opapi/")
        file_name = "aclnn_kernel_register_auto.cc"
        gen_utils.save_file(save_path, file_name, res_str)
