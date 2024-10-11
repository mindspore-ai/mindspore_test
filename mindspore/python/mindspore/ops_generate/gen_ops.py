# Copyright 2023 Huawei Technologies Co., Ltd
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
Generate operator definition from ops.yaml
"""
import logging
import os
import shutil
import pathlib
from gen_utils import (check_change_and_replace_file, merge_files,
                       merge_files_append, safe_load_yaml)
from op_prim_py_generator import OpPrimPyGenerator
from op_def_py_generator import OpDefPyGenerator
from aclnn_kernel_register_auto_cc_generator import AclnnKernelRegisterAutoCcGenerator
from cpp_create_prim_instance_helper_generator import CppCreatePrimInstanceHelperGenerator
from ops_def_cc_generator import OpsDefCcGenerator
from ops_primitive_h_generator import OpsPrimitiveHGenerator
from lite_ops_cpp_generator import LiteOpsCcGenerator, LiteOpsHGenerator
from ops_name_h_generator import OpsNameHGenerator
from functional_map_cpp_generator import FunctionalMapCppGenerator

from op_proto import OpProto
from tensor_func_proto import load_func_protos_from_yaml
from tensor_func_reg_cpp_generator import TensorFuncRegCppGenerator
from gen_pyboost_func import gen_pyboost_code

import gen_constants as K


def generate_ops_prim_file(work_path, op_protos, doc_dict, file_pre):
    generator = OpPrimPyGenerator()
    generator.generate(work_path, op_protos, doc_dict, file_pre)


def generate_ops_def_file(work_path, os_protos, doc_dict, file_pre):
    generator = OpDefPyGenerator()
    generator.generate(work_path, os_protos, doc_dict, file_pre)


def generate_ops_py_files(work_path, op_protos, doc_dict, file_pre):
    """
    Generate ops python file from yaml.
    """
    generate_ops_prim_file(work_path, op_protos, doc_dict, file_pre)
    generate_ops_def_file(work_path, op_protos, doc_dict, file_pre)
    shutil.copy(os.path.join(work_path, K.PY_OPS_GEN_PATH, 'ops_auto_generate_init.txt'),
                os.path.join(work_path, K.PY_AUTO_GEN_PATH, "__init__.py"))


def call_ops_def_cc_generator(work_path, op_protos):
    generator = OpsDefCcGenerator()
    generator.generate(work_path, op_protos)


def call_ops_primitive_h_generator(work_path, op_protos):
    generator = OpsPrimitiveHGenerator()
    generator.generate(work_path, op_protos)


def call_lite_ops_h_generator(work_path, op_protos):
    h_generator = LiteOpsHGenerator()
    h_generator.generate(work_path, op_protos)


def call_lite_ops_cc_generator(work_path, op_protos):
    generator = LiteOpsCcGenerator()
    generator.generate(work_path, op_protos)


def call_ops_name_h_generator(work_path, op_protos):
    h_generator = OpsNameHGenerator()
    h_generator.generate(work_path, op_protos)


def generate_ops_cc_files(work_path, op_protos):
    """
    Generate ops c++ file from yaml.
    """
    call_ops_def_cc_generator(work_path, op_protos)
    call_ops_primitive_h_generator(work_path, op_protos)
    call_lite_ops_h_generator(work_path, op_protos)
    call_lite_ops_cc_generator(work_path, op_protos)
    call_ops_name_h_generator(work_path, op_protos)


def generate_create_instance_helper_file(work_path, op_protos):
    """
    Generate C++ helper file from yaml.
    """
    generator = CppCreatePrimInstanceHelperGenerator()
    generator.generate(work_path, op_protos)


def generate_aclnn_reg_file(work_path, op_protos):
    """
    Generate nnacl kernelmod register
    """
    generator = AclnnKernelRegisterAutoCcGenerator()
    generator.generate(work_path, op_protos)


def generate_arg_handler_files(work_path):
    """
    Generate arg handler files.
    """
    dst_dir = os.path.join(work_path, K.PY_AUTO_GEN_PATH)
    src_arg_handler_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'arg_handler.py')
    dst_arg_handler_path = os.path.join(dst_dir, 'gen_arg_handler.py')
    tmp_dst_arg_handler_path = os.path.join(dst_dir, 'tmp_gen_arg_handler.py')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, mode=0o700)
    shutil.copy(src_arg_handler_path, tmp_dst_arg_handler_path)
    check_change_and_replace_file(dst_arg_handler_path, tmp_dst_arg_handler_path)

    src_arg_dtype_cast_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'arg_dtype_cast.py')
    dst_arg_dtype_cast_path = os.path.join(dst_dir, 'gen_arg_dtype_cast.py')
    tmp_arg_dtype_cast_path = os.path.join(dst_dir, 'tmp_arg_dtype_cast.py')
    shutil.copy(src_arg_dtype_cast_path, tmp_arg_dtype_cast_path)
    check_change_and_replace_file(dst_arg_dtype_cast_path, tmp_arg_dtype_cast_path)


def gen_tensor_func_code(work_path, func_protos):
    generator = TensorFuncRegCppGenerator()
    generator.generate(work_path, func_protos)

def gen_functional_map_code(work_path, func_protos):
    generator = FunctionalMapCppGenerator()
    generator.generate(work_path, func_protos)


def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    work_path = os.path.join(current_path, '../../../../')

    # merge ops yaml
    doc_yaml_path, ops_yaml_path, tensor_yaml_path = merge_ops_yaml(work_path)

    # make auto_generate dir
    cc_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_PATH)
    pathlib.Path(cc_path).mkdir(parents=True, exist_ok=True)

    # generate arg_handler files
    generate_arg_handler_files(work_path)

    # read ops definition str and doc str
    ops_yaml_dict = safe_load_yaml(ops_yaml_path)
    doc_yaml_dict = safe_load_yaml(doc_yaml_path)
    tensor_yaml_dict = safe_load_yaml(tensor_yaml_path)
    op_protos = load_op_protos_from_ops_yaml(ops_yaml_dict)
    func_protos = load_func_protos_from_yaml(tensor_yaml_dict, op_protos)

    # generate ops python files
    generate_ops_py_files(work_path, op_protos, doc_yaml_dict, "gen")
    # generate ops c++ files
    generate_ops_cc_files(work_path, op_protos)
    # generate create prim instance helper file
    generate_create_instance_helper_file(work_path, op_protos)
    # generate pyboost code
    gen_pyboost_code(work_path, op_protos, doc_yaml_dict, func_protos)
    # generate aclnn kernelmod register
    generate_aclnn_reg_file(work_path, op_protos)
    # generate tensor_py func code
    gen_tensor_func_code(work_path, func_protos)
    # generate functional map code
    gen_functional_map_code(work_path, func_protos)


def load_ops_yaml_to_op_protos(ops_yaml_data):
    """
    Converts YAML operator data to OpProto objects.

    Args:
        ops_yaml_data (dict): YAML data containing operator definitions.

    Returns:
        List[OpProto]: A list of OpProto objects created from the YAML data.
    """
def load_op_protos_from_ops_yaml(ops_yaml_data):
    op_protos = []
    for operator_name, operator_data in ops_yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        op_protos.append(op_proto)
    return op_protos


def merge_ops_yaml(work_path):
    """
    Merges operator YAML files scattered in different directories into a single file.

    Args:
        work_path (str): The path to the working directory.

    Returns:
        tuple: Paths to the merged documentation and operators YAML files.
    """
    ops_yaml_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'ops.yaml')
    ops_yaml_dir_path = os.path.join(work_path, K.MS_OP_DEF_YAML_PATH)
    infer_ops_yaml_dir_path = os.path.join(ops_yaml_dir_path, "infer")
    merge_files(ops_yaml_dir_path, ops_yaml_path, '*op.yaml')
    merge_files_append(infer_ops_yaml_dir_path, ops_yaml_path, '*op.yaml')

    doc_yaml_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'ops_doc.yaml')
    doc_yaml_dir_path = os.path.join(ops_yaml_dir_path, "doc")
    merge_files(doc_yaml_dir_path, doc_yaml_path, '*doc.yaml')

    tensor_yaml_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'tensor.yaml')
    tensor_yaml_dir_path = os.path.join(work_path, K.MS_TENSOR_YAML_PATH)
    merge_files(tensor_yaml_dir_path, tensor_yaml_path, '*.yaml')

    return doc_yaml_path, ops_yaml_path, tensor_yaml_path


if __name__ == "__main__":
    try:
        main()
    # pylint: disable=broad-except
    except Exception as e:
        logging.critical("Auto generate failed, err info: %s", e)
