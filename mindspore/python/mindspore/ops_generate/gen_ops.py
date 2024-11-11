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
import copy
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
from ops_def_h_generator import OpsDefHGenerator
from ops_primitive_h_generator import OpsPrimitiveHGenerator
from lite_ops_cpp_generator import LiteOpsCcGenerator, LiteOpsHGenerator
from ops_name_h_generator import OpsNameHGenerator
from functional_map_cpp_generator import FunctionalMapCppGenerator
from add_tensor_docs_generator import AddTensorDocsGenerator
from functional_overload_py_generator import FunctionalOverloadPyGenerator

from op_proto import OpProto
from op_api_proto import load_api_protos_from_yaml
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


def call_ops_def_h_generator(work_path, op_protos):
    generator = OpsDefHGenerator()
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


def generate_ops_cc_files(work_path, op_protos, op_protos_with_deprecated):
    """
    Generate ops c++ file from yaml.
    """
    call_ops_def_cc_generator(work_path, op_protos_with_deprecated)
    call_ops_def_h_generator(work_path, op_protos_with_deprecated)
    call_ops_primitive_h_generator(work_path, op_protos)
    call_lite_ops_h_generator(work_path, op_protos)
    call_lite_ops_cc_generator(work_path, op_protos)
    call_ops_name_h_generator(work_path, op_protos)


def get_tensor_op_protos_with_deprecated(func_protos, op_protos):
    """
    Get op_protos with deprecated op_protos from func_protos.
    """
    tensor_op_protos = copy.deepcopy(op_protos)
    for _, item in func_protos.items():
        for func_proto in item:
            op_name = func_proto.op_proto.op_name
            if "deprecated" in func_proto.op_proto.op_name:
                func_proto.op_proto.op_class.name = ''.join(word.capitalize() for word in op_name.split('_'))
                tensor_op_protos.append(func_proto.op_proto)
    return tensor_op_protos


def generate_create_instance_helper_file(work_path, op_protos_with_deprecated):
    """
    Generate C++ helper file from yaml.
    """
    generator = CppCreatePrimInstanceHelperGenerator()
    generator.generate(work_path, op_protos_with_deprecated)


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


def gen_tensor_func_code(work_path, func_protos, alias_api_mapping):
    generator = TensorFuncRegCppGenerator()
    generator.generate(work_path, func_protos, alias_api_mapping)


def gen_functional_map_code(work_path, tensor_method_protos, mint_func_protos, alias_api_mapping):
    generator = FunctionalMapCppGenerator()
    generator.generate(work_path, tensor_method_protos, mint_func_protos, alias_api_mapping)


def gen_tensor_docs_code(work_path, tensor_docs_data):
    generator = AddTensorDocsGenerator()
    generator.generate(work_path, tensor_docs_data)


def gen_functional_overload_py(work_path, mint_func_protos, alias_api_mapping):
    generator = FunctionalOverloadPyGenerator()
    generator.generate(work_path, mint_func_protos, alias_api_mapping)


def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    work_path = os.path.join(current_path, '../../../../')

    # merge ops yaml
    doc_yaml_path, ops_yaml_path, deprecated_ops_yaml_path, ops_api_yaml_path, tensor_method_doc_yaml_path \
        = merge_ops_yaml(work_path)

    # make auto_generate dir
    cc_path = os.path.join(work_path, K.MS_OP_DEF_AUTO_GENERATE_PATH)
    pathlib.Path(cc_path).mkdir(parents=True, exist_ok=True)

    # generate arg_handler files
    generate_arg_handler_files(work_path)

    # read ops definition str and tensor method doc str
    ops_yaml_dict = safe_load_yaml(ops_yaml_path)
    doc_yaml_dict = safe_load_yaml(doc_yaml_path)
    deprecated_ops_yaml_dict = safe_load_yaml(deprecated_ops_yaml_path)
    ops_api_yaml_dict = safe_load_yaml(ops_api_yaml_path)
    tensor_method_doc_yaml_dict = safe_load_yaml(tensor_method_doc_yaml_path)

    op_protos = load_op_protos_from_ops_yaml(ops_yaml_dict)
    deprecated_op_protos = load_deprecated_op_protos_from_ops_yaml(deprecated_ops_yaml_dict)
    tensor_method_protos, mint_func_protos, alias_api_mapping \
        = load_api_protos_from_yaml(ops_api_yaml_dict, op_protos, deprecated_op_protos)
    # for generate tensor method deprecated in graph mode
    op_protos_with_deprecated = get_tensor_op_protos_with_deprecated(tensor_method_protos, op_protos)

    # generate ops python files
    generate_ops_py_files(work_path, op_protos, doc_yaml_dict, "gen")
    # generate ops c++ files
    generate_ops_cc_files(work_path, op_protos, op_protos_with_deprecated)
    # generate create prim instance helper file
    generate_create_instance_helper_file(work_path, op_protos_with_deprecated)
    # generate pyboost code
    gen_pyboost_code(work_path, op_protos, doc_yaml_dict, tensor_method_protos, mint_func_protos, alias_api_mapping)
    # generate aclnn kernelmod register
    generate_aclnn_reg_file(work_path, op_protos)
    # generate tensor_py func code
    gen_tensor_func_code(work_path, tensor_method_protos, alias_api_mapping)
    # generate functional map code
    gen_functional_map_code(work_path, tensor_method_protos, mint_func_protos, alias_api_mapping)
    # generate _tensor_docs.py that attaches docs to tensor func APIs when import mindspore
    gen_tensor_docs_code(work_path, tensor_method_doc_yaml_dict)
    # generate functional_overload.py which init pybind mint APIs from cpp
    gen_functional_overload_py(work_path, mint_func_protos, alias_api_mapping)


def load_op_protos_from_ops_yaml(ops_yaml_data):
    op_protos = []
    for operator_name, operator_data in ops_yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        op_protos.append(op_proto)
    return op_protos


def load_deprecated_op_protos_from_ops_yaml(ops_yaml_data):
    op_protos = []
    for operator_name, operator_data in ops_yaml_data.items():
        op_proto = OpProto.load_from_yaml(operator_name, operator_data)
        op_proto.op_name = 'deprecated_' + operator_name
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
    ops_yaml_dir_path = os.path.join(work_path, K.MS_OP_DEF_YAML_PATH)
    ops_yaml_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'ops.yaml')
    infer_ops_yaml_dir_path = os.path.join(ops_yaml_dir_path, "infer")
    merge_files(ops_yaml_dir_path, ops_yaml_path, '*op.yaml')
    merge_files_append(infer_ops_yaml_dir_path, ops_yaml_path, '*op.yaml')

    doc_yaml_dir_path = os.path.join(ops_yaml_dir_path, "doc")
    doc_yaml_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'ops_doc.yaml')
    merge_files(doc_yaml_dir_path, doc_yaml_path, '*doc.yaml')

    ops_api_yaml_dir_path = os.path.join(work_path, K.MS_OP_API_YAML_PATH)
    ops_api_yaml_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'api_def.yaml')
    merge_files(ops_api_yaml_dir_path, ops_api_yaml_path, '*.yaml')

    deprecated_ops_yaml_dir_path = os.path.join(work_path, K.MS_OP_DEPRECATED_DEF_YAML_PATH)
    deprecated_ops_yaml_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'deprecated_ops.yaml')
    merge_files(deprecated_ops_yaml_dir_path, deprecated_ops_yaml_path, '*_op.yaml')

    tensor_method_doc_yaml_dir_path = os.path.join(work_path, K.MS_TENSOR_METHOD_DOC_YAML_PATH)
    tensor_method_doc_yaml_path = os.path.join(work_path, K.PY_OPS_GEN_PATH, 'tensor_method_doc.yaml')
    merge_files(tensor_method_doc_yaml_dir_path, tensor_method_doc_yaml_path, '*doc.yaml')

    return doc_yaml_path, ops_yaml_path, deprecated_ops_yaml_path, ops_api_yaml_path, tensor_method_doc_yaml_path


if __name__ == "__main__":
    try:
        main()
    # pylint: disable=broad-except
    except Exception as e:
        logging.critical("Auto generate failed, err info: %s", e)
        raise e
