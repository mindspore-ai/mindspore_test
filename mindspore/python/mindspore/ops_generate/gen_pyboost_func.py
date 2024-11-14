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
Generate pyboost function from pyboost_op.yaml
"""

from pyboost_inner_prim_generator import PyboostInnerPrimGenerator
from pyboost_functions_py_generator import PyboostFunctionsPyGenerator
from pyboost_functions_cpp_generator import PyboostFunctionsGenerator
from pyboost_grad_function_cpp_generator import PyboostGradFunctionsGenerator
from pyboost_native_grad_functions_generator import (
    PyboostGradFunctionsHeaderGenerator,
    PyboostGradFunctionsCppGenerator,
)
from pyboost_op_cpp_code_generator import (
    PyboostCommonOpHeaderGenerator,
    PyboostOpHeaderGenerator,
    PyboostOpCppGenerator,
    PyboostViewOpCppGenerator,
    AclnnOpCppCodeGenerator,
    delete_residual_files,
    PyboostOpRegisterCppCodeGenerator,
)
from pyboost_overload_functions_cpp_generator import PyboostOverloadFunctionsGenerator


def gen_pyboost_code(work_path, op_protos, doc_yaml_data, tensor_method_protos, mint_func_protos, alias_func_mapping):
    """ gen_pyboost_code """
    call_pyboost_inner_prim_generator(work_path, op_protos)
    call_pyboost_functions_py_generator(work_path, op_protos, doc_yaml_data)
    call_pyboost_functions_cpp_generator(work_path, op_protos, tensor_method_protos)
    call_pyboost_overload_functions_cpp_generator(work_path, op_protos, mint_func_protos, alias_func_mapping)
    call_pyboost_grad_functions_cpp_generator(work_path, op_protos)
    call_pyboost_native_grad_functions_generator(work_path, op_protos)
    call_pyboost_op_cpp_code_generator(work_path, op_protos)


def call_pyboost_inner_prim_generator(work_path, op_protos):
    generator = PyboostInnerPrimGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_functions_py_generator(work_path, op_protos, doc_yaml_data):
    generator = PyboostFunctionsPyGenerator()
    generator.generate(work_path, op_protos, doc_yaml_data)


def call_pyboost_functions_cpp_generator(work_path, op_protos, tensor_method_protos):
    generator = PyboostFunctionsGenerator()
    generator.generate(work_path, op_protos, tensor_method_protos)


def call_pyboost_overload_functions_cpp_generator(work_path, op_protos, mint_func_protos, alias_func_mapping):
    generator = PyboostOverloadFunctionsGenerator()
    generator.generate(work_path, op_protos, mint_func_protos, alias_func_mapping)


def call_pyboost_grad_functions_cpp_generator(work_path, op_protos):
    generator = PyboostGradFunctionsGenerator()
    generator.generate(work_path, op_protos)


def call_pyboost_native_grad_functions_generator(work_path, op_protos):
    h_generator = PyboostGradFunctionsHeaderGenerator()
    h_generator.generate(work_path, op_protos)

    cc_generator = PyboostGradFunctionsCppGenerator()
    cc_generator.generate(work_path, op_protos)


def call_pyboost_op_cpp_code_generator(work_path, op_protos):
    call_PyboostCommonOpCppCodeGenerator(work_path, op_protos)
    call_PyboostOpHeaderGenerator(work_path, op_protos)
    call_PyboostOpCppGenerator(work_path, op_protos)
    call_PyboostViewOpCppGenerator(work_path, op_protos)
    call_AclnnOpCppCodeGenerator(work_path, op_protos)
    delete_residual_files(work_path, op_protos)
    call_PyboostOpRegisterCppCodeGenerator(work_path, op_protos)


def call_PyboostCommonOpCppCodeGenerator(work_path, op_protos):
    generator = PyboostCommonOpHeaderGenerator()
    generator.generate(work_path, op_protos)


def call_PyboostOpHeaderGenerator(work_path, op_protos):
    generator = PyboostOpHeaderGenerator('ascend')
    generator.generate(work_path, op_protos)

    generator = PyboostOpHeaderGenerator('gpu')
    generator.generate(work_path, op_protos)

    generator = PyboostOpHeaderGenerator('cpu')
    generator.generate(work_path, op_protos)


def call_PyboostOpCppGenerator(work_path, op_protos):
    ascend_op_cpp_generator = PyboostOpCppGenerator('ascend')
    ascend_op_cpp_generator.generate(work_path, op_protos)

    cpu_op_cpp_generator = PyboostOpCppGenerator('cpu')
    cpu_op_cpp_generator.generate(work_path, op_protos)

    gpu_op_cpp_generator = PyboostOpCppGenerator('gpu')
    gpu_op_cpp_generator.generate(work_path, op_protos)


def call_PyboostViewOpCppGenerator(work_path, op_protos):
    ascend_view_op_cpp_generator = PyboostViewOpCppGenerator('ascend')
    ascend_view_op_cpp_generator.generate(work_path, op_protos)

    cpu_view_op_cpp_generator = PyboostViewOpCppGenerator('cpu')
    cpu_view_op_cpp_generator.generate(work_path, op_protos)

    gpu_view_op_cpp_generator = PyboostViewOpCppGenerator('gpu')
    gpu_view_op_cpp_generator.generate(work_path, op_protos)


def call_AclnnOpCppCodeGenerator(work_path, op_protos):
    ascend_aclnn_cpp_generator = AclnnOpCppCodeGenerator('ascend')
    ascend_aclnn_cpp_generator.generate(work_path, op_protos)

    cpu_aclnn_cpp_generator = AclnnOpCppCodeGenerator('cpu')
    cpu_aclnn_cpp_generator.generate(work_path, op_protos)

    gpu_aclnn_cpp_generator = AclnnOpCppCodeGenerator('gpu')
    gpu_aclnn_cpp_generator.generate(work_path, op_protos)


def call_PyboostOpRegisterCppCodeGenerator(work_path, op_protos):
    op_register_cpp_generator = PyboostOpRegisterCppCodeGenerator()
    op_register_cpp_generator.generate(work_path, op_protos)
