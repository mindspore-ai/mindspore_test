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
from ops_header_files_generator import OpHeaderFileGenerator
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


def gen_pyboost_code(work_path, op_protos, doc_yaml_data):
    """ gen_pyboost_code """
    call_pyboost_inner_prim_generator(work_path, op_protos)
    call_pyboost_functions_py_generator(work_path, op_protos, doc_yaml_data)
    call_ops_header_files_generator(work_path, op_protos)
    call_pyboost_functions_cpp_generator(work_path, op_protos)
    call_pyboost_grad_functions_cpp_generator(work_path, op_protos)
    call_pyboost_native_grad_functions_generator(work_path, op_protos)
    call_pyboost_op_cpp_code_generator(work_path, op_protos)

def call_pyboost_inner_prim_generator(work_path, ops_yaml_data):
    generator = PyboostInnerPrimGenerator()
    generator.generate(work_path, ops_yaml_data)

def call_pyboost_functions_py_generator(work_path, ops_yaml_data, doc_yaml_data):
    generator = PyboostFunctionsPyGenerator()
    generator.generate(work_path, ops_yaml_data, doc_yaml_data)

def call_ops_header_files_generator(work_path, ops_yaml_data):
    generator = OpHeaderFileGenerator()
    generator.generate(work_path, ops_yaml_data)

def call_pyboost_functions_cpp_generator(work_path, ops_yaml_data):
    generator = PyboostFunctionsGenerator()
    generator.generate(work_path, ops_yaml_data)

def call_pyboost_grad_functions_cpp_generator(work_path, ops_yaml_data):
    generator = PyboostGradFunctionsGenerator()
    generator.generate(work_path, ops_yaml_data)

def call_pyboost_native_grad_functions_generator(work_path, ops_yaml_data):
    h_generator = PyboostGradFunctionsHeaderGenerator()
    h_generator.generate(work_path, ops_yaml_data)

    cc_generator = PyboostGradFunctionsCppGenerator()
    cc_generator.generate(work_path, ops_yaml_data)

def call_pyboost_op_cpp_code_generator(work_path, ops_yaml_data):
    call_PyboostCommonOpCppCodeGenerator(work_path, ops_yaml_data)
    call_PyboostOpHeaderGenerator(work_path, ops_yaml_data)
    call_PyboostOpCppGenerator(work_path, ops_yaml_data)
    call_PyboostViewOpCppGenerator(work_path, ops_yaml_data)
    call_AclnnOpCppCodeGenerator(work_path, ops_yaml_data)
    delete_residual_files(work_path, ops_yaml_data)
    call_PyboostOpRegisterCppCodeGenerator(work_path, ops_yaml_data)

def call_PyboostCommonOpCppCodeGenerator(work_path, ops_yaml_data):
    generator = PyboostCommonOpHeaderGenerator()
    generator.generate(work_path, ops_yaml_data)

def call_PyboostOpHeaderGenerator(work_path, ops_yaml_data):
    generator = PyboostOpHeaderGenerator('ascend')
    generator.generate(work_path, ops_yaml_data)

    generator = PyboostOpHeaderGenerator('gpu')
    generator.generate(work_path, ops_yaml_data)

    generator = PyboostOpHeaderGenerator('cpu')
    generator.generate(work_path, ops_yaml_data)

def call_PyboostOpCppGenerator(work_path, ops_yaml_data):
    ascend_op_cpp_generator = PyboostOpCppGenerator('ascend')
    ascend_op_cpp_generator.generate(work_path, ops_yaml_data)

    cpu_op_cpp_generator = PyboostOpCppGenerator('cpu')
    cpu_op_cpp_generator.generate(work_path, ops_yaml_data)

    gpu_op_cpp_generator = PyboostOpCppGenerator('gpu')
    gpu_op_cpp_generator.generate(work_path, ops_yaml_data)

def call_PyboostViewOpCppGenerator(work_path, ops_yaml_data):
    ascend_view_op_cpp_generator = PyboostViewOpCppGenerator('ascend')
    ascend_view_op_cpp_generator.generate(work_path, ops_yaml_data)

    cpu_view_op_cpp_generator = PyboostViewOpCppGenerator('cpu')
    cpu_view_op_cpp_generator.generate(work_path, ops_yaml_data)

    gpu_view_op_cpp_generator = PyboostViewOpCppGenerator('gpu')
    gpu_view_op_cpp_generator.generate(work_path, ops_yaml_data)

def call_AclnnOpCppCodeGenerator(work_path, ops_yaml_data):
    ascend_aclnn_cpp_generator = AclnnOpCppCodeGenerator('ascend')
    ascend_aclnn_cpp_generator.generate(work_path, ops_yaml_data)

    cpu_aclnn_cpp_generator = AclnnOpCppCodeGenerator('cpu')
    cpu_aclnn_cpp_generator.generate(work_path, ops_yaml_data)

    gpu_aclnn_cpp_generator = AclnnOpCppCodeGenerator('gpu')
    gpu_aclnn_cpp_generator.generate(work_path, ops_yaml_data)

def call_PyboostOpRegisterCppCodeGenerator(work_path, ops_yaml_data):
    op_register_cpp_generator = PyboostOpRegisterCppCodeGenerator()
    op_register_cpp_generator.generate(work_path, ops_yaml_data)
