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
"""Templates for code auto generation."""
import re
import os
import gen_constants as K


class Template:
    """
    template for generate c++/python code
    """
    regular_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    regular_match = re.compile(regular_str, re.MULTILINE)

    def __init__(self, code_pattern):
        self.code_pattern = code_pattern

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, "r") as f:
            return Template(f.read())

    def replace(self, **kwargs):
        """
        replace param.
        :param kwargs:
        :return:
        """

        def find(key: str):
            if key in kwargs:
                return kwargs[key]
            raise TypeError(f"{key} should be in kwargs!")

        def add_indent(indent, var):
            return "".join([indent + line + "\n" for data in var for line in str(data).splitlines()]).rstrip()

        def extract_variable(key):
            start = ""
            end = ""
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    start = ","
                    key = key[1:]
                if key[-1] == ",":
                    end = ", "
                    key = key[:-1]
            return find(key), start, end

        def match_rule(match):
            indent = match.group(1)
            key = match.group(2)
            var, start, end = extract_variable(key)
            if indent is not None:
                if not isinstance(var, list):
                    return add_indent(indent, [var])
                return add_indent(indent, var)
            if isinstance(var, list):
                code = ", ".join(str(x) for x in var)
                if not var:
                    return code
                return start + code + end
            return str(var)

        return self.regular_match.sub(match_rule, self.code_pattern)


NEW_LINE = "\n"
WORK_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../")

PYTHON_PRIM_TEMPLATE = Template("""

class _Pyboost${class_name}Prim(${class_name}Prim_):
    def __call__(self, ${input_args}):
        ${process_func}
        return _convert_stub(super().__call__([${processed_args}]))


${func_impl_name}_impl = _Pyboost${class_name}Prim()
""")

IMPORT_PYBOOST_PRIM_HEADER = f"""
from mindspore.common._stub_tensor import _convert_stub
from mindspore.ops.auto_generate.gen_arg_handler import *
"""

IMPORT_PYBOOST_FUNC_HEADER = f"""
from mindspore.common import dtype as mstype
from mindspore.ops.auto_generate.pyboost_inner_prim import *

"""

REGISTER_DEFINE_TEMPLATE = Template(
    """
    (void)py::class_<${class_name}PrimAdapter, PrimitiveFunctionAdapter, std::shared_ptr<${class_name}PrimAdapter>>(
      *m, "${class_name}Prim_")
      .def(py::init<>())
      .def("__call__", &${class_name}PrimAdapter::Call, "Call ${class_name} op.");
    m->def(\"${pyboost_op_name}\", &mindspore::pynative::${pyboost_cfunc_name}, \"Encrypt the data.\");""")
REGISTER_TEMPLATE = Template("void RegisterPyBoostFunction(py::module *m) {${register_func}\n}")

REGISTER_PYBOOST_GRAD_DEFINE_TEMPLATE = Template(
    "MS_REG_PYBOOST_GRAD_OP(${pyboost_op_name}, mindspore::runtime::${pyboost_cfunc_name});\n")
REGISTER_PYBOOST_GRAD_TEMPLATE = Template("${register_func}")

PYBOOST_FUNCTION_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_function.tpl'))

PYBOOST_COMM_FUNCTION_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_comm_function.tpl'))

PYBOOST_HEADER_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_function_header.tpl'))

PYBOOST_OVERLOAD_FUNCTIONS_CC_TEMPLATE = Template.load_from_file(
    os.path.join(
        WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_overload_functions_cc.tpl'))

PYBOOST_MINT_CLASS_DEF = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_mint_class_def.tpl'))

PYBOOST_OVERLOAD_MINT_CLASS_DEF = Template.load_from_file(
    os.path.join(
        WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/pyboost_overload_mint_class_def.tpl'))

# template path need to be moved
FUNCTIONAL_OVERLOAD_PY_TEMPLATE = Template.load_from_file(
    os.path.join(
        WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/functional_overload_py.tpl')
)

TENSOR_FUNC_CC_REG = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_func_cc_reg.tpl'))

TENSOR_FUNC_HEADER_REG = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_func_header_reg.tpl'))

TENSOR_FUNC_HEADER_BODY = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_func_header_body.tpl'))

TENSOR_FUNC_CALL_BODY = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_func_call_body.tpl'))

TENSOR_FUNC_OVERLOAD_CALL_BODY = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_func_overload_call_body.tpl'))

TENSOR_FUNC_UT_BODY = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_func_ut_body.tpl'))

TENSOR_FUNC_UT_OVERLOAD_BODY = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_func_ut_overload_body.tpl'))

TENSOR_CPP_METHOD = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_cpp_method.tpl'))

TENSOR_FUNC_CLASS_REG = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_func_class_reg.tpl'))


PYBOOST_GRAD_FUNCTION_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/runtime/pynative/op_function/template/pyboost_grad_function.tpl'))

PYBOOST_GRAD_HEADER_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/runtime/pynative/op_function/template/pyboost_grad_function_header.tpl'))

PYBOOST_NATIVE_GRAD_FUNCTION_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/grad/function/template/native_grad_function.tpl'))

PYBOOST_NATIVE_GRAD_FUNCTIONS_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/pipeline/pynative/grad/function/template/pyboost_native_grad_functions.tpl'))

PYBOOST_NATIVE_GRAD_FUNCTIONS_HEADER_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ccsrc/pipeline/pynative/grad/function/template/pyboost_native_grad_functions_header.tpl'))

GEN_OPS_DEF_HEADER_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/python/mindspore/ops_generate/gen_ops_def_header.tpl'))

PYBOOST_BASE_OP_DEFINE_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, f'./{K.MS_COMMON_PYBOOST_KERNEL_PATH}/template/pyboost_op_header.tpl'))

PYBOOST_OP_REGISTER_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, f'./{K.MS_COMMON_PYBOOST_KERNEL_PATH}/template/pyboost_op_register.tpl'))

# Ascend op generate
PYBOOST_ASCEND_OP_HEADER_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_OPS_KERNEL_PATH}/ascend/pyboost/template/pyboost_aclnn_header_template.tpl'))

PYBOOST_ASCEND_OP_SOURCE_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_OPS_KERNEL_PATH}/ascend/pyboost/template/pyboost_aclnn_source_template.tpl'))

PYBOOST_ASCEND_CALL_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_OPS_KERNEL_PATH}/ascend/pyboost/template/pyboost_ascend_call_template.tpl'))

PYBOOST_ASCEND_VIEW_CALL_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_COMMON_PYBOOST_KERNEL_PATH}/template/'
                 'pyboost_view_template.tpl'))

PYBOOST_ASCEND_CUSTOMIZE_CALL_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_OPS_KERNEL_PATH}/ascend/pyboost/template'
                 '/pyboost_ascend_customize_call_template.tpl'))

# GPU op generate
PYBOOST_GPU_OP_HEADER_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ops/kernel/gpu/pyboost/template/pyboost_gpu_header_template.tpl'))

PYBOOST_GPU_OP_SOURCE_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ops/kernel/gpu/pyboost/template/pyboost_gpu_source_template.tpl'))

PYBOOST_GPU_CALL_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ops/kernel/gpu/pyboost/template/pyboost_gpu_call_template.tpl'))

PYBOOST_GPU_VIEW_CALL_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_COMMON_PYBOOST_KERNEL_PATH}/template/pyboost_view_template.tpl'))

PYBOOST_GPU_CUSTOMIZE_CALL_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 './mindspore/ops/kernel/gpu/pyboost/template'
                 '/pyboost_gpu_customize_call_template.tpl'))

# CPU op generate
PYBOOST_CPU_OP_HEADER_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/template/pyboost_cpu_header_template.tpl'))

PYBOOST_CPU_OP_SOURCE_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/template/pyboost_cpu_source_template.tpl'))

PYBOOST_CPU_CALL_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/template/pyboost_cpu_call_template.tpl'))

PYBOOST_CPU_VIEW_CALL_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_COMMON_PYBOOST_KERNEL_PATH}/template/pyboost_view_template.tpl'))

PYBOOST_CPU_CUSTOMIZE_CALL_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./{K.MS_OPS_KERNEL_PATH}/cpu/pyboost/template'
                 '/pyboost_cpu_customize_call_template.tpl'))

PYBOOST_PY_FUNC_IMPORT_HEADEAR = Template(
    """from mindspore._c_expression import ${class_name}Prim_\n"""
)

PYBOOST_PY_FUNC_TEMPLATE = Template("""
def ${func_name}(${func_args}):
    r\"\"\"
    ${description}
    \"\"\"
    return ${func_impl_name}_impl(${input_args})\n\n""")

OP_PROTO_TEMPLATE = Template("""
${func_impl_declaration}
OpDef g${class_name} = {
  /*.name_=*/"${class_name}",
  /*.args_=*/ {
    ${input_args}
  },
  /* .returns_ = */ {
    ${return_args}
  },
  /*.signatures_ =*/ {
    ${signatures}
  },
  /*.indexes_ =*/ {
    ${indexes}
  },
  /*.func_impl_=*/${func_impl_define},
  /*.enable_dispatch_ =*/${enable_dispatch},
  /*.is_view_ =*/${is_view},
};
REGISTER_PRIMITIVE_OP_DEF(${class_name}, &g${class_name});
""")

MULTI_OUTPUT_TEMPLATE = """
ValuePtrList values;
(void)std::transform(op->outputs().begin(), op->outputs().end(), std::back_inserter(values),
                   [](const auto &value){ return value;});
auto output_value = std::make_shared<ValueTuple>(values);
"""

PY_LICENCE_STR = f"""# Copyright 2023 Huawei Technologies Co., Ltd
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

OPS_PY_PRIM_HEADER = f"""
\"\"\"Operators definition generated by gen_ops.py, includes primitive classes.\"\"\"

from mindspore.ops.primitive import Primitive, prim_arg_register
from mindspore.ops import signature as sig
from mindspore.common import dtype as mstype
from mindspore.common._decorator import deprecated
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate.gen_arg_dtype_cast import type_it
from mindspore.ops.auto_generate.gen_arg_handler import *
from mindspore._c_expression import OpDtype
from mindspore.common._stub_tensor import _convert_stub
"""

OP_PRIM_CLASS_DEFINE_TEMPLATE = Template("""

class ${class_name}(Primitive):
${class_desc}${signature_code}${deprecated_code}
${init_method}

${call_method}""")

OPS_PY_DEF_HEADER = f"""
\"\"\"Operators definition generated by gen_ops.py, includes functions.\"\"\"

from .gen_ops_prim import *
from .pyboost_inner_prim import *
from mindspore.ops.operations.manually_defined.ops_def import *
from mindspore.ops._primitive_cache import _get_cache_prim
"""

PRIMITIVE_CLASS_DESC = """    r\"\"\"
    .. code-block::

        prim = ops.$class_name($init_args_str)
        out = prim($input_args_str)

    is equivalent to

    .. code-block::

        ops.$func_name($args_str)

    Refer to :func:`mindspore.ops.$func_name` for more details.
    \"\"\"
"""

CC_LICENSE_STR = f"""/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */"""

arg_default_value = Template("""
\"\"\"Operator labels and args default value.\"\"\"
from mindspore.common import dtype as mstype

op_args_default_value = {
$gen_default_py
}
""")


op_labels_template = Template("""
op_labels = {
$gen_label_py
}
""")


lite_ops_h_class_template = Template("""class OPS_API ${op_name} : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(${op_name});
  ${op_name}() : BaseOperator(kName${op_name}) {}${lite_ops_h_code}
};

""")

op_cc_template = Template("""class OPS_API ${op_name} : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(${op_name});
  ${op_name}() : BaseOperator(kName${op_name}) {}${arg_prim_init_list}
};

""")

op_template = Template("""void ${op_name}::set_${arg_name}(const ${dtype} &${arg_name})"""
                       """             { (void)this->AddAttr("${arg_name}", api::MakeValue(${arg_name})); }\n\n"""
                       """${dtype} ${op_name}::get_${arg_name}() const"""
                       """             { return GetValue<${dtype}>(GetAttr("${arg_name}")); }\n\n""")

ACLNN_KERNEL_CC_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, f'./{K.MS_OPS_KERNEL_PATH}/ascend/opapi/template/aclnn_kernel_cc.tpl'))

ACLNN_KERNEL_H_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, f'./{K.MS_OPS_KERNEL_PATH}/ascend/opapi/template/aclnn_kernel_h.tpl'))

update_output_shape_and_size_template = Template("""
void ${kernelmod_name}::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &,
                                                const std::vector<KernelTensor *> &outputs) {
  // Delete these comment and complete the function:
  // Using outputs[index_x]->SetShapeVector(update_shape) and outputs[index_x]->set_size(update_size)
}
""")


UPDATE_OUTPUT_SHAPE_AND_SIZE = """
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
"""
TUPLE_TENSOR_NOT_SUPPORTED = Template("""
    It is not supported for ${op_name} with tuple[tensor] inputs when using auto generate.
    Please provide a KernelMod name in yaml and using python gen_aclnn_implement.py -n xx manually.""")

FUNCTIONAL_MAP_CC_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./mindspore/ccsrc/pipeline/pynative/op_function/template/functional_map_cc.tpl'))

FUNCTIONAL_MAP_H_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH,
                 f'./mindspore/ccsrc/pipeline/pynative/op_function/template/functional_map_h.tpl'))

ADD_TENSOR_DOCS_TEMPLATE = Template.load_from_file(
    os.path.join(WORK_PATH, './mindspore/ccsrc/pipeline/pynative/op_function/template/tensor_docs_py.tpl'))
