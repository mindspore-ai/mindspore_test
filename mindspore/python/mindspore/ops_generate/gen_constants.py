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
This module contains the constant strings used in generating ops files.

Constants:
    PY_LICENSE: License strings used for .py files
    CC_LICENSE: License strings used for .h/.cc files
    ......
    Other constant strings in the module are used for generation paths
"""

import os

# py license
PY_LICENSE = f"""# Copyright 2024 Huawei Technologies Co., Ltd
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

# cc license
CC_LICENSE = f"""/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

# op_def
OP_DEF_AUTO_GENERATE_PATH = "op_def/auto_generate"
MS_OP_DEF_AUTO_GENERATE_PATH = "mindspore/ops/op_def/auto_generate"
MS_OP_DEF_YAML_PATH = "mindspore/ops/op_def/yaml"
MS_OP_DEPRECATED_DEF_YAML_PATH = "mindspore/ops/op_def/deprecated"
MS_OP_API_YAML_PATH = "mindspore/ops/api_def"
MS_TENSOR_METHOD_DOC_YAML_PATH = "mindspore/ops/api_def/method_doc"
MS_MINT_FUNC_PATH = "mindspore/python/mindspore/mint"
PY_AUTO_GEN_PATH = "mindspore/python/mindspore/ops/auto_generate"
PY_OPS_GEN_PATH = "mindspore/python/mindspore/ops_generate"
PYBOOST_NATIVE_GRAD_FUNC_GEN_PATH = "mindspore/ccsrc/pipeline/pynative/grad/function/auto_generate"
PIPELINE_PYBOOST_FUNC_GEN_PATH = "mindspore/ccsrc/pipeline/pynative/op_function/auto_generate"
RUNTIME_PYBOOST_FUNC_GEN_PATH = "mindspore/ccsrc/runtime/pynative/op_function/auto_generate"
TENSOR_FUNC_REGISTER_PATH = "mindspore/ccsrc/pybind_api/ir/"
ADD_TENSOR_DOCS_PY_PATH = "mindspore/python/mindspore/common"

# yaml keys def
OP_KEYS = {'args', 'args_signature', 'returns', 'function', 'class', 'view', 'dispatch', 'labels'}
ARG_KEYS = {'dtype', 'default', 'prim_init', 'type_cast', 'arg_handler'}
RETURN_KEYS = {'dtype', 'inplace', 'type_cast'}
ARG_SIGNATURE_KEYS = {'rw_write', 'rw_read', 'rw_ref', 'dtype_group'}
CLASS_KEYS = {'name', 'disable'}
FUNCTION_KEYS = {'name', 'disable'}
DISPATCH_KEYS = {'enable', 'is_comm_op', 'Ascend', 'GPU', 'CPU'}
TENSOR_FUNC_KEYS = {'op_yaml', 'py_method', 'kwonlyargs', 'alias', 'Ascend', 'GPU', 'CPU', 'interface'}

# infer
MS_OPS_FUNC_IMPL_PATH = "mindspore/ops/infer/ops_func_impl"

# view
MS_OPS_VIEW_PATH = "mindspore/ops/view"

# kernel
MS_OPS_KERNEL_PATH = "mindspore/ops/kernel"
MS_COMMON_PYBOOST_KERNEL_PATH = os.path.join(MS_OPS_KERNEL_PATH, "common/pyboost")

MS_COMMON_PYBOOST_KERNEL_PATH = os.path.join(MS_OPS_KERNEL_PATH, "common/pyboost")

OP_NAME_OP_DEF = """
#ifndef MINDSPORE_CORE_OP_NAME_H_
#define MINDSPORE_CORE_OP_NAME_H_

namespace mindspore::ops {
$ops_namespace_body
}  // namespace mindspore::ops

#endif  // MINDSPORE_CORE_OP_NAME_H_
"""

OP_PRIM_OP_DEF = """
#ifndef MINDSPORE_CORE_OPS_GEN_OPS_PRIMITIVE_H_
#define MINDSPORE_CORE_OPS_GEN_OPS_PRIMITIVE_H_

#include <memory>
#include "ir/anf.h"
#include "ir/primitive.h"
#include "$auto_gen_path/gen_ops_name.h"
#include "mindapi/base/macros.h"

namespace mindspore::prim {
$ops_prim_gen
}  // namespace mindspore::prim
#endif  // MINDSPORE_CORE_OPS_GEN_OPS_PRIMITIVE_H_
"""

LITE_OPS_CC = """
#include "$auto_gen_path/gen_lite_ops.h"
#include "mindapi/helper.h"
#include "ops/primitive_c.h"
#include "ops/base_operator.h"
#include "abstract/abstract_value.h"

namespace mindspore::ops {
$ops_namespace_body

}  // namespace mindspore::ops
    """

LITE_OPS_H = """
#ifndef MINDSPORE_CORE_OPS_GEN_LITE_OPS_H_
#define MINDSPORE_CORE_OPS_GEN_LITE_OPS_H_

#include <vector>
#include "ops/base_operator.h"
#include "$auto_gen_path/gen_ops_name.h"

namespace mindspore::ops {
$ops_namespace_body

}  // namespace mindspore::ops
#endif  // MINDSPORE_CORE_OPS_GEN_LITE_OPS_H_
"""

CC_OPS_DEF = """

#include "$auto_generate_path/gen_ops_def.h"
#include "ir/signature.h"
$gen_include

namespace mindspore::ops {$gen_cc_code
}  // namespace mindspore::ops
"""

ACLNN_REG_CODE = """
#include "$ops_gen_kernel_path/ascend/opapi/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {

$aclnn_reg_code
}  // namespace kernel
}  // namespace mindspore
"""
