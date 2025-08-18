/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
 */
#ifndef MINDSPORE_CCSRC_MS_EXTENSION_ALL_H_
#define MINDSPORE_CCSRC_MS_EXTENSION_ALL_H_
#include "ms_extension/api.h"

#include "ir/tensor.h"
#include "mindspore/ccsrc/frontend/ir/tensor_py.h"

// pyboost headfiles
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "backend/common/device_address_utils.h"
#include "runtime/pynative/op_runner.h"
#include "mindspore/ccsrc/pyboost/op_runner.h"
#include "mindspore/ccsrc/pyboost/functions/auto_generate/functions.h"
#include "mindspore/ccsrc/tools/profiler/profiler.h"

// ascend files
#ifdef CUSTOM_ASCEND_OP
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/custom_aclnn_utils.h"
#include "mindspore/ccsrc/pynative/backward/function.h"
#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/custom_launch_aclnn.h"
#endif  // CUSTOM_ASCEND_OP

// The BaseTensor is deprecated
namespace mindspore {
namespace tensor {
using BaseTensor = Tensor;
using BaseTensorPtr = TensorPtr;
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MS_EXTENSION_ALL_H_
