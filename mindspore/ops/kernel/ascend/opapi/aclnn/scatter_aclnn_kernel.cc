/**
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
 */
#include "kernel/ascend/opapi/aclnn/scatter_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace kernel {

void ScatterAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  dim_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  reduce_ = transform::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  GetWorkspaceForResize(inputs[kIndex0], dim_, inputs[kIndex2], inputs[kIndex3], reduce_, outputs[kIndex0]);
}

bool ScatterAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto status = CALL_ASCEND_API(aclrtMemcpyAsync, outputs[0]->device_ptr(), outputs[0]->size(), inputs[0]->device_ptr(),
                                inputs[0]->size(), ACL_MEMCPY_DEVICE_TO_DEVICE, stream_ptr);
  if (status != ACL_ERROR_NONE) {
    MS_LOG(EXCEPTION) << "ScatterAscend Launch and call rtMemcpyAsync failed, ret = 0x" << status;
  }
  RunOp(stream_ptr, workspace, inputs[kIndex0], dim_, inputs[kIndex2], inputs[kIndex3], reduce_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Scatter, ScatterAscend);
}  // namespace kernel
}  // namespace mindspore
