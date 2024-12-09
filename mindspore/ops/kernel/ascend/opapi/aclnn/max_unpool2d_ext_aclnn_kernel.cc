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
#include "kernel/ascend/opapi/aclnn/max_unpool2d_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace kernel {

void MaxUnpool2DExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  ShapeVector shape = outputs[kIndex0]->GetShapeVector();
  size_t size = shape.size();
  output_size_.clear();
  output_size_.push_back(shape[size - kDim2]);
  output_size_.push_back(shape[size - kDim1]);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], output_size_, outputs[kIndex0]);
}

bool MaxUnpool2DExtAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], output_size_, outputs[kIndex0]);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(MaxUnpool2DExt, MaxUnpool2DExtAscend);
}  // namespace kernel
}  // namespace mindspore
