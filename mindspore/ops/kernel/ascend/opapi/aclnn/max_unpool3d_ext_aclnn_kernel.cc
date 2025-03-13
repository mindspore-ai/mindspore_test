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
#include "kernel/ascend/opapi/aclnn/max_unpool3d_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
void MaxUnpool3DExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  kernel_size_ = inputs[kIndex2]->GetValueWithCheck<std::vector<int64_t>>();
  stride_ = kernel_size_;
  if (inputs[kIndex3]->type_id() != kMetaTypeNone) {
    stride_ = inputs[kIndex3]->GetValueWithCheck<std::vector<int64_t>>();
  }
  padding_ = inputs[kIndex4]->GetValueWithCheck<std::vector<int64_t>>();
  if (stride_.size() == 1) {
    stride_.resize(kDim3, stride_[0]);
  }
  if (padding_.size() == 1) {
    padding_.resize(kDim3, padding_[0]);
  }
  ShapeVector out_shape = outputs[kIndex0]->GetShapeVector();
  output_size_.assign(out_shape.end() - kDim3, out_shape.end());
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], output_size_, stride_, padding_, outputs[kIndex0]);
}

bool MaxUnpool3DExtAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], output_size_, stride_, padding_, outputs[kIndex0]);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(MaxUnpool3DExt, MaxUnpool3DExtAscend);
}  // namespace kernel
}  // namespace mindspore
