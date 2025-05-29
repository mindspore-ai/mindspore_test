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
#include "kernel/ascend/opapi/aclnn/swiglu_quant_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <string>
#include "ir/tensor.h"
#include "plugin/res_manager/ascend/op_adapter/op_adapter_base.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
namespace swiglu_quant {
void SwigluQuantAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);
  auto quant_mode_int = inputs[kIndex5]->GetValueWithCheck<int64_t>();
  quant_mode_ = device::ascend::AscendQuantMode::ConvertEnumToString(quant_mode_int);
  activate_left_ = inputs[kIndex4]->GetValueWithCheck<bool>();
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], activate_left_, quant_mode_,
                        outputs[kIndex0], outputs[kIndex1]);
}

bool SwigluQuantAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], activate_left_,
        quant_mode_, outputs[kIndex0], outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SwigluQuant, SwigluQuantAscend);
}  // namespace swiglu_quant
}  // namespace kernel
}  // namespace mindspore
