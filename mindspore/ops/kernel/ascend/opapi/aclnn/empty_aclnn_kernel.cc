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

#include "kernel/ascend/opapi/aclnn/empty_aclnn_kernel.h"
#include <vector>
#include <string>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
void EmptyAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  return;
}

bool EmptyAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  auto device_name_opt = inputs[kIndex2]->GetOptionalValueWithCheck<int64_t>();
  if (device_name_opt.has_value()) {
    auto device_name_enum = device_name_opt.value();
    if (device_name_enum != DEVICE_ASCEND && device_name_enum != DEVICE_NPU_LOWER) {
      MS_LOG(EXCEPTION) << "Empty kbk mode only support ['Ascend', 'npu'] for device";
    }
  }
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(Empty, EmptyAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
