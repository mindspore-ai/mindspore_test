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
#include "kernel/ascend/opapi/aclnn/silent_check_v3_aclnn_kernel.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "include/common/utils/utils.h"
#include "ir/tensor.h"
#include "kernel/kernel.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/acl_helper.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "transform/symbol/acl_rt_symbol.h"
#include "transform/symbol/symbol_utils.h"

namespace mindspore {
namespace kernel {
void SilentCheckV3Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  c_thresh_l1_ = inputs[kIndex8]->GetValueWithCheck<pyfloat>();
  c_thresh_l2_ = inputs[kIndex9]->GetValueWithCheck<pyfloat>();
  beta1_ = inputs[kIndex10]->GetValueWithCheck<pyfloat>();
  npu_asd_detect_ = inputs[kIndex11]->GetValueWithCheck<int64_t>();
  ClearOpsWorkSpaceList();
  GetWorkspaceForResizeSilentCheckV3(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3],
                                     inputs[kIndex4], inputs[kIndex5], inputs[kIndex6], inputs[kIndex7], c_thresh_l1_,
                                     c_thresh_l2_, beta1_, npu_asd_detect_, outputs[kIndex3]);
  GetWorkspaceForResizeInputGradCopy(outputs[kIndex1], inputs[kIndex3]);
}

bool SilentCheckV3Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOpSilentCheckV3(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3],
                     inputs[kIndex4], inputs[kIndex5], inputs[kIndex6], inputs[kIndex7], c_thresh_l1_, c_thresh_l2_,
                     beta1_, npu_asd_detect_, outputs[kIndex3]);
  RunOpInputGradCopy(stream_ptr, workspace, outputs[kIndex1], inputs[kIndex3]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SilentCheckV3, SilentCheckV3Ascend);
}  // namespace kernel
}  // namespace mindspore
