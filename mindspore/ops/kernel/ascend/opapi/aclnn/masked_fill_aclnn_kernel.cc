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
#include "kernel/ascend/opapi/aclnn/masked_fill_aclnn_kernel.h"
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
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {

void MaskedFillAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  GetWorkspaceForResizeInplaceCopy(outputs[kIndex0], inputs[kIndex0]);
  GetWorkspaceForResizeInplaceMaskedFillTensor(outputs[kIndex0], inputs[kIndex1], inputs[kIndex2]);
}

bool MaskedFillAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOpInplaceCopy(stream_ptr, workspace, outputs[kIndex0], inputs[kIndex0]);
  RunOpInplaceMaskedFillTensor(stream_ptr, workspace, outputs[kIndex0], inputs[kIndex1], inputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MaskedFill, MaskedFillAscend);
}  // namespace kernel
}  // namespace mindspore
