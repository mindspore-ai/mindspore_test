/**
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
 */
#include "kernel/ascend/opapi/aclnn/batch_norm_gather_stats_with_counts_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
void BatchNormGatherStatsWithCountsAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs) {
  auto momentum = transform::ConvertKernelTensor<float>(inputs[kIndex5]);
  auto eps = transform::ConvertKernelTensor<float>(inputs[kIndex6]);
  double momentum_d = static_cast<double>(momentum);
  double eps_d = static_cast<double>(eps);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4], momentum_d,
                        eps_d, inputs[kIndex7], outputs[kIndex0], outputs[kIndex1]);
}

bool BatchNormGatherStatsWithCountsAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &workspace,
                                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto momentum = transform::ConvertKernelTensor<float>(inputs[kIndex5]);
  auto eps = transform::ConvertKernelTensor<float>(inputs[kIndex6]);
  double momentum_d = static_cast<double>(momentum);
  double eps_d = static_cast<double>(eps);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
        momentum_d, eps_d, inputs[kIndex7], outputs[kIndex0], outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(BatchNormGatherStatsWithCounts, BatchNormGatherStatsWithCountsAscend);
}  // namespace kernel
}  // namespace mindspore
