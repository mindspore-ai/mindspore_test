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

#include "plugin/device/ascend/kernel/internal/split.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalSplit::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                    const internal::OutputsImmutableInfoList &outputs_ii,
                                                    const std::vector<KernelTensor *> &ms_inputs,
                                                    const std::vector<KernelTensor *> &ms_outputs) {
  auto ori_split_sizes = ms_inputs[1]->GetValueWithCheck<std::vector<int64_t>>();
  internal::SplitParam param;
  param.split_dim = static_cast<int32_t>(ms_inputs[2]->GetValueWithCheck<int64_t>());
  auto rank = ms_inputs[0]->GetShapeVector().size();
  if (param.split_dim < 0) {
    param.split_dim += static_cast<int32_t>(rank);
  }
  for (const auto size : ori_split_sizes) {
    param.split_sizes.emplace_back(static_cast<uint32_t>(size));
  }

  return internal::CreateSplitOp(inputs_ii, outputs_ii, param, internal::kInternalSplitOpName);
}
MS_INTERNAL_KERNEL_FACTORY_REG(SplitWithSize, internal::kInternalSplitOpName, InternalSplit);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(SplitWithSize, INPUT_NUM_1, INDEX_0);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(SplitWithSize, INTERNEL_KERNEL_IN_OUT_MUTABLE_LENGTH);
}  // namespace kernel
}  // namespace mindspore
