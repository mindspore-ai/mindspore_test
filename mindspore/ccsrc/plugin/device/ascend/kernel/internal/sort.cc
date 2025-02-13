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

#include "plugin/device/ascend/kernel/internal/sort.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalSort::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                     const internal::OutputsImmutableInfoList &outputs_ii,
                                                     const std::vector<KernelTensor *> &ms_inputs,
                                                     const std::vector<KernelTensor *> &ms_outputs) {
  internal::SortParam param;
  auto k = ms_inputs[1]->GetValueWithCheck<int64_t>();
  param.num = {static_cast<int32_t>(k)};
  return internal::CreateSortOp(inputs_ii, outputs_ii, param, internal::kInternalSortOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(TopK, internal::kInternalSortOpName, InternalSort);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(TopK, INPUT_NUM_1, INDEX_0);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(TopK, OUTPUT_NUM_2, INDEX_0, INDEX_1);

// MS_INTERNAL_KERNEL_FACTORY_REG(TopkExt, internal::kInternalSortOpName, InternalSort);
// REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(TopkExt, INPUT_NUM_1, INDEX_0);
// REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(TopkExt, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
