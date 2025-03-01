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

#include "plugin/device/ascend/kernel/internal/dynamic_ntk.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalDynamicNTK::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                         const internal::OutputsImmutableInfoList &outputs_ii,
                                                         const std::vector<KernelTensor *> &ms_inputs,
                                                         const std::vector<KernelTensor *> &ms_outputs) {
  param_.out_type = ms_inputs[kIndex3]->GetValueWithCheck<int64_t>();
  return internal::CreateDynamicNTKOp(inputs_ii, outputs_ii, param_, internal::kInternalDynamicNTKOpName);
}

uint64_t InternalDynamicNTK::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return InternalTilingCache::GenerateKey(kernel_name_, inputs);
}
MS_INTERNAL_KERNEL_FACTORY_REG(DynamicNTK, internal::kInternalDynamicNTKOpName, InternalDynamicNTK);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(DynamicNTK, INPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(DynamicNTK, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
