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

#include "plugin/device/ascend/kernel/internal/grouped_matmul.h"

#include <memory>
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalGroupedMatmul::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                            const internal::OutputsImmutableInfoList &outputs,
                                                            const std::vector<KernelTensor *> &ms_inputs,
                                                            const std::vector<KernelTensor *> &ms_outputs) {
  internal::MatmulParam param;
  param.transpose_a = ms_inputs[kIndex10]->GetValueWithCheck<bool>();
  param.transpose_b = ms_inputs[kIndex11]->GetValueWithCheck<bool>();
  param.with_bias = !(ms_inputs[kIndex2]->GetType()->isa<TypeNone>());
  param.enable_shuffle = false;  // the real definition is in internal
  param.enable_dequant = (ms_inputs[kIndex0]->GetType() == kInt8);
  output_format_ = outputs[0].GetFormat();
  return internal::CreateGroupedMatmulOp(inputs, outputs, param, internal::kInternalGroupedMatmulOpName);
}

uint64_t InternalGroupedMatmul::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, output_format_);
}

// MS_INTERNAL_KERNEL_FACTORY_REG(GroupedMatmul, internal::kInternalGroupedMatmulOpName, InternalGroupedMatmul);
// REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(GroupedMatmul, INPUT_NUM_8, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4,
//                                      INDEX_5, INDEX_6, INDEX_7);
// REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(GroupedMatmul, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
