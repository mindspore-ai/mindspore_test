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

#include "kernel/ascend/internal/quant_linear_sparse.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalQuantLinearSparse::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                                const internal::OutputsImmutableInfoList &outputs_ii,
                                                                const std::vector<KernelTensor *> &ms_inputs,
                                                                const std::vector<KernelTensor *> &ms_outputs) {
  output_format_ = outputs_ii[0].GetFormat();
  return internal::CreateQuantLinearSparseOp(inputs_ii, outputs_ii, internal::kInternalQuantLinearSparseOpName);
}

uint64_t InternalQuantLinearSparse::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, output_format_);
}
MS_INTERNAL_KERNEL_FACTORY_REG(QuantLinearSparse, internal::kInternalQuantLinearSparseOpName,
                               InternalQuantLinearSparse);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantLinearSparse, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_4, INDEX_2, INDEX_3);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantLinearSparse, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
