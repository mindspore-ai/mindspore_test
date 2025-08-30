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

#include "kernel/ascend/internal/quant_batch_matmul.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalQuantBatchMatmul::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                               const internal::OutputsImmutableInfoList &outputs,
                                                               const std::vector<KernelTensor *> &ms_inputs,
                                                               const std::vector<KernelTensor *> &ms_outputs) {
  internal::MatmulParam param;
  param.transpose_a = ms_inputs[kIndex6]->GetValueWithCheck<bool>();
  param.transpose_b = ms_inputs[kIndex7]->GetValueWithCheck<bool>();
  param.with_pertoken_scale = !(ms_inputs[kIndex5]->GetType()->isa<TypeNone>());
  param.with_bias = !(ms_inputs[kIndex4]->GetType()->isa<TypeNone>());
  param.enable_shuffle = false;  // the real definition is in internal
  param.enable_dequant = true;
  bool has_element_type = primitive_->HasAttr("ElemwiseType");
  auto value_str = primitive_->GetAttr("ElemwiseType");
  if (has_element_type && (value_str != nullptr)) {
    if (GetValue<std::string>(value_str) == "fastgelu") {
      param.with_fastgelu = true;
    }
  }
  output_format_ = outputs[0].GetFormat();
  return internal::CreateMatmulOp(inputs, outputs, param, internal::kInternalMatMulOpName);
}

uint64_t InternalQuantBatchMatmul::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, output_format_);
}
MS_INTERNAL_KERNEL_FACTORY_REG(QuantBatchMatmul, internal::kInternalMatMulOpName, InternalQuantBatchMatmul);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(QuantBatchMatmul, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_4, INDEX_2, INDEX_5);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(QuantBatchMatmul, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
