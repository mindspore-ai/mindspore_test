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

#include "plugin/device/ascend/kernel/internal/pyboost/quant_batch_matmul.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr AcmeKernelInfoQuantBatchMatmul::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                             const internal::OutputsImmutableInfoList &outputs) {
  internal::MatmulParam param;
  param.transpose_a = transpose_a_;
  param.transpose_b = transpose_b_;
  param.with_bias = has_bias_;
  param.enable_shuffle = false;  // the real definition is in acme
  param.enable_dequant = true;
  return internal::CreateMatmulOp(inputs, outputs, param, internal::kInternalMatMulOpName);
}

void AcmeKernelInfoQuantBatchMatmul::Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) {
  const BaseTensorPtr &x1_tensor = input_values[kIndex0]->cast<BaseTensorPtr>();
  const BaseTensorPtr &x2_tensor = input_values[kIndex1]->cast<BaseTensorPtr>();
  const BaseTensorPtr &scale_tensor = input_values[kIndex2]->cast<BaseTensorPtr>();
  const BaseTensorPtr &offset_tensor = input_values[kIndex3] == mindspore::kNone ? nullptr : input_values[kIndex3]->cast<BaseTensorPtr>();
  const BaseTensorPtr &bias_tensor = input_values[kIndex4] == mindspore::kNone ? nullptr : input_values[kIndex4]->cast<BaseTensorPtr>();
  const BaseTensorPtr &per_token_scale_tensor = input_values[kIndex5] == mindspore::kNone ? nullptr : input_values[kIndex5]->cast<BaseTensorPtr>();
  transpose_a_ = GetValueWithCheck<bool>(input_values[kIndex6]);
  transpose_b_ = GetValueWithCheck<bool>(input_values[kIndex7]);
  has_bias_ = input_values[kIndex4] != mindspore::kNone;
  
  const std::vector<BaseTensorPtr> inputs = {x1_tensor, x2_tensor, scale_tensor, offset_tensor, bias_tensor, per_token_scale_tensor};
  auto op_key = CalcAcmeOpApiHash(kernel_name_, inputs, transpose_a_, transpose_b_, has_bias_);
  CallAcmeOp(op, inputs, op_key);
}
MS_ACME_KERNEL_INFO_FACTORY_REG(QuantBatchMatmul, internal::kInternalMatMulOpName, AcmeKernelInfoQuantBatchMatmul);
}  // namespace kernel
}  // namespace mindspore
