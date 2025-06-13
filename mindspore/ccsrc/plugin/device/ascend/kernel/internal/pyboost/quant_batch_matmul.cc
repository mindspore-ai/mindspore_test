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

#include "plugin/device/ascend/kernel/internal/pyboost/quant_batch_matmul.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr QuantBatchMatmul::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                       const internal::OutputsImmutableInfoList &outputs) {
  internal::MatmulParam param;
  param.transpose_a = transpose_a_;
  param.transpose_b = transpose_b_;
  param.with_bias = with_bias_;
  param.enable_shuffle = false;
  param.enable_dequant = true;
  output_format_ = outputs[0].GetFormat();
  return internal::CreateMatmulOp(inputs, outputs, param, internal::kInternalMatMulOpName);
}

uint64_t QuantBatchMatmul::GenerateTilingKey(const std::string &kernel_name, const std::vector<BaseTensorPtr> &inputs) {
  return CalcInternalOpTilingHash(kernel_name, inputs, output_format_);
}

void QuantBatchMatmul::Call(const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &x,
                            const BaseTensorPtr &y, const BaseTensorPtr &scale,
                            const std::optional<BaseTensorPtr> &offset, const std::optional<BaseTensorPtr> &bias,
                            const std::optional<BaseTensorPtr> &pertoken_scale, const bool transpose_a,
                            const bool transpose_b, const int64_t dtype) {
  std::vector<BaseTensorPtr> inputs = {x, y, bias.has_value() ? bias.value() : nullptr, scale};
  BaseTensorPtrList outputs = op->outputs();
  transpose_a_ = transpose_a;
  transpose_b_ = transpose_b;
  with_bias_ = bias.has_value();
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);

  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, transpose_a_, transpose_b_, outputs);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(QuantBatchMatmul, internal::kInternalMatMulOpName, QuantBatchMatmul);
}  // namespace kernel
}  // namespace mindspore
