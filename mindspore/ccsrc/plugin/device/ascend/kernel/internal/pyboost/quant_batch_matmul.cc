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
  param_.enable_shuffle = false;
  param_.enable_dequant = true;
  return internal::CreateMatmulOp(inputs, outputs, param_, internal::kInternalMatMulOpName);
}

uint64_t QuantBatchMatmul::GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const {
  return CalcInternalOpTilingHash(kernel_name_, tiling_key, output_format_);
}

void QuantBatchMatmul::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                            const uint64_t &tiling_key, const BaseTensorPtr &x, const BaseTensorPtr &y,
                            const BaseTensorPtr &scale, const std::optional<BaseTensorPtr> &offset,
                            const std::optional<BaseTensorPtr> &bias,
                            const std::optional<BaseTensorPtr> &pertoken_scale, const bool transpose_a,
                            const bool transpose_b, const int64_t dtype) {
  std::vector<BaseTensorPtr> inputs = {x, y, bias.has_value() ? bias.value() : nullptr, scale};
  BaseTensorPtrList outputs = op->outputs();
  TransInternalShapes(inputs, outputs);

  param_.transpose_a = transpose_a;
  param_.transpose_b = transpose_b;
  param_.with_bias = bias.has_value();
  auto device_sync = outputs[kIndex0]->device_address();
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  MS_EXCEPTION_IF_NULL(device_address);
  output_format_ = TransInternalFormat(GetFormatFromStrToEnum(device_address->format()));

  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(QuantBatchMatmul, QuantBatchMatmul);
}  // namespace kernel
}  // namespace mindspore
