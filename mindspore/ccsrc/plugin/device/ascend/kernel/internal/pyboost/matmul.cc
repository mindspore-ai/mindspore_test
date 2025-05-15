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

#include "plugin/device/ascend/kernel/internal/pyboost/matmul.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoMatMul::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                               const internal::OutputsImmutableInfoList &outputs) {
  internal::MatmulParam param;
  param.transpose_a = transpose_a_;
  param.transpose_b = transpose_b_;
  return internal::CreateMatmulOp(inputs, outputs, param, internal::kInternalMatMulOpName);
}

uint64_t InternalKernelInfoMatMul::GenerateTilingKey(const std::string &kernel_name,
                                                     const std::vector<BaseTensorPtr> &inputs) {
  return CalcInternalOpTilingHash(kernel_name, inputs, output_format_);
}

void InternalKernelInfoMatMul::Call(const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &input_tensor,
                                    const BaseTensorPtr &mat2_tensor, const bool &transpose_a,
                                    const bool &transpose_b) {
  std::vector<BaseTensorPtr> inputs = {input_tensor, mat2_tensor};
  std::vector<BaseTensorPtr> outputs = op->outputs();
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);
  transpose_a_ = transpose_a;
  transpose_b_ = transpose_b;
  auto device_sync = outputs[0]->device_address();
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(device_sync);
  output_format_ = device_address->format();
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, transpose_a_, transpose_b_);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(MatMul, internal::kInternalMatMulOpName, InternalKernelInfoMatMul);
}  // namespace kernel
}  // namespace mindspore
