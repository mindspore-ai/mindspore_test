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

#include "plugin/device/ascend/kernel/internal/pyboost/transpose.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoTranspose::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                                  const internal::OutputsImmutableInfoList &outputs) {
  internal::TransposeParam param;
  param.axes = axis_;
  return internal::CreateTransposeOp(inputs, outputs, param, internal::kInternalTransposeOpName);
}

void InternalKernelInfoTranspose::Call(const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &input,
                                       const std::vector<int64_t> &input_perm) {
  std::vector<BaseTensorPtr> inputs = {input};
  std::vector<BaseTensorPtr> outputs = op->outputs();
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);
  axis_ = input_perm;
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, axis_);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(Transpose, internal::kInternalTransposeOpName, InternalKernelInfoTranspose);
}  // namespace kernel
}  // namespace mindspore
