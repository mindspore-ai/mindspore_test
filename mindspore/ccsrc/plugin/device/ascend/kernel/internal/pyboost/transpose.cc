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

#include "plugin/device/ascend/kernel/internal/pyboost/transpose.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoTranspose::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                      const internal::OutputsImmutableInfoList &outputs) {
  internal::TransposeParam param;
  param.axes = axis_;
  return internal::CreateTransposeOp(inputs, outputs, param, internal::kInternalTransposeOpName);
}

void InternalKernelInfoTranspose::Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) {
  GetInputAndOutputIndex(op, input_values);
  std::vector<BaseTensorPtr> inputs;
  std::vector<BaseTensorPtr> outputs;
  Init(input_values, inputs, outputs, op->outputs());
  axis_ = GetValueWithCheck<std::vector<int64_t>>(input_values[kIndex1]);
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, axis_);
  GetOrCreateKernel(inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(Transpose, internal::kInternalTransposeOpName, InternalKernelInfoTranspose);
}  // namespace kernel
}  // namespace mindspore
