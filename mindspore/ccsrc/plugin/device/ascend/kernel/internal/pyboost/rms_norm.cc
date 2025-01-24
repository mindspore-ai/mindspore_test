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

#include "plugin/device/ascend/kernel/internal/pyboost/rms_norm.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoRmsNorm::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                                const internal::OutputsImmutableInfoList &outputs) {
  internal::NormParam param;
  param.eps = eps_;
  return internal::CreateRmsNormOp(inputs, outputs, param, internal::kInternalRmsNormOpName);
}

void InternalKernelInfoRmsNorm::Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) {
  GetInputAndOutputIndex(op, input_values);
  std::vector<BaseTensorPtr> inputs;
  std::vector<BaseTensorPtr> outputs;
  Init(input_values, inputs, outputs, op->outputs());
  eps_ = GetValueWithCheck<float>(input_values[kIndex2]);
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, eps_);
  GetOrCreateKernel(inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(RmsNorm, internal::kInternalRmsNormOpName, InternalKernelInfoRmsNorm);
}  // namespace kernel
}  // namespace mindspore
