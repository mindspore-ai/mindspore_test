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

#include "plugin/device/ascend/kernel/internal/pyboost/gather.h"

#include <memory>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr AcmeKernelInfoGather::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                   const internal::OutputsImmutableInfoList &outputs) {
  internal::GatherParam param;
  param.axes.emplace_back(axis_);
  param.batch_dims = batch_dims_;
  return internal::CreateGatherOp(inputs, outputs, param, internal::kInternalGatherOpName);
}

void AcmeKernelInfoGather::Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) {
  const auto &input_tensor = input_values[kIndex0]->cast<BaseTensorPtr>();
  const auto &indices_tensor = input_values[kIndex1]->cast<BaseTensorPtr>();
  axis_ = GetValueWithCheck<int64_t>(input_values[kIndex2]);
  batch_dims_ = GetValueWithCheck<int64_t>(input_values[kIndex3]);

  const std::vector<BaseTensorPtr> inputs = {input_tensor, indices_tensor};
  auto op_key = CalcAcmeOpApiHash(kernel_name_, inputs, axis_, batch_dims_);
  CallAcmeOp(op, inputs, op_key);
}
MS_ACME_KERNEL_INFO_FACTORY_REG(Gather, internal::kInternalGatherOpName, AcmeKernelInfoGather);
}  // namespace kernel
}  // namespace mindspore
