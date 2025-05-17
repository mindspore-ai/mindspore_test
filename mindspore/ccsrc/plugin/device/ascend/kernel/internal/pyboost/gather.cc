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

#include "plugin/device/ascend/kernel/internal/pyboost/gather.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoGather::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                               const internal::OutputsImmutableInfoList &outputs) {
  internal::GatherParam param;
  param.axes.emplace_back(axis_);
  param.batch_dims = batch_dims_;
  return internal::CreateGatherOp(inputs, outputs, param, internal::kInternalGatherOpName);
}

void InternalKernelInfoGather::Call(const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &input_tensor,
                                    const BaseTensorPtr &input_indices, const int64_t &axis,
                                    const int64_t &batch_dims) {
  std::vector<BaseTensorPtr> inputs = {input_tensor, input_indices};
  std::vector<BaseTensorPtr> outputs = op->outputs();
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);
  axis_ = axis;
  batch_dims_ = batch_dims;
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, axis_, batch_dims_);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(Gather, internal::kInternalGatherOpName, InternalKernelInfoGather);
}  // namespace kernel
}  // namespace mindspore
