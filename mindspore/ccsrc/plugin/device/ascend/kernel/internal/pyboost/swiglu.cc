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

#include "plugin/device/ascend/kernel/internal/pyboost/swiglu.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoSwiglu::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                               const internal::OutputsImmutableInfoList &outputs) {
  internal::SwiGLUParam param;
  param.axis = -1;
  return internal::CreateSwiGLUOp(inputs, outputs, param, internal::kInternalSwiGLUOpName);
}

void InternalKernelInfoSwiglu::Call(const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &input,
                                    const int64_t &dim) {
  std::vector<BaseTensorPtr> inputs = {input};
  std::vector<BaseTensorPtr> outputs = op->outputs();
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);
  dim_ = dim;
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, dim_);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(Swiglu, internal::kInternalSwiGLUOpName, InternalKernelInfoSwiglu);
}  // namespace kernel
}  // namespace mindspore
