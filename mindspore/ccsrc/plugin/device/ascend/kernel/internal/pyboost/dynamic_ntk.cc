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

#include "plugin/device/ascend/kernel/internal/pyboost/dynamic_ntk.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr DynamicNTK::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                 const internal::OutputsImmutableInfoList &outputs) {
  internal::DynamicNTKParam param;
  if (dtype_ == kNumberTypeFloat16) {
    param.out_type = static_cast<int64_t>(DynamicNTKOutType::Float16);
  } else if (dtype_ == kNumberTypeBFloat16) {
    param.out_type = static_cast<int64_t>(DynamicNTKOutType::BFloat16);
  } else if (dtype_ == kNumberTypeFloat32) {
    param.out_type = static_cast<int64_t>(DynamicNTKOutType::Float32);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported output type: " << dtype_;
  }
  return internal::CreateDynamicNTKOp(inputs, outputs, param, internal::kInternalDynamicNTKOpName);
}

void DynamicNTK::Call(const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &position_ids_tensor,
                      const BaseTensorPtr &inv_freq_tensor, const BaseTensorPtr &seq_lens_tensor, const TypeId &dtype) {
  BaseTensorPtrList inputs = {position_ids_tensor, inv_freq_tensor, seq_lens_tensor};
  BaseTensorPtrList outputs = op->outputs();
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);
  dtype_ = dtype;
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, dtype_, outputs);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(DynamicNTK, internal::kInternalDynamicNTKOpName, DynamicNTK);
}  // namespace kernel
}  // namespace mindspore
