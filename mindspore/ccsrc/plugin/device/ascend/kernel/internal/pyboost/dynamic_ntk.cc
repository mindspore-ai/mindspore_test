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
  return internal::CreateDynamicNTKOp(inputs, outputs, param_, internal::kInternalDynamicNTKOpName);
}

void DynamicNTK::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
                      const BaseTensorPtr &position_ids_tensor, const BaseTensorPtr &inv_freq_tensor,
                      const BaseTensorPtr &seq_lens_tensor, const TypeId &dtype) {
  BaseTensorPtrList inputs = {position_ids_tensor, inv_freq_tensor, seq_lens_tensor};
  BaseTensorPtrList outputs = op->outputs();
  TransInternalShapes(inputs, outputs);

  if (dtype == kNumberTypeFloat16) {
    param_.out_type = static_cast<int64_t>(DynamicNTKOutType::Float16);
  } else if (dtype == kNumberTypeBFloat16) {
    param_.out_type = static_cast<int64_t>(DynamicNTKOutType::BFloat16);
  } else if (dtype == kNumberTypeFloat32) {
    param_.out_type = static_cast<int64_t>(DynamicNTKOutType::Float32);
  } else {
    MS_LOG(EXCEPTION) << "Unsupported output type: " << dtype;
  }

  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(DynamicNTK, DynamicNTK);
}  // namespace kernel
}  // namespace mindspore
