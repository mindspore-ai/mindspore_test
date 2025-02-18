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
#include "kernel/ascend/opapi/aclnn/dropout_gen_mask_ext_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
void DropoutGenMaskExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);
  if (primitive_->HasAttr("enable_keep_prob") && GetValue<bool>(primitive_->GetAttr("enable_keep_prob"))) {
    auto keep_prob_dtype = inputs[kIndex1]->dtype_id();
    if (keep_prob_dtype == kNumberTypeFloat16) {
      p_value_ = static_cast<double>(1.0) - static_cast<double>(inputs[kIndex1]->GetValueWithCheck<float16>());
    } else if (keep_prob_dtype == kNumberTypeBFloat16) {
      p_value_ = static_cast<double>(1.0) - static_cast<double>(inputs[kIndex1]->GetValueWithCheck<bfloat16>());
    } else if (keep_prob_dtype == kNumberTypeFloat32) {
      p_value_ = static_cast<double>(1.0) - static_cast<double>(inputs[kIndex1]->GetValueWithCheck<float>());
    } else {
      MS_LOG(EXCEPTION)
        << "For 'DropoutGenMaskExt', the 'keep_prob' type must be in [Float16, BFloat16, Float32], but got "
        << TypeIdToString(keep_prob_dtype);
    }
  } else {
    p_value_ = static_cast<double>(inputs[kIndex1]->GetValueWithCheck<float>());
  }

  MS_LOG(DEBUG) << primitive_->name() << " got a (" + TypeIdToString(inputs[kIndex1]->dtype_id()) + ")p = " << p_value_;

  shape_ = device::ascend::ConvertKernelTensor<ShapeVector>(inputs[kIndex0]);
  seed_value_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  offset_value_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  dtype_value_ = static_cast<TypeId>(inputs[kIndex4]->GetValueWithCheck<int64_t>());

  GetWorkspaceForResize(shape_, p_value_, seed_value_, offset_value_, dtype_value_, outputs[kIndex0]);
}

bool DropoutGenMaskExtAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, shape_, p_value_, seed_value_, offset_value_, dtype_value_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(DropoutGenMaskExt, DropoutGenMaskExtAscend);
}  // namespace kernel
}  // namespace mindspore
