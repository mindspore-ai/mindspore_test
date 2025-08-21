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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/muls_aclnn_kernel.h"
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace muls {

void MulsAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  auto other_scalar_dtype_id = inputs[kIndex1]->dtype_id();
  switch (other_scalar_dtype_id) {
    case kNumberTypeBool: {
      auto other_scalar_value = inputs[kIndex1]->GetValueWithCheck<bool>();
      MAKE_SCALAR(other_scalar_value, other_scalar_dtype_id, this->other_scalar_);
      break;
    }
    case kNumberTypeFloat32: {
      double other_scalar_value = static_cast<double>((inputs[kIndex1]->GetValueWithCheck<float>()));
      MAKE_SCALAR(other_scalar_value, kNumberTypeFloat64, this->other_scalar_);
      break;
    }
    case kNumberTypeFloat64: {
      auto other_scalar_value = inputs[kIndex1]->GetValueWithCheck<double>();
      MAKE_SCALAR(other_scalar_value, other_scalar_dtype_id, this->other_scalar_);
      break;
    }
    case kNumberTypeInt32: {
      auto other_scalar_value = inputs[kIndex1]->GetValueWithCheck<int32_t>();
      MAKE_SCALAR(other_scalar_value, other_scalar_dtype_id, this->other_scalar_);
      break;
    }
    case kNumberTypeInt64: {
      auto other_scalar_value = inputs[kIndex1]->GetValueWithCheck<int64_t>();
      MAKE_SCALAR(other_scalar_value, other_scalar_dtype_id, this->other_scalar_);
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "AddExt only support bool, float32, float64, int32 and int64, but got "
                        << other_scalar_dtype_id;
  }
  GetWorkspaceForResize(inputs[kIndex0], this->other_scalar_, outputs[kIndex0]);
}

bool MulsAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], this->other_scalar_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Muls, MulsAscend);
}  // namespace muls
}  // namespace kernel
}  // namespace mindspore
