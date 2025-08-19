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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/inplace_uniform_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "abstract/ops/primitive_infer_map.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace inplace_uniform {
namespace {
double GetInputValueToDouble(const std::vector<KernelTensor *> &inputs, const size_t kIndex) {
  double value = 0;
  auto dtype_id = inputs[kIndex]->dtype_id();

  switch (dtype_id) {
    case kNumberTypeBool: {
      value = static_cast<double>(inputs[kIndex]->GetValueWithCheck<bool>());
      break;
    }
    case kNumberTypeFloat32: {
      value = static_cast<double>(inputs[kIndex]->GetValueWithCheck<float>());
      break;
    }
    case kNumberTypeFloat64: {
      value = static_cast<double>(inputs[kIndex]->GetValueWithCheck<double>());
      break;
    }
    case kNumberTypeInt32: {
      value = static_cast<double>(inputs[kIndex]->GetValueWithCheck<int32_t>());
      break;
    }
    case kNumberTypeInt64: {
      value = static_cast<double>(inputs[kIndex]->GetValueWithCheck<int64_t>());
      break;
    }
    default:
      MS_EXCEPTION(TypeError) << "For InplaceUniform, the dtype of input `from` and `to` only support "
                              << "[float, int, bool], but got " << dtype_id << ".";
  }
  return value;
}
}  // namespace

void InplaceUniformAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  from_ = GetInputValueToDouble(inputs, kIndex1);
  to_ = GetInputValueToDouble(inputs, kIndex2);
  seed_ = static_cast<uint64_t>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]));
  offset_ = static_cast<uint64_t>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex4]));

  GetWorkspaceForResize(inputs[kIndex0], from_, to_, seed_, offset_);
}

bool InplaceUniformAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  RunOp(stream_ptr, workspace, inputs[kIndex0], from_, to_, seed_, offset_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(InplaceUniform, InplaceUniformAscend);
}  // namespace inplace_uniform
}  // namespace kernel
}  // namespace mindspore
