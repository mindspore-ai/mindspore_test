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
#include "kernel/ascend/opapi/aclnn/unique_consecutive_aclnn_kernel.h"
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void UniqueConsecutiveAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  auto return_inverse = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex1]);
  auto return_counts = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex2]);

  constexpr int64_t NoneN = 1000;
  auto dim_value_opt = inputs[kIndex3]->GetOptionalValueWithCheck<int64_t>();
  dim_ = dim_value_opt.has_value() ? dim_value_opt.value() : NoneN;

  GetWorkspaceForResize(inputs[kIndex0], return_inverse, return_counts, dim_, outputs[kIndex0], outputs[kIndex1],
                        outputs[kIndex2]);
}

bool UniqueConsecutiveAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run UniqueConsecutive start.";

  auto return_inverse = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex1]);
  auto return_counts = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex2]);

  auto res = GEN_EXECUTOR_CUST(op_type_, inputs[kIndex0], return_inverse, return_counts, dim_, outputs[kIndex0],
                               outputs[kIndex1], outputs[kIndex2]);
  UpdateWorkspace(res);
  executor_ = std::get<kIndex1>(res);
  auto &all_acl_tensor = std::get<kIndex2>(res);
  RunOpSync(stream_ptr, workspace);
  MS_LOG(DEBUG) << "Run UniqueConsecutive end.";

  // update output shape
  size_t output_size = 3;
  output_shapes_.resize(output_size);
  output_shapes_[kIndex0] = device::ascend::UpdateOutputShape(all_acl_tensor.get<kIndex4>());
  output_shapes_[kIndex1] = device::ascend::UpdateOutputShape(all_acl_tensor.get<kIndex5>());
  // For scalar tensor input, shape of return_inverse should be '{}', otherwise ACLNN returns a weird empty shape.
  const auto &input_tensor_shape = inputs[kIndex0]->GetShape()->GetShapeVector();
  if (input_tensor_shape.empty()) {
    output_shapes_[kIndex1].clear();
  }

  output_shapes_[kIndex2] = device::ascend::UpdateOutputShape(all_acl_tensor.get<kIndex6>());

  MS_LOG(DEBUG) << "Run UniqueConsecutive end.";
  return true;
}

void UniqueConsecutiveAscend::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  for (size_t i = 0; i < output_shapes_.size(); ++i) {
    outputs[i]->SetShapeVector(output_shapes_[i]);
    size_t dtype_byte = GetTypeByte(TypeIdToType(outputs[i]->dtype_id()));
    size_t update_size = LongToSize(
      std::accumulate(output_shapes_[i].begin(), output_shapes_[i].end(), dtype_byte, std::multiplies<int64_t>()));
    outputs[i]->set_size(update_size);
  }
}

MS_ACLNN_KERNEL_FACTORY_REG(UniqueConsecutive, UniqueConsecutiveAscend);
}  // namespace kernel
}  // namespace mindspore
