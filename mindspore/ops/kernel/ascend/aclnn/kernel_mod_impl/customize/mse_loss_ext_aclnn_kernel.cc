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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/mse_loss_ext_aclnn_kernel.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace mse_loss_ext {

void MSELossExtAclnnKernelMod::SetExpandTensor(KernelTensor *input_tensor, const std::vector<KernelTensor *> &inputs,
                                               const size_t &input_index) {
  input_tensor->SetType(inputs[input_index]->GetType());
  input_tensor->SetShape(std::make_shared<abstract::TensorShape>(broadcast_shape_));
}

void MSELossExtAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  expand_indices_.clear();

  const auto &reduction_imm = static_cast<Reduction>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]));
  // transform reduction enum value to corresponding value
  reduction_value_ = ops::ConvertReductionForAclnn(reduction_imm);
  const std::vector<int64_t> &input_shape = inputs[kIndex0]->GetShapeVector();
  const std::vector<int64_t> &target_shape = inputs[kIndex1]->GetShapeVector();
  broadcast_shape_ = ops::CalBroadCastShapeV3(input_shape, target_shape);

  KernelTensor *input_ptr = inputs[kIndex0];
  KernelTensor *target_ptr = inputs[kIndex1];

  if (input_shape != broadcast_shape_) {
    SetExpandTensor(&input_expand_, inputs, kIndex0);
    input_ptr = &input_expand_;
    GetWorkspaceForResizeDoExpandInput(inputs[kIndex0], broadcast_shape_, &input_expand_);
    expand_indices_.emplace_back(kIndex0);
  }
  if (target_shape != broadcast_shape_) {
    SetExpandTensor(&target_expand_, inputs, kIndex1);
    target_ptr = &target_expand_;
    GetWorkspaceForResizeDoExpandTarget(inputs[kIndex1], broadcast_shape_, &target_expand_);
    expand_indices_.emplace_back(kIndex1);
  }

  GetWorkspaceForResizeMSELoss(input_ptr, target_ptr, reduction_value_, outputs[kIndex0]);

  expand_count_ = expand_indices_.size();
  for (size_t i = 0; i < expand_count_; i++) {
    const size_t &output_size =
      ops::CalOutputSize(broadcast_shape_, mindspore::abstract::TypeIdSize(inputs[expand_indices_[i]]->dtype_id()));
    workspace_size_list_.emplace_back(output_size);
  }
}

bool MSELossExtAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  KernelTensor *input_ptr = inputs[kIndex0];
  KernelTensor *target_ptr = inputs[kIndex1];

  for (size_t i = 0; i < expand_count_; i++) {
    KernelTensor *output_ptr;
    if (expand_indices_[i] == kIndex0) {
      input_expand_.set_device_ptr(workspace[workspace.size() - expand_count_ + i]->device_ptr());
      output_ptr = &input_expand_;
      input_ptr = &input_expand_;
      RunOpDoExpandInput(stream_ptr, workspace, inputs[kIndex0], broadcast_shape_, output_ptr);
    } else {
      target_expand_.set_device_ptr(workspace[workspace.size() - expand_count_ + i]->device_ptr());
      output_ptr = &target_expand_;
      target_ptr = &target_expand_;
      RunOpDoExpandTarget(stream_ptr, workspace, inputs[kIndex1], broadcast_shape_, output_ptr);
    }
  }

  RunOpMSELoss(stream_ptr, workspace, input_ptr, target_ptr, reduction_value_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MSELossExt, MSELossExtAclnnKernelMod);
}  // namespace mse_loss_ext
}  // namespace kernel
}  // namespace mindspore
