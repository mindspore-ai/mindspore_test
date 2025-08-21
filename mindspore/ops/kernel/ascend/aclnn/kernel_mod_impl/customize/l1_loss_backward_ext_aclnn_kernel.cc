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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/l1_loss_backward_ext_aclnn_kernel.h"
#include <memory>
#include <vector>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace l1_loss_backward_ext {
void L1LossBackwardExtAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  expand_indices_.clear();

  const auto &reduction_imm = static_cast<Reduction>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]));
  // transform reduction enum value to corresponding value
  reduction_ = ops::ConvertReductionForAclnn(reduction_imm);

  const auto &input_shape = inputs[kIndex1]->GetShapeVector();
  const auto &target_shape = inputs[kIndex2]->GetShapeVector();

  MS_EXCEPTION_IF_NULL(primitive_);
  broadcast_shape_ = ops::CalBroadCastShapeV3(input_shape, target_shape);

  auto input = inputs[kIndex1];
  auto target = inputs[kIndex2];

  if (input_shape != broadcast_shape_) {
    input_expand_.SetType(inputs[kIndex1]->GetType());
    input_expand_.SetShape(std::make_shared<abstract::TensorShape>(broadcast_shape_));
    input = &input_expand_;
    GetWorkspaceForResizeExpandInput(inputs[kIndex1], broadcast_shape_, &input_expand_);
    expand_indices_.emplace_back(kIndex1);
  }
  if (target_shape != broadcast_shape_) {
    target_expand_.SetType(inputs[kIndex2]->GetType());
    target_expand_.SetShape(std::make_shared<abstract::TensorShape>(broadcast_shape_));
    target = &target_expand_;
    GetWorkspaceForResizeExpandTarget(inputs[kIndex2], broadcast_shape_, &target_expand_);
    expand_indices_.emplace_back(kIndex2);
  }
  GetWorkspaceForResizeL1LossBackwardExt(inputs[kIndex0], input, target, reduction_, outputs[kIndex0]);

  for (size_t idx = 0; idx < expand_indices_.size(); ++idx) {
    const auto &output_size =
      ops::CalOutputSize(broadcast_shape_, mindspore::abstract::TypeIdSize(inputs[expand_indices_[idx]]->dtype_id()));
    workspace_size_list_.emplace_back(output_size);
  }
}

bool L1LossBackwardExtAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto input = inputs[kIndex1];
  auto target = inputs[kIndex2];

  for (size_t idx = 0; idx < expand_indices_.size(); ++idx) {
    size_t workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(expand_indices_.size())) + idx;
    if (expand_indices_[idx] == kIndex1) {
      input_expand_.set_device_ptr(workspace[workspace_offset]->device_ptr());
      input = &input_expand_;
      RunOpExpandInput(stream_ptr, workspace, inputs[kIndex1], broadcast_shape_, &input_expand_);
    } else {
      target_expand_.set_device_ptr(workspace[workspace_offset]->device_ptr());
      target = &target_expand_;
      RunOpExpandTarget(stream_ptr, workspace, inputs[kIndex2], broadcast_shape_, &target_expand_);
    }
  }
  RunOpL1LossBackwardExt(stream_ptr, workspace, inputs[kIndex0], input, target, reduction_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(L1LossBackwardExt, L1LossBackwardExtAclnnKernelMod);
}  // namespace l1_loss_backward_ext
}  // namespace kernel
}  // namespace mindspore
