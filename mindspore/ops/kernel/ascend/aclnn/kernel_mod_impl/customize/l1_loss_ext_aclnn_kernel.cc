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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/l1_loss_ext_aclnn_kernel.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/types.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace l1_loss_ext {
void L1LossExtAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  expand_indices_.clear();

  const auto &reduction_imm = static_cast<Reduction>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]));
  // transform reduction enum value to corresponding value
  reduction_ = ops::ConvertReductionForAclnn(reduction_imm);

  const auto &input_shape = inputs[kIndex0]->GetShapeVector();
  const auto &target_shape = inputs[kIndex1]->GetShapeVector();

  MS_EXCEPTION_IF_NULL(primitive_);
  broadcast_shape_ = ops::CalBroadCastShapeV3(input_shape, target_shape);

  auto input = inputs[kIndex0];
  auto target = inputs[kIndex1];

  if (input_shape != broadcast_shape_) {
    input_expand_.SetType(inputs[kIndex0]->GetType());
    input_expand_.SetShape(std::make_shared<abstract::TensorShape>(broadcast_shape_));
    input = &input_expand_;
    GetWorkspaceForResizeExpandInput(inputs[kIndex0], broadcast_shape_, &input_expand_);
    expand_indices_.emplace_back(kIndex0);
  }
  if (target_shape != broadcast_shape_) {
    target_expand_.SetType(inputs[kIndex1]->GetType());
    target_expand_.SetShape(std::make_shared<abstract::TensorShape>(broadcast_shape_));
    target = &target_expand_;
    GetWorkspaceForResizeExpandTarget(inputs[kIndex1], broadcast_shape_, &target_expand_);
    expand_indices_.emplace_back(kIndex1);
  }
  GetWorkspaceForResizeL1LossExt(input, target, reduction_, outputs[kIndex0]);

  for (size_t idx = 0; idx < expand_indices_.size(); ++idx) {
    const auto &output_size =
      ops::CalOutputSize(broadcast_shape_, mindspore::abstract::TypeIdSize(inputs[expand_indices_[idx]]->dtype_id()));
    workspace_size_list_.emplace_back(output_size);
  }
}

bool L1LossExtAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto input = inputs[kIndex0];
  auto target = inputs[kIndex1];

  for (size_t idx = 0; idx < expand_indices_.size(); ++idx) {
    size_t workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(expand_indices_.size())) + idx;
    if (expand_indices_[idx] == kIndex0) {
      input_expand_.set_device_ptr(workspace[workspace_offset]->device_ptr());
      input = &input_expand_;
      RunOpExpandInput(stream_ptr, workspace, inputs[kIndex0], broadcast_shape_, &input_expand_);
    } else {
      target_expand_.set_device_ptr(workspace[workspace_offset]->device_ptr());
      target = &target_expand_;
      RunOpExpandTarget(stream_ptr, workspace, inputs[kIndex1], broadcast_shape_, &target_expand_);
    }
  }
  RunOpL1LossExt(stream_ptr, workspace, input, target, reduction_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(L1LossExt, L1LossExtAclnnKernelMod);
}  // namespace l1_loss_ext
}  // namespace kernel
}  // namespace mindspore
