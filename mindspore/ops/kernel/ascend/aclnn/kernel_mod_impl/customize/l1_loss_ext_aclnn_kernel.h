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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_L1_LOSS_EXT_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_L1_LOSS_EXT_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace l1_loss_ext {
using TensorParams = device::ascend::TensorParams;

class L1LossExtAclnnKernelMod : public AclnnKernelMod {
 public:
  L1LossExtAclnnKernelMod() : AclnnKernelMod(std::move("aclnnL1Loss")) {}
  ~L1LossExtAclnnKernelMod() = default;

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnExpand, ExpandInput)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnExpand, ExpandTarget)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnL1Loss, L1LossExt)

  KernelTensor input_expand_;
  KernelTensor target_expand_;

  std::vector<size_t> expand_indices_{};
  ShapeVector broadcast_shape_{};

  int64_t reduction_{1};
};

}  // namespace l1_loss_ext
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_L1_LOSS_EXT_ACLNN_KERNEL_MOD_H_
