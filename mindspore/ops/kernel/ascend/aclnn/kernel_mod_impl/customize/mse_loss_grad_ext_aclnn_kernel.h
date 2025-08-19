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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MSE_LOSS_GRAD_EXT_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MSE_LOSS_GRAD_EXT_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include <string>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace mse_loss_grad_ext {
using TensorParams = device::ascend::TensorParams;

class MSELossGradExtAclnnKernelMod : public AclnnKernelMod {
 public:
  MSELossGradExtAclnnKernelMod() : AclnnKernelMod("aclnnMseLossBackward") {}
  ~MSELossGradExtAclnnKernelMod() = default;

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnExpand, DoExpandInput)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnExpand, DoExpandTarget)
  DEFINE_GET_WORKSPACE_FOR_OPS(aclnnMseLossBackward, MSELossGrad)

  void SetExpandTensor(KernelTensor *input_tensor, const std::vector<KernelTensor *> &inputs,
                       const size_t &input_index);

  const std::string mse_loss_grad_ext_do_expand_input{"aclnnExpand"};
  const std::string mse_loss_grad_ext_do_expand_target{"aclnnExpand"};
  size_t expand_count_{0};
  std::vector<size_t> expand_indices_{};
  std::vector<int64_t> broadcast_shape_{};

  KernelTensor input_expand_;
  KernelTensor target_expand_;

  int64_t reduction_value_{0};
};

}  // namespace mse_loss_grad_ext
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MSE_LOSS_GRAD_EXT_ACLNN_KERNEL_MOD_H_
