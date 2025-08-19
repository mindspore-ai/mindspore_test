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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MOE_DISTRIBUTE_COMBINE_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MOE_DISTRIBUTE_COMBINE_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include <string>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace moe_distribute_combine {
class MoeDistributeCombineAscend : public AclnnKernelMod {
 public:
  MoeDistributeCombineAscend() : AclnnKernelMod(std::move("aclnnMoeDistributeCombine")) {}
  ~MoeDistributeCombineAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  std::string group_ep_;
  std::string group_tp_;
  int64_t ep_world_size_;
  int64_t ep_rank_id_;
  int64_t moe_expert_num_;
  int64_t tp_world_size_;
  int64_t tp_rank_id_;
  int64_t expert_shard_type_;
  int64_t shard_expert_num_;
  int64_t shared_expert_rank_num_;
  int64_t global_bs_;
  int64_t out_dtype_;
  int64_t common_quant_mode_;
  int64_t group_list_type_;
};
}  // namespace moe_distribute_combine
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MOE_DISTRIBUTE_COMBINE_ACLNN_KERNEL_MOD_H_
