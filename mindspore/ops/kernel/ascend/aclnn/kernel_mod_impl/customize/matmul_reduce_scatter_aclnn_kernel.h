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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MATMUL_REDUCE_SCATTER_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MATMUL_REDUCE_SCATTER_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <string>
#include <utility>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace matmul_reduce_scatter {
class MatmulReduceScatterAscend : public AclnnKernelMod {
 public:
  MatmulReduceScatterAscend() : AclnnKernelMod(std::move("aclnnMatmulReduceScatter")) {}
  ~MatmulReduceScatterAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  std::pair<KernelTensor *, bool> input_;
  std::pair<KernelTensor *, bool> x2_;
  std::string group_;
  int64_t world_size_ = 0;
  std::string hccl_inner_comm_name_;
  std::string reduce_op_;
  int64_t comm_turn_ = 0;
  bool trans_input_ = false;
  bool trans_x2_ = false;
  int64_t stream_mode_ = 1;
};
}  // namespace matmul_reduce_scatter
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MATMUL_REDUCE_SCATTER_ACLNN_KERNEL_MOD_H_
