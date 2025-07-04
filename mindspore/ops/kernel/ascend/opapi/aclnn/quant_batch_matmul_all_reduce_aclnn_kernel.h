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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_QUANT_BATCH_MATMUL_ALL_REDUCE_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_QUANT_BATCH_MATMUL_ALL_REDUCE_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <string>
#include <utility>
#include "ops/base_operator.h"
#include "kernel/ascend/opapi/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace quant_batch_matmul_all_reduce {

class QuantBatchMatmulAllReduceAscend : public AclnnKernelMod {
 public:
  QuantBatchMatmulAllReduceAscend() : AclnnKernelMod(std::move("aclnnQuantMatmulAllReduceV2")) {}
  ~QuantBatchMatmulAllReduceAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  void InitializeCommonAttributes();
  std::pair<KernelTensor *, bool> input_a_;
  std::pair<KernelTensor *, bool> input_b_;
  bool trans_a_;
  bool trans_b_;
  std::string group_;
  std::string hccl_inner_comm_name_;
  int64_t comm_turn_ = 0;
  std::string reduce_op_;
  int64_t stream_mode_ = 1;
};
}  // namespace quant_batch_matmul_all_reduce
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_QUANT_BATCH_MATMUL_ALL_REDUCE_ACLNN_KERNEL_MOD_H_
