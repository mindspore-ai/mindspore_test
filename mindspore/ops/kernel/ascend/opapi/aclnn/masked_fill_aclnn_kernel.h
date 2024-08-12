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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MASKED_FILL_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MASKED_FILL_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <utility>
#include <string>
#include "ops/base_operator.h"
#include "kernel/ascend/opapi/aclnn_kernel_mod.h"
#include "transform/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {

class MaskedFillAscend : public AclnnKernelMod {
 public:
  MaskedFillAscend() : AclnnKernelMod(std::move("aclnnInplaceMaskedFillTensor")) {}
  ~MaskedFillAscend() = default;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()

  void SetWorkspaceForInplaceCopy(const KernelTensor *output, const KernelTensor *input) {
    copy_hash_id_ = transform::CalcOpApiHash(inplace_copy_str_, input);
    if (cache_hash_.count(copy_hash_id_) == 0) {
      auto return_value = GEN_EXECUTOR_CUST(inplace_copy_str_, output, input);
      UpdateInplacemWorkspace(std::get<kWsSizeIndex>(return_value), false);
    } else {
      auto return_value = GEN_EXECUTOR_BOOST(inplace_copy_str_, copy_hash_id_, output, input);
      UpdateInplacemWorkspace(std::get<kWsSizeIndex>(return_value), true, std::get<kHashIdIndex>(return_value));
    }
  }

  inline void UpdateInplacemWorkspace(uint64_t ws_size, bool boost, uint64_t new_hash_id = 0) {
    copy_ws_size_ = ws_size;
    if (copy_ws_size_ != 0) {
      workspace_size_list_.emplace_back(ws_size);
    }

    if (boost) {
      copy_hash_id_ = new_hash_id;
    }
  }

  const std::string inplace_copy_str_{"aclnnInplaceCopy"};
  bool copy_ws_size_{0};
  uint64_t copy_hash_id_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_MASKED_FILL_ACLNN_KERNEL_MOD_H_
