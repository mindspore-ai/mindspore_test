/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PROMPT_FLASH_ATTENTION_ACLNN_KERNEL_MOD_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PROMPT_FLASH_ATTENTION_ACLNN_KERNEL_MOD_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "ops/base_operator.h"
#include "kernel/ascend/aclnn/kernel_mod_impl/aclnn_kernel_mod.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "infer/ops_func_impl/prompt_flash_attention.h"
#include "utils/llm_manager.h"

namespace mindspore {
using mindspore::device::ascend::FASInputLayoutMode;
namespace kernel {
namespace prompt_flash_attention {
using TensorParams = device::ascend::TensorParams;

class PromptFlashAttentionAscend : public AclnnKernelMod {
 public:
  PromptFlashAttentionAscend() : AclnnKernelMod("aclnnPromptFlashAttentionV3") {}
  ~PromptFlashAttentionAscend() = default;

  void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
    if (outputs[kIndex0]->type_id() != kObjectTypeTensorType) {
      MS_LOG(EXCEPTION) << "now only support tensor type for EmptyKernelTensor in " << op_type_;
    }
    auto &llm_manager = LLMManager::GetInstance();
    llm_manager.add_force_resize_kernel(kernel_name_);
    return true;
  }

 protected:
  DEFINE_GET_WORKSPACE_FOR_RESIZE()
  void SetScalarParam(const std::vector<KernelTensor *> &inputs);
  int64_t num_heads_ = 0;
  double scale_value_d_ = 0;
  int64_t pre_tokens_ = 0;
  int64_t next_tokens_ = 0;
  std::string input_layout_str_ = "";
  int64_t num_key_value_heads_ = 0;
  int64_t sparse_mode_ = 0;
  int64_t inner_precise_ = 0;
  std::pair<std::vector<int64_t>, bool> actual_q_lengths_vector_pair_;
  std::pair<std::vector<int64_t>, bool> actual_kv_lengths_vector_pair_;
  std::pair<std::vector<int64_t>, bool> actual_shared_prefix_lengths_vector_pair_;
};
}  // namespace prompt_flash_attention
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_PROMPT_FLASH_ATTENTION_ACLNN_KERNEL_MOD_H_
