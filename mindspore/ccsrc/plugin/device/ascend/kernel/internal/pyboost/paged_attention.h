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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_PAGED_ATTENTION_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_PAGED_ATTENTION_H_

#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class InternalKernelInfoPagedAttention : public InternalKernelInfo {
 public:
  InternalKernelInfoPagedAttention() : InternalKernelInfo(std::move("PagedAttention")) {}
  ~InternalKernelInfoPagedAttention() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) override;

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  inline bool GetSeqLenFromInputTensor(const mindspore::tensor::BaseTensorPtr &input_tensor,
                                       std::vector<int32_t> *seq_len) {
    if (input_tensor == nullptr) {
      return false;
    }

    auto input_tensor_value = static_cast<int32_t *>(input_tensor->data_c());
    auto input_tensor_value_num = input_tensor->Size() / sizeof(int32_t);
    seq_len->clear();
    for (size_t i = 0; i < input_tensor_value_num; i++) {
      (*seq_len).emplace_back(input_tensor_value[i]);
    }
    return true;
  }
  int32_t head_num_;
  int32_t kv_head_num_;
  std::vector<int32_t> kv_seq_len_;
  std::vector<int32_t> q_seq_len_;
  float tor_;
  int32_t kv_cache_quant_mode_;
  bool has_attn_mask_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_PAGED_ATTENTION_H_
