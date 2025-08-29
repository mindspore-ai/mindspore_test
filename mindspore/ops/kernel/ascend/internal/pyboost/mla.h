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
#ifndef MINDSPORE_CCSRC_PLUGIN_KERNEL_INTERNAL_PYBOOST_MLA_H_
#define MINDSPORE_CCSRC_PLUGIN_KERNEL_INTERNAL_PYBOOST_MLA_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "kernel/ascend/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class Mla : public InternalKernelInfo {
 public:
  explicit Mla(std::string &&kernel_name) : InternalKernelInfo(std::move(kernel_name)) {}
  ~Mla() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
            const TensorPtr &query, const TensorPtr &q_rope, const TensorPtr &kv_cache, const TensorPtr &k_rope,
            const TensorPtr &block_tables, const std::optional<TensorPtr> &mask,
            const std::optional<TensorPtr> &deq_scale_qk, const std::optional<TensorPtr> &deq_scale_pv,
            const std::optional<TensorPtr> &q_seq_lens, const std::optional<TensorPtr> &context_lens,
            const int64_t &head_num, const float &scale_value, const int64_t &kv_head_num, const int64_t &mask_mode,
            const int64_t &is_ring);

 protected:
  uint64_t GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const override;
  bool UpdateParam() override;
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  internal::MLAParam param_;
  bool created_flag_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_KERNEL_INTERNAL_PYBOOST_MLA_H_
