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

#include "kernel/ascend/internal/pyboost/mla.h"

#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
inline bool GetSeqLenFromInputTensor(const TensorPtr &input_tensor, std::vector<int32_t> *seq_len) {
  if (input_tensor == nullptr) {
    return false;
  }

  auto tensor_on_cpu = input_tensor->cpu();
  auto input_tensor_value = static_cast<int32_t *>(tensor_on_cpu->data_c());
  auto input_tensor_value_num = tensor_on_cpu->Size() / sizeof(int32_t);
  seq_len->clear();
  for (size_t i = 0; i < input_tensor_value_num; i++) {
    (*seq_len).emplace_back(input_tensor_value[i]);
  }
  return true;
}

internal::InternalOpPtr Mla::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                          const internal::OutputsImmutableInfoList &outputs) {
  param_.type = internal::MLAParam::kSplitCache;
  created_flag_ = true;
  return internal::CreateMLAOp(inputs, outputs, param_, internal::kInternalMLAOpName);
}

bool Mla::UpdateParam() {
  if (created_flag_) {
    created_flag_ = false;
    return true;
  }

  auto ret = internal_op_->UpdateParam(&param_);
  if (ret != internal::kInternalOk) {
    MS_LOG(ERROR) << "Mla UpdateParam failed, kernel_name: " << kernel_name_;
    return false;
  }
  return true;
}

uint64_t Mla::GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const {
  return CalcInternalOpTilingHash(kernel_name_, tiling_key, param_.q_seq_len, param_.kv_seq_len);
}

void Mla::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
               const TensorPtr &query, const TensorPtr &q_rope, const TensorPtr &kv_cache, const TensorPtr &k_rope,
               const TensorPtr &block_tables, const std::optional<TensorPtr> &mask,
               const std::optional<TensorPtr> &deq_scale_qk, const std::optional<TensorPtr> &deq_scale_pv,
               const std::optional<TensorPtr> &q_seq_lens, const std::optional<TensorPtr> &context_lens,
               const int64_t &head_num, const float &scale_value, const int64_t &kv_head_num, const int64_t &mask_mode,
               const int64_t &is_ring) {
  TensorPtrList inputs = {query,
                          q_rope,
                          kv_cache,
                          k_rope,
                          block_tables,
                          mask.has_value() ? mask.value() : nullptr,
                          deq_scale_qk.has_value() ? deq_scale_qk.value() : nullptr,
                          deq_scale_pv.has_value() ? deq_scale_pv.value() : nullptr};
  auto outputs = op->outputs();

  TransInternalShapes(inputs, outputs);

  param_.head_size = static_cast<int32_t>(head_num);
  param_.tor = scale_value;
  param_.kv_head = static_cast<int32_t>(kv_head_num);
  param_.mask_type = static_cast<internal::MLAParam::MaskType>(mask_mode);
  param_.is_ring = static_cast<int32_t>(is_ring);

  if (!q_seq_lens.has_value() || !context_lens.has_value()) {
    MS_LOG(EXCEPTION) << "For MLA, the q_seq_lens and context_lens must not be None.";
  }

  (void)GetSeqLenFromInputTensor(q_seq_lens.value(), &param_.q_seq_len);
  (void)GetSeqLenFromInputTensor(context_lens.value(), &param_.kv_seq_len);

  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(Mla, Mla);
}  // namespace kernel
}  // namespace mindspore
