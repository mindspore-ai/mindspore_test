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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_INTERNAL_PYBOOST_QUANT_BATCH_MATMUL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_INTERNAL_PYBOOST_QUANT_BATCH_MATMUL_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class QuantBatchMatmul : public InternalKernelInfo {
 public:
  explicit QuantBatchMatmul(std::string &&kernel_name) : InternalKernelInfo(std::move(kernel_name)) {}
  ~QuantBatchMatmul() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
            const BaseTensorPtr &x, const BaseTensorPtr &y, const BaseTensorPtr &scale,
            const std::optional<BaseTensorPtr> &offset, const std::optional<BaseTensorPtr> &bias,
            const std::optional<BaseTensorPtr> &pertoken_scale, const bool transpose_a, const bool transpose_b,
            const int64_t dtype);

 protected:
  uint64_t GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const override;
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  internal::MatmulParam param_;
  internal::TensorFormat output_format_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_INTERNAL_PYBOOST_QUANT_BATCH_MATMUL_H_
