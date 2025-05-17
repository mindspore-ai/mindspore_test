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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_MATMUL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_MATMUL_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class InternalKernelInfoMatMul : public InternalKernelInfo {
 public:
  InternalKernelInfoMatMul() : InternalKernelInfo(std::move("MatMul")) {}
  ~InternalKernelInfoMatMul() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &input_tensor,
            const BaseTensorPtr &mat2_tensor, const bool &transpose_a, const bool &transpose_b);

 protected:
  uint64_t GenerateTilingKey(const std::string &kernel_name, const std::vector<BaseTensorPtr> &inputs) override;
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  bool transpose_a_{false};
  bool transpose_b_{false};
  std::string output_format_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_MATMUL_H_
