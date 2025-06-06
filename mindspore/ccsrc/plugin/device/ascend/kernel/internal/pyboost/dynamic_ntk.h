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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_PYBOOST_DYNAMIC_NTK_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_PYBOOST_DYNAMIC_NTK_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class DynamicNTK : public InternalKernelInfo {
 public:
  DynamicNTK() : InternalKernelInfo(std::move("DynamicNTK")) {}
  ~DynamicNTK() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &position_ids_tensor,
            const BaseTensorPtr &inv_freq_tensor, const BaseTensorPtr &seq_lens_tensor, const TypeId &dtype);

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;
  enum class DynamicNTKOutType { Float16 = 0, BFloat16 = 1, Float32 = 2 };

 private:
  TypeId dtype_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_PYBOOST_DYNAMIC_NTK_H_
