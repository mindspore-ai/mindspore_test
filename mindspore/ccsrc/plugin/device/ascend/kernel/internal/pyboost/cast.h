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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_CAST_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_CAST_H_

#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class InternalKernelInfoCast : public InternalKernelInfo {
 public:
  InternalKernelInfoCast() : InternalKernelInfo(std::move("Cast")) {}
  ~InternalKernelInfoCast() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) override;

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  TypeId dst_type_{kTypeUnknown};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_CAST_H_
