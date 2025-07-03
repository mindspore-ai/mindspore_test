/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/pyexecute/raise_cpu_kernel.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

#include "Eigen/Core"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "mindspore/ops/op_def/framework_op_name.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
bool RaiseCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  constexpr size_t min_input_size = 2;
  MS_EXCEPTION_IF_CHECK_FAIL(
    inputs.size() >= min_input_size,
    "Input size should be at least " + std::to_string(min_input_size) + " but got " + std::to_string(inputs.size()));
  MS_EXCEPTION_IF_NULL(primitive_);
  return true;
}

bool RaiseCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                               const std::vector<KernelTensor *> &outputs) {
  constexpr size_t exception_type_abs_index = 0;
  constexpr size_t exception_msg_abs_index = 1;
  auto exception_type_abs = inputs[exception_type_abs_index];
  auto exception_msg_abs = inputs[exception_msg_abs_index];
  MS_EXCEPTION_IF_NULL(exception_type_abs);
  MS_EXCEPTION_IF_NULL(exception_msg_abs);
  const auto &exception_type_str = GetValue<std::string>(exception_type_abs->BuildValue());
  const auto &exception_msg = GetValue<std::string>(exception_msg_abs->BuildValue());
  auto iter = exception_types_map.find(exception_type_str);
  if (iter == exception_types_map.end()) {
    MS_LOG(ERROR) << "Found unexpected exception type " << exception_type_str;
    return false;
  }
  auto exception_type = iter->second;
  auto &handler = LogWriter::GetExceptionHandler();
  MS_EXCEPTION_IF_NULL(handler);
  handler(exception_type, exception_msg);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, raise, RaiseCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
