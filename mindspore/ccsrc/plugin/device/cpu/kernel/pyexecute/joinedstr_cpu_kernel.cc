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

#include "plugin/device/cpu/kernel/pyexecute/joinedstr_cpu_kernel.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <utility>

#include "Eigen/Core"
#include "abstract/utils.h"
#include "ir/anf.h"
#include "mindspore/ops/op_def/framework_op_name.h"
#include "utils/log_adapter.h"
#include "frontend/ir/py_execute_py.h"

namespace mindspore {
namespace kernel {
bool JoinedStrCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "The input size is " + std::to_string(inputs.size());
  MS_EXCEPTION_IF_NULL(primitive_);
  return true;
}

std::string ConvertAbsToStr(KernelTensor *input) {
  auto py_tensor = ValueToPyData(PyExecuteInitializer::GetValueFromAbstract(input));
  MS_EXCEPTION_IF_NULL(py_tensor);
  return py::str(py_tensor).cast<std::string>();
}

bool JoinedStrCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                   const std::vector<KernelTensor *> &outputs) {
  py::gil_scoped_acquire gil_acquire;
  std::string exception_msg;
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    exception_msg += ConvertAbsToStr(input);
  }
  AbstractBase *output = outputs[0];
  output->set_user_data<string>("str_exception_result", std::make_shared<string>(exception_msg));
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, JoinedStr, JoinedStrCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
