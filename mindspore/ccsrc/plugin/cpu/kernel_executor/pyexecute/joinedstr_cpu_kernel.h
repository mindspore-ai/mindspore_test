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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_JOINEDSTR_KERNEL_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_JOINEDSTR_KERNEL_H

#include <map>
#include <memory>
#include <string>
#include <vector>
#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "kernel/cpu/cpu_kernel.h"

namespace py = pybind11;
namespace mindspore {
namespace kernel {
class JoinedStrCpuKernelMod : public NativeCpuKernelMod {
 public:
  JoinedStrCpuKernelMod() {}
  ~JoinedStrCpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override;
};

std::string ConvertAbsToStr(KernelTensor *input);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_JOINEDSTR_KERNEL_H
