/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ENVIRON_ENVIRON_GPU_GET_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ENVIRON_ENVIRON_GPU_GET_H_

#include <vector>
#include <string>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class EnvironGetGpuKernelMod : public NativeGpuKernelMod {
 public:
  EnvironGetGpuKernelMod() : value_type_attr_(kObjectTypeTensorType), handle_size_(0), key_size_(0), value_size_(0) {}
  ~EnvironGetGpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  };
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  // The type of env tensor get.
  TypeId value_type_attr_;

  size_t handle_size_;
  size_t key_size_;
  size_t value_size_;
};

MS_REG_GPU_KERNEL(EnvironGet, EnvironGetGpuKernelMod)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ENVIRON_ENVIRON_GPU_GET_H_
