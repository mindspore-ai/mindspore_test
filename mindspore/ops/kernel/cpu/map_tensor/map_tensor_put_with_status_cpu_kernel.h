/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_TENSOR_PUT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_TENSOR_PUT_CPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "mindspore/ops/infer/map_tensor_put_with_status.h"
#include "kernel/cpu/map_tensor/map_tensor_cpu_kernel.h"

namespace mindspore {
namespace kernel {
constexpr size_t kMapTensorPutWithStatusInputNum = 4;
constexpr size_t kMapTensorPutWithStatusOutputNum = 1;

class MapTensorPutWithStatusCpuKernelMod : public MapTensorCpuKernelMod {
 public:
  MapTensorPutWithStatusCpuKernelMod() = default;
  ~MapTensorPutWithStatusCpuKernelMod() override = default;

  std::vector<KernelAttr> GetOpSupport() override;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_launch_func_(this, inputs, workspace, outputs);
  }

 private:
  template <typename KeyType, typename ValueType>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  void InitSizeLists(const ShapeVector &keys_shape, const ShapeVector &values_shape);

  size_t input_key_type_size_{0};
  size_t input_value_type_size_{0};

  using MapTensorPutLaunchFunc =
    std::function<bool(MapTensorPutWithStatusCpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;

  static std::vector<std::pair<KernelAttr, MapTensorPutLaunchFunc>> map_tensor_put_with_status_func_list_;

  MapTensorPutLaunchFunc kernel_launch_func_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAP_TENSOR_PUT_CPU_KERNEL_H_
