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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FFTNBASE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FFTNBASE_CPU_KERNEL_H_

#include <vector>
#include <complex>
#include <utility>
#include <map>
#include <functional>
#include <algorithm>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace mindspore {
namespace kernel {
namespace fftnbase_cpu {
class FFTNBaseCpuKernelMod : public NativeCpuKernelMod {
 public:
  FFTNBaseCpuKernelMod() = default;
  ~FFTNBaseCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void ResetResource();
  void FFTNGetAttr();

  template <typename T_in, typename T_out>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  template <typename T_in, typename T_out>
  bool LaunchKernelC2C(const std::vector<kernel::KernelTensor *> &inputs,
                       const std::vector<kernel::KernelTensor *> &outputs);

  template <typename T_in, typename T_out>
  bool LaunchKernelR2C(const std::vector<kernel::KernelTensor *> &inputs,
                       const std::vector<kernel::KernelTensor *> &outputs);

  template <typename T_in, typename T_out>
  bool LaunchKernelC2R(const std::vector<kernel::KernelTensor *> &inputs,
                       const std::vector<kernel::KernelTensor *> &outputs);

  using FFTNBaseFunc = std::function<bool(FFTNBaseCpuKernelMod *, const std::vector<KernelTensor *> &,
                                          const std::vector<KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, FFTNBaseFunc>> func_list_;
  FFTNBaseFunc kernel_func_;

  bool forward_;
  bool s_is_none_{false};
  bool dim_is_none_{false};
  int64_t x_rank_;
  int64_t input_element_nums_;
  int64_t calculate_element_nums_;
  int64_t fft_nums_;
  double norm_weight_;
  mindspore::NormMode norm_;

  std::vector<int64_t> dim_;
  std::vector<int64_t> s_;
  std::vector<int64_t> tensor_shape_;
  std::vector<int64_t> calculate_shape_;
};
}  // namespace fftnbase_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_FFTNBASE_CPU_KERNEL_H_
