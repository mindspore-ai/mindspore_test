/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <functional>
#include <map>
#include "unsupported/Eigen/CXX11/Tensor"
#include "kernel/cpu/eigen/bessel_k0e_cpu_kernel.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "mindspore/ops/infer/bessel_k0e.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselK0eInputsNum = 1;
constexpr size_t kBesselK0eOutputsNum = 1;
}  // namespace

bool BesselK0eCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kBesselK0eInputsNum || outputs.size() != kBesselK0eOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselK0eInputsNum << " and "
                  << kBesselK0eOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_dtype_ = inputs[0]->dtype_id();

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselK0eCpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselK0eCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselK0eCpuKernelMod::LaunchKernel<Eigen::half>;
      break;
    default:
      MS_LOG(ERROR) << "BesselK0e kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselK0eCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());

  return 0;
}

template <typename T>
bool BesselK0eCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                         const std::vector<kernel::KernelTensor *> &outputs) {
  int block_size = 1000;
  size_t tensor_size = inputs[0]->size() / sizeof(T);
  Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> input(static_cast<T *>(inputs[0]->device_ptr()), tensor_size);
  Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> output(static_cast<T *>(outputs[0]->device_ptr()), tensor_size);

  auto task = [this, &input, &output](size_t start, size_t end) {
    Eigen::array<Eigen::Index, 1> offsets = {static_cast<int64_t>(start)};
    Eigen::array<Eigen::Index, 1> extends = {static_cast<int64_t>(end - start)};
    output.slice(offsets, extends) = input.slice(offsets, extends).bessel_k0e();
  };
  ParallelLaunch(task, input_size_, block_size);

  return true;
}

std::vector<KernelAttr> BesselK0eCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselK0e, BesselK0eCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
