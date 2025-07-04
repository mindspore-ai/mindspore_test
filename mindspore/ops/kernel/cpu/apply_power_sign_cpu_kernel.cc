/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/apply_power_sign_cpu_kernel.h"
#include <algorithm>
#include <map>
#include <functional>
#include "common/common_utils.h"
#include "kernel/cpu/nnacl/fp32/adam_fp32.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace apply_power_sign_cpu {
namespace {
constexpr size_t kPowerSignInputsNum = 7;
constexpr size_t kPowerSignOutputsNum = 2;
constexpr size_t kIndexVar = 0;
constexpr size_t kIndexM = 1;
constexpr size_t kIndexLr = 2;
constexpr size_t kIndexLogBase = 3;
constexpr size_t kIndexSignDecay = 4;
constexpr size_t kIndexBeta = 5;
constexpr size_t kIndexGrad = 6;

template <typename T>
int Sgn(const T &x) {
  if (x > T(0)) {
    return 1;
  }
  if (x < T(0)) {
    return -1;
  }
  return 0;
}
}  // namespace

template <typename T>
void ApplyPowerSignCpuKernelMod::LaunchPowerSign(const std::vector<kernel::KernelTensor *> &inputs,
                                                 const std::vector<kernel::KernelTensor *> &) {
  T *var = GetDeviceAddress<T>(inputs, kIndexVar);
  T *m = GetDeviceAddress<T>(inputs, kIndexM);
  T *lr = GetDeviceAddress<T>(inputs, kIndexLr);
  T *logbase = GetDeviceAddress<T>(inputs, kIndexLogBase);
  T *sign_decay = GetDeviceAddress<T>(inputs, kIndexSignDecay);
  T *beta = GetDeviceAddress<T>(inputs, kIndexBeta);
  T *gradient = GetDeviceAddress<T>(inputs, kIndexGrad);

  for (int64_t b = 0; b < batch_size_; b++) {
    // multithreading
    auto task = [this, &var, &m, &gradient, &lr, &beta, &logbase, &sign_decay](size_t start, size_t end) {
      T one = static_cast<T>(1.0);
      for (size_t i = start; i < end; i++) {
        m[i] = gradient[i] * (one - beta[0]) + m[i] * beta[0];
        T sign_value = static_cast<T>(Sgn(gradient[i]) * Sgn(m[i]));
        T update = exp(logbase[0] * sign_decay[0] * sign_value) * gradient[i];
        var[i] = var[i] - lr[0] * update;
      }
    };
    ParallelLaunchAutoSearch(task, LongToSize(input_elements_), this, &parallel_search_info_);
    var = var + input_elements_;
    m = m + input_elements_;
    gradient = gradient + input_elements_;
    lr++;
    beta++;
    logbase++;
    sign_decay++;
  }
}

bool ApplyPowerSignCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  dtype_ = inputs[kIndex0]->dtype_id();
  batch_rank_ = ops::get_batch_rank(primitive_);
  return true;
}

bool ApplyPowerSignCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kPowerSignInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kPowerSignOutputsNum, kernel_name_);

  if (dtype_ == kNumberTypeFloat32) {
    LaunchPowerSign<float>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat16) {
    LaunchPowerSign<float16>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'var' should be Float16 or Float32, but got "
                      << TypeIdToType(dtype_)->ToString();
  }
  return true;
}

int ApplyPowerSignCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  std::vector<int64_t> var_shape = inputs[kIndexVar]->GetShapeVector();
  std::vector<int64_t> m_shape = inputs[kIndexM]->GetShapeVector();
  std::vector<int64_t> lr_shape = inputs[kIndexLr]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kIndexGrad]->GetShapeVector();

  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(var_shape, m_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'accum' must be the same as the shape of 'var', "
                     "but got the shape of 'accum': "
                  << m_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }

  if (!IsSameShape(var_shape, grad_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'grad' must be the same as the shape of 'var', "
                     "but got the shape of 'grad': "
                  << grad_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }

  if ((batch_rank_ != 0) && (lr_shape.size() != static_cast<size_t>(batch_rank_))) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape size of 'lr' must be equal to 'batch_rank', "
                     "but got the shape of 'lr': "
                  << lr_shape << " and 'batch_rank': " << batch_rank_;
    return KRET_RESIZE_FAILED;
  }

  if (!lr_shape.empty()) {
    batch_size_ = std::accumulate(lr_shape.begin(), lr_shape.end(), 1, std::multiplies<int64_t>());
  }

  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', batch_size_ must be greater than 0, but got batch_size: " << batch_size_;
    return KRET_RESIZE_FAILED;
  }

  input_elements_ = std::accumulate(var_shape.begin(), var_shape.end(), 1, std::multiplies<int64_t>());
  input_elements_ = input_elements_ / batch_size_;

  return ret;
}

std::vector<KernelAttr> ApplyPowerSignCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddInputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutputAttr(kNumberTypeFloat32)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1),
                                                     KernelAttr()
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddInputAttr(kNumberTypeFloat16)
                                                       .AddOutputAttr(kNumberTypeFloat16)
                                                       .AddOutputAttr(kNumberTypeFloat16)
                                                       .AddOutInRef(0, 0)
                                                       .AddOutInRef(1, 1)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ApplyPowerSign, ApplyPowerSignCpuKernelMod);
}  // namespace apply_power_sign_cpu
}  // namespace kernel
}  // namespace mindspore
