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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CPU_KERNRL_EXTRA_FUNC_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CPU_KERNRL_EXTRA_FUNC_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common/kernel.h"
#include "runtime/hardware/device_context.h"
#include "mindspore/ccsrc/pyboost/pyboost_kernel_extra_func.h"
#include "kernel/cpu/utils/visible.h"
#include "mindapi/base/macros.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class OPS_HOST_API PyboostCPUKernelExtraFunc : public PyboostKernelExtraFunc {
 public:
  void SetThreadPool(const kernel::KernelModPtr &kernel) override;
  bool IsKernelModRegistered(const std::string &op_name) override;
  bool IsPyBoostCustomRegistered(const std::string &op_name) override;
  bool IsEnableProfiler() override;
  void LaunchKernelWithProfiler(const std::string &op_name, const device::DeviceContext *device_context,
                                const std::vector<BaseShapePtr> &base_shape,
                                const std::function<void()> &func) override;
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CPU_KERNRL_EXTRA_FUNC_H_
