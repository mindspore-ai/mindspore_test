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

#include "plugin/device/cpu/kernel/custom/custom_op_plugin_kernel.h"
#include "plugin/device/cpu/kernel/custom/op_plugin_utils.h"
#include "common/ms_factory.h"

namespace mindspore::kernel {
static bool g_init_op_plugin_kernels = []() {
  const auto &op_names = GetAllOpPluginKernelNames();
  for (const auto &op_name : op_names) {
    Factory<CustomOpPluginCpuKernelMod>::Instance().Register(
      op_name, []() { return std::make_shared<CustomOpPluginCpuKernelMod>(); });
  }
  return true;
}();
}  // namespace mindspore::kernel
