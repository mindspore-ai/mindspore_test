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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_CUSTOM_CUSTOM_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_CUSTOM_CUSTOM_KERNEL_FACTORY_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include "common/kernel.h"
#include "kernel/ascend/visible.h"

namespace mindspore {
namespace kernel {
using KernelModPtr = std::shared_ptr<KernelMod>;
using KernelCreator = std::function<KernelModPtr()>;

class OPS_ASCEND_API CustomKernelFactory {
 public:
  static CustomKernelFactory &Instance();
  bool Register(const std::string &op_name, const KernelCreator &creator);
  KernelModPtr Create(const std::string &op_name);
  bool IsRegistered(const std::string &op_name);

 private:
  CustomKernelFactory() = default;
  ~CustomKernelFactory() = default;
  CustomKernelFactory(const CustomKernelFactory &) = delete;
  CustomKernelFactory &operator=(const CustomKernelFactory &) = delete;

  std::unordered_map<std::string, KernelCreator> creators_;
};

#define MS_CUSTOM_KERNEL_FACTORY_REG(NAME, CLASS)       \
  static const bool g_custom_kernel_reg_##__COUNTER__ = \
    mindspore::kernel::CustomKernelFactory::Instance().Register(NAME, []() { return std::make_shared<CLASS>(); });

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_KERNEL_CUSTOM_CUSTOM_KERNEL_FACTORY_H_
