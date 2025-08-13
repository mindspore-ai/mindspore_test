/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_KERNEL_PLUGIN_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_KERNEL_PLUGIN_H_

#include <memory>
#include <vector>
#include <string>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore::kernel {
class KernelPlugin {
 public:
  ~KernelPlugin() = default;

  virtual KernelModPtr BuildKernel(const AnfNodePtr &anf_node) = 0;
  virtual bool IsRegisteredKernel(const AnfNodePtr &anf_node) = 0;
  virtual void GetValidKernelBuildInfoWithInternalFormat(const AnfNodePtr &node,
                                                         std::vector<std::string> *input_formats,
                                                         std::vector<std::string> *output_formats) {}
  virtual void InitInternalLog() {}
};

#define MS_PLUGIN_FACTORY_REG(BASE, NAME, DERIVE)                                              \
  static_assert(std::is_base_of<BASE, DERIVE>::value, #DERIVE " must be derived from " #BASE); \
  static const KernelRegistrar<BASE> g_##NAME##_##BASE##_reg(#NAME, []() { return std::make_shared<DERIVE>(); })

#define MS_KERNEL_PLUGIN_FACTORY_REG(NAME, DERIVE) MS_KERNEL_FACTORY_REG(KernelPlugin, NAME, DERIVE)
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_INTERNAL_KERNEL_PLUGIN_H_
