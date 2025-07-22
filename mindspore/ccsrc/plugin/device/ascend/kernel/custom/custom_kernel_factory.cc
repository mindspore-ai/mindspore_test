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

#include "plugin/device/ascend/kernel/custom/custom_kernel_factory.h"

namespace mindspore {
namespace kernel {

CustomKernelFactory &CustomKernelFactory::Instance() {
  static CustomKernelFactory instance;
  return instance;
}

bool CustomKernelFactory::Register(const std::string &op_name, const KernelCreator &creator) {
  return creators_.emplace(op_name, creator).second;
}

KernelModPtr CustomKernelFactory::Create(const std::string &op_name) {
  auto it = creators_.find(op_name);
  if (it != creators_.end()) {
    return (it->second)();
  }
  return nullptr;
}

bool CustomKernelFactory::IsRegistered(const std::string &op_name) {
  return creators_.find(op_name) != creators_.end();
}

}  // namespace kernel
}  // namespace mindspore
