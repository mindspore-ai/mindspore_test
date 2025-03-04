/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/dump/adapter.h"
#include <utility>
#include <memory>

namespace mindspore {
namespace dump {
AdapterManager &AdapterManager::Instance() {
  static AdapterManager adapter_manager;
  return adapter_manager;
}

void AdapterManager::RegisterAdapter(device::DeviceType backend, std::shared_ptr<Adapter> adapter_ptr) {
  registered_adapters_.emplace(std::pair<device::DeviceType, std::shared_ptr<Adapter>>(backend, adapter_ptr));
}

std::shared_ptr<Adapter> AdapterManager::GetAdapterForBackend(device::DeviceType backend) {
  auto iter = registered_adapters_.find(backend);
  if (iter == registered_adapters_.end()) {
    return nullptr;
  }
  return iter->second;
}
}  // namespace dump
}  // namespace mindspore
