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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_KERNEL_FACTORY_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_DEVICE_STATISTIC_KERNEL_FACTORY_H_

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "tools/data_dump/device_statistic/statistic_kernel.h"

namespace mindspore {
namespace datadump {

class KernelFactory {
 public:
  using KernelCreator = std::function<std::shared_ptr<StatisticKernel>(const DeviceContext *)>;

  static KernelFactory &Instance() {
    static KernelFactory instance;
    return instance;
  }

  void RegisterKernel(const std::string &name, KernelCreator creator) { creators_[name] = creator; }

  std::shared_ptr<StatisticKernel> CreateKernel(const std::string &name, const DeviceContext *device_context) {
    auto it = creators_.find(name);
    if (it == creators_.end()) {
      MS_LOG(EXCEPTION) << "No kernel named " << name << " found.";
    }

    return it->second(device_context);
  }

 private:
  std::unordered_map<std::string, KernelCreator> creators_;

  KernelFactory() = default;
  ~KernelFactory() = default;
  KernelFactory(const KernelFactory &) = delete;
  KernelFactory &operator=(const KernelFactory &) = delete;
};

#define REGISTER_KERNEL(name, type)                                                                        \
  namespace {                                                                                              \
  struct type##Register {                                                                                  \
    type##Register() {                                                                                     \
      KernelFactory::Instance().RegisterKernel(                                                            \
        name, [](const DeviceContext *device_context) { return std::make_shared<type>(device_context); }); \
    }                                                                                                      \
  };                                                                                                       \
  static type##Register global_##type##Register;                                                           \
  }  // namespace

}  // namespace datadump
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_KERNEL_FACTORY_H_
