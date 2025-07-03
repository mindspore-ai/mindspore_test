/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_FACTORY_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_FACTORY_H_

#include <functional>
#include <string>
#include <memory>

#include "utils/hash_map.h"
#include "runtime/hardware/device_context.h"
#include "src/extendrt/delegate_graph_executor.h"
#include "include/api/context.h"
#include "src/common/config_infos.h"
#include "extendrt/session/lite_graph_executor.h"

namespace mindspore {
using mindspore::LiteGraphExecutor;
// (zhaizhiqiang): Wrap graph executor as delegate.
// typedef std::shared_ptr<GraphSinkDelegate> (*DelegateCreator)(const std::shared_ptr<Context> &);
template <typename T>
using DelegateCreator = std::function<T(const std::shared_ptr<Context> &, const ConfigInfos &)>;

template <typename T>
class MS_API DelegateRegistry {
 public:
  DelegateRegistry() = default;
  virtual ~DelegateRegistry() = default;

  static DelegateRegistry<T> &GetInstance();

  void RegDelegate(const mindspore::DeviceType &device_type, const std::string &provider, DelegateCreator<T> *creator);
  void UnRegDelegate(const mindspore::DeviceType &device_type, const std::string &provider);
  T GetDelegate(const mindspore::DeviceType &device_type, const std::string &provider,
                const std::shared_ptr<Context> &ctx, const ConfigInfos &config_infos);

 private:
  mindspore::HashMap<DeviceType, mindspore::HashMap<std::string, DelegateCreator<T> *>> creator_map_;
};

template <typename T>
class DelegateRegistrar {
 public:
  DelegateRegistrar(const mindspore::DeviceType &device_type, const std::string &provider,
                    DelegateCreator<T> *creator) {
    DelegateRegistry<T>::GetInstance().RegDelegate(device_type, provider, creator);
  }
  ~DelegateRegistrar() = default;
};

#define REG_DELEGATE(device_type, provider, creator)                                                                  \
  using t = decltype(creator(std::declval<const std::shared_ptr<Context> &>(), std::declval<const ConfigInfos &>())); \
  static DelegateCreator<t> func = [](const std::shared_ptr<Context> &context, const ConfigInfos &config_infos) {     \
    return creator(context, config_infos);                                                                            \
  };                                                                                                                  \
  static DelegateRegistrar<t> g_##device_type##provider##Delegate(device_type, provider, &func);
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_DELEGATE_FACTORY_H_
