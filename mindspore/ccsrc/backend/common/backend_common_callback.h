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

#ifndef MINDSPORE_BACKEND_COMMON_CALLBACK_H_
#define MINDSPORE_BACKEND_COMMON_CALLBACK_H_

#include <string>
#include <functional>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <memory>
#include "include/backend/visible.h"
#include "utils/log_adapter.h"

namespace mindspore::backend_common {
// Base class for type-erased callbacks
struct CallbackBase {
  virtual ~CallbackBase() = default;
};

// Derived template for specific function signatures
template <typename Ret, typename... Args>
struct Callback : CallbackBase {
  using ret_type = Ret;
  using args_type = std::tuple<Args...>;
  std::function<Ret(Args...)> func;

  explicit Callback(std::function<Ret(Args...)> f) : func(f) {}

  Ret invoke(Args... args) { return func(args...); }
};

class BACKEND_COMMON_EXPORT BackendCommonCallback {
 public:
  static BackendCommonCallback &GetInstance() {
    static BackendCommonCallback instance;
    return instance;
  }

  // Delete copy constructor and assignment operator
  BackendCommonCallback(const BackendCommonCallback &) = delete;
  BackendCommonCallback &operator=(const BackendCommonCallback &) = delete;

  // Register a callback function with a name
  template <typename Ret, typename... Args>
  void RegisterCallback(const std::string &name, std::function<Ret(Args...)> func) {
    callback_map_[name] = std::make_unique<Callback<Ret, Args...>>(func);
  }

  // Get a registered callback function by name
  template <typename Ret, typename... Args>
  std::function<Ret(Args...)> GetCallback(const std::string &name) const {
    auto it = callback_map_.find(name);
    if (it != callback_map_.end()) {
      auto callback = static_cast<Callback<Ret, Args...> *>(it->second.get());
      return callback->func;
    }
    MS_LOG(WARNING) << "Kernel callback function " << name << " not found";
    return nullptr;
  }

 private:
  BackendCommonCallback() = default;
  std::unordered_map<std::string, std::unique_ptr<CallbackBase>> callback_map_;
};

template <typename Func>
class BackendCommonCallbackRegister {
 public:
  BackendCommonCallbackRegister(const std::string &name, Func func) { register_impl(name, std::move(func)); }

 private:
  template <typename R, typename... Args>
  void register_impl(const std::string &name, std::function<R(Args...)> func) {
    BackendCommonCallback::GetInstance().RegisterCallback<R, Args...>(name, std::move(func));
  }
};

#define REGISTER_BACKEND_COMMON_CALLBACK(func)                                                         \
  static const mindspore::backend_common::BackendCommonCallbackRegister<std::function<decltype(func)>> \
    g_##func##_callback_register(#func, func)
}  // namespace mindspore::backend_common

#endif  // MINDSPORE_BACKEND_COMMON_CALLBACK_H_
