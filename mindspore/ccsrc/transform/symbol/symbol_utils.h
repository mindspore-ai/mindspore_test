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

#ifndef MINDSPORE_CCSRC_TRANSFORM_SYMBOL_SYMBOL_UTILS_H_
#define MINDSPORE_CCSRC_TRANSFORM_SYMBOL_SYMBOL_UTILS_H_
#include <string>
#include "utils/log_adapter.h"
#include "acl/acl.h"
#ifndef BUILD_LITE
#include "utils/ms_exception.h"
#endif

template <typename Function, typename... Args>
auto RunAscendApi(Function f, const char *file, int line, const char *call_f, const char *func_name, Args... args) {
  MS_LOG(DEBUG) << "Call ascend api <" << func_name << "> in <" << call_f << "> at " << file << ":" << line;
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
#ifndef BUILD_LITE
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f), Args...>, int>) {
    auto ret = f(args...);
    if (mindspore::UCEException::GetInstance().is_enable_uce()) {
      if (ret == ACL_ERROR_RT_DEVICE_MTE_ERROR && !mindspore::UCEException::GetInstance().get_has_throw_error()) {
        mindspore::UCEException::GetInstance().set_uce_flag(true);
      }
      if (ret == ACL_ERROR_RT_DEVICE_TASK_ABORT) {
        mindspore::UCEException::GetInstance().set_force_stop_flag(true);
      }
    }
    return ret;
  } else {
    return f(args...);
  }
#else
  return f(args...);
#endif
}

template <typename Function>
auto RunAscendApi(Function f, const char *file, int line, const char *call_f, const char *func_name) {
  MS_LOG(DEBUG) << "Call ascend api <" << func_name << "> in <" << call_f << "> at " << file << ":" << line;
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
#ifndef BUILD_LITE
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f)>, int>) {
    auto ret = f();
    if (mindspore::UCEException::GetInstance().is_enable_uce()) {
      if (ret == ACL_ERROR_RT_DEVICE_MTE_ERROR && !mindspore::UCEException::GetInstance().get_has_throw_error()) {
        mindspore::UCEException::GetInstance().set_uce_flag(true);
      }
      if (ret == ACL_ERROR_RT_DEVICE_TASK_ABORT) {
        mindspore::UCEException::GetInstance().set_force_stop_flag(true);
      }
    }
    return ret;
  } else {
    return f();
  }
#else
  return f();
#endif
}

template <typename Function>
bool HasAscendApi(Function f) {
  return f != nullptr;
}

namespace mindspore {
namespace transform {

#define CALL_ASCEND_API(func_name, ...) \
  RunAscendApi(mindspore::transform::func_name##_, FILE_NAME, __LINE__, __FUNCTION__, #func_name, ##__VA_ARGS__)

#define HAS_ASCEND_API(func_name) HasAscendApi(mindspore::transform::func_name##_)

std::string GetAscendPath();
void *GetLibHandler(const std::string &lib_path);
void LoadAscendApiSymbols();
void LoadSimulationApiSymbols();
}  // namespace transform
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_SYMBOL_UTILS_H_
