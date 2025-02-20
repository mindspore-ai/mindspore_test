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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_MSTX_MSTXSYMBOL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_MSTX_MSTXSYMBOL_H_
#include <string>
#include "acl/acl_prof.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "utils/dlopen_macro.h"

namespace mindspore {
namespace profiler {
namespace ascend {
ORIGIN_METHOD(mstxMarkA, void, const char *, void *)
ORIGIN_METHOD(mstxRangeStartA, uint64_t, const char *, void *)
ORIGIN_METHOD(mstxRangeEnd, void, uint64_t)

extern mstxMarkAFunObj mstxMarkA_;
extern mstxRangeStartAFunObj mstxRangeStartA_;
extern mstxRangeEndFunObj mstxRangeEnd_;

void LoadMstxApiSymbol(const std::string &ascend_path);

template <typename Function, typename... Args>
auto RunMstxApi(Function f, const char *file, int line, const char *call_f, const char *func_name, Args... args) {
  MS_LOG(DEBUG) << "Call mstx api <" << func_name << "> in <" << call_f << "> at " << file << ":" << line;
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
#ifndef BUILD_LITE
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f), Args...>, int>) {
    auto ret = f(args...);
    if ((mindspore::UCEException::GetInstance().enable_uce() || mindspore::UCEException::GetInstance().enable_arf()) &&
        ret == 0) {
      MS_LOG(INFO) << "Call mstx api <" << func_name << "> in <" << call_f << "> at " << file << ":" << line
                   << " failed, return val [" << ret << "].";
    }
    return ret;
  } else {
    return f(args...);
  }
#else
  return f(args...);
#endif
}

#define CALL_MSTX_API(func_name, ...) \
  RunMstxApi(func_name##_, FILE_NAME, __LINE__, __FUNCTION__, #func_name, ##__VA_ARGS__)
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_MSTX_MSTXSYMBOL_H_
