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

#ifndef MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXSYMBOL_H_
#define MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXSYMBOL_H_
#include <string>
#include "utils/dlopen_macro.h"
#include "utils/ms_exception.h"

namespace mindspore {
namespace profiler {
struct mstxDomainRegistration_st {};
typedef struct mstxDomainRegistration_st mstxDomainRegistration_t;
typedef mstxDomainRegistration_t *mstxDomainHandle_t;

ORIGIN_METHOD(mstxMarkA, void, const char *, void *)
ORIGIN_METHOD(mstxRangeStartA, uint64_t, const char *, void *)
ORIGIN_METHOD(mstxRangeEnd, void, uint64_t)
ORIGIN_METHOD(mstxDomainCreateA, mstxDomainHandle_t, const char *)
ORIGIN_METHOD(mstxDomainDestroy, void, mstxDomainHandle_t)
ORIGIN_METHOD(mstxDomainMarkA, void, mstxDomainHandle_t, const char *, void *)
ORIGIN_METHOD(mstxDomainRangeStartA, uint64_t, mstxDomainHandle_t, const char *, void *)
ORIGIN_METHOD(mstxDomainRangeEnd, void, mstxDomainHandle_t, uint64_t)

extern mstxMarkAFunObj mstxMarkA_;
extern mstxRangeStartAFunObj mstxRangeStartA_;
extern mstxRangeEndFunObj mstxRangeEnd_;
extern mstxDomainCreateAFunObj mstxDomainCreateA_;
extern mstxDomainDestroyFunObj mstxDomainDestroy_;
extern mstxDomainMarkAFunObj mstxDomainMarkA_;
extern mstxDomainRangeStartAFunObj mstxDomainRangeStartA_;
extern mstxDomainRangeEndFunObj mstxDomainRangeEnd_;

void LoadMstxApiSymbol(const std::string &ascend_path);
bool IsCannSupportMstxApi();
bool IsCannSupportMstxDomainApi();

template <typename Function, typename... Args>
auto RunMstxApi(Function f, const char *file, int line, const char *call_f, const char *func_name, Args... args) {
  MS_LOG(DEBUG) << "Call mstx api <" << func_name << "> in <" << call_f << "> at " << file << ":" << line;
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
#ifndef BUILD_LITE
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f), Args...>, int>) {
    auto ret = f(args...);
    if ((mindspore::UCEException::IsEnableUCE() || mindspore::UCEException::GetInstance().enable_arf()) && ret == 0) {
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

}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_DEBUG_PROFILER_MSTX_MSTXSYMBOL_H_
