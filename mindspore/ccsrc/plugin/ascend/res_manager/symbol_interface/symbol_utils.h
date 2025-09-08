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
#include "plugin/ascend/res_manager/symbol_interface/acl_base_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "utils/log_adapter.h"
#include "acl/acl.h"
#include "utils/ms_exception.h"
#include "include/backend/visible.h"

#ifndef ACL_ERROR_RT_DEVICE_MEM_ERROR
#define ACL_ERROR_RT_DEVICE_MEM_ERROR 507053
#endif
#ifndef ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR
#define ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR 507054
#endif
#ifndef ACL_ERROR_RT_COMM_OP_RETRY_FAIL
#define ACL_ERROR_RT_COMM_OP_RETRY_FAIL 507904
#endif
#ifndef ACL_ERROR_RT_DEVICE_TASK_ABORT
#define ACL_ERROR_RT_DEVICE_TASK_ABORT 107022
#endif

inline mindspore::UCEError GetErrorType(int error_code) {
  switch (error_code) {
    case ACL_ERROR_RT_DEVICE_MEM_ERROR:
      return mindspore::UCEError::kDeviceMemError;
    case ACL_ERROR_RT_HBM_MULTI_BIT_ECC_ERROR:
      return mindspore::UCEError::kHbmMultBitEccError;
    case ACL_ERROR_RT_COMM_OP_RETRY_FAIL:
      return mindspore::UCEError::kCommOpRetryFailError;
    case ACL_ERROR_RT_DEVICE_TASK_ABORT:
      return mindspore::UCEError::kForceStopError;
    default:
      return mindspore::UCEError::kUnknownError;
  }
}

template <typename Function, typename... Args>
auto RunAscendApi(Function f, const char *file, int line, const char *call_f, const char *func_name, Args... args) {
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
#ifndef BUILD_LITE
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f), Args...>, int>) {
    auto ret = f(args...);
    auto aclrt_get_last_error = mindspore::device::ascend::aclrtGetLastError_;
    auto acl_get_recent_err_msg = mindspore::device::ascend::aclGetRecentErrMsg_;
    if (ret != ACL_SUCCESS && aclrt_get_last_error != nullptr &&
        (mindspore::UCEException::IsEnableUCE() || mindspore::UCEException::IsEnableHCCE())) {
      auto error_code = aclrt_get_last_error(ACL_RT_THREAD_LEVEL);
      auto error_type = GetErrorType(error_code);
      mindspore::UCEException::GetInstance().ProcessApiUceError(mindspore::FuncInfo{file, line, call_f, func_name},
                                                                error_code, acl_get_recent_err_msg, error_type, true);
    }
    if (mindspore::UCEException::GetInstance().enable_arf()) {
      if (ret != ACL_SUCCESS && aclrt_get_last_error != nullptr) {
        auto error_code = aclrt_get_last_error(ACL_RT_THREAD_LEVEL);
        MS_LOG(DEBUG) << "Call ascend api <" << func_name << "> in <" << call_f << "> at " << file << ":" << line
                      << " failed, error code [" << error_code << "].";
        if (error_code == ACL_ERROR_RT_DEVICE_TASK_ABORT) {
          mindspore::UCEException::GetInstance().set_force_stop_flag(true);
        }
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
  if (f == nullptr) {
    MS_LOG(EXCEPTION) << func_name << " is null.";
  }
#ifndef BUILD_LITE
  if constexpr (std::is_same_v<std::invoke_result_t<decltype(f)>, int>) {
    auto ret = f();
    auto aclrt_get_last_error = mindspore::device::ascend::aclrtGetLastError_;
    auto acl_get_recent_err_msg = mindspore::device::ascend::aclGetRecentErrMsg_;
    if (ret != ACL_SUCCESS && aclrt_get_last_error != nullptr &&
        (mindspore::UCEException::IsEnableUCE() || mindspore::UCEException::IsEnableHCCE())) {
      auto error_code = aclrt_get_last_error(ACL_RT_THREAD_LEVEL);
      auto error_type = GetErrorType(error_code);
      mindspore::UCEException::GetInstance().ProcessApiUceError(mindspore::FuncInfo{file, line, call_f, func_name},
                                                                error_code, acl_get_recent_err_msg, error_type, true);
    }
    if (mindspore::UCEException::GetInstance().enable_arf()) {
      if (ret != ACL_SUCCESS && aclrt_get_last_error != nullptr) {
        auto error_code = aclrt_get_last_error(ACL_RT_THREAD_LEVEL);
        MS_LOG(DEBUG) << "Call ascend api <" << func_name << "> in <" << call_f << "> at " << file << ":" << line
                      << " failed, error code [" << error_code << "].";
        if (error_code == ACL_ERROR_RT_DEVICE_TASK_ABORT) {
          mindspore::UCEException::GetInstance().set_force_stop_flag(true);
        }
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

namespace mindspore::device::ascend {

#define CALL_ASCEND_API(func_name, ...) \
  RunAscendApi(mindspore::device::ascend::func_name##_, FILE_NAME, __LINE__, __FUNCTION__, #func_name, ##__VA_ARGS__)

#define HAS_ASCEND_API(func_name) HasAscendApi(mindspore::device::ascend::func_name##_)

std::string GetAscendPath();
void *GetLibHandler(const std::string &lib_path, bool if_global = false);
void LoadAscendApiSymbols();
void LoadSimulationApiSymbols();
}  // namespace mindspore::device::ascend

#endif  // MINDSPORE_CCSRC_TRANSFORM_SYMBOL_SYMBOL_UTILS_H_
