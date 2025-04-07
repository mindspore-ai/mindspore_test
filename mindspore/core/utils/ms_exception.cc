/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "utils/ms_exception.h"
#include <string>
namespace mindspore {
namespace {
constexpr char kStrUceTimeBegin[] = "time us=";
constexpr size_t kStrUceTimeBeginLen = sizeof(kStrUceTimeBegin) - 1;
constexpr char kStrUceTimeEnd[] = ".[FUNC:ProcHBMRas]";
}  // namespace

MsException &MsException::Instance() {
  static MsException instance{};
  return instance;
}

StaticAnalysisException &StaticAnalysisException::Instance() {
  static StaticAnalysisException instance{};
  return instance;
}

UCEException &UCEException::GetInstance() {
  static UCEException instance{};
  return instance;
}

void UCEException::ProcessUceError(const FuncInfo &fn_info, int error_code, FuncGetRecentErrMsg fn_get_recent_err_msg,
                                   UCEError error_type) {
  MS_LOG(ERROR) << fn_info.api_msg << " in <" << fn_info.caller_func << "> at " << fn_info.caller_file << ":"
                << fn_info.caller_line << " failed, error code [" << error_code << "].";
  if (error_type == UCEError::kHbmMultBitEccError && fn_get_recent_err_msg != nullptr) {
    set_uce_occur_time(ExtractUceTime(fn_get_recent_err_msg()));
  }
  if ((error_type == UCEError::kDeviceMemError || error_type == UCEError::kHbmMultBitEccError) &&
      !get_has_throw_error()) {
    uce_error_type_ = error_type;
    MS_LOG(ERROR) << "UCEError error occurs when execute, error_code=" << error_code;
  }
  if (error_type == UCEError::kForceStopError) {
    set_force_stop_flag(true);
    MS_LOG(ERROR) << GetForceStopErrorMsg();
  }
}

void UCEException::ProcessApiUceError(const FuncInfo &fn_info, int error_code,
                                      FuncGetRecentErrMsg fn_get_recent_err_msg, UCEError error_type,
                                      bool throw_exception) {
  const std::string &api_func = fn_info.api_msg;
  if (api_func == "aclrtProcessReport" || api_func == "acltdtReceiveTensor" || api_func == "aclDestroyDataBuffer") {
    MS_LOG(DEBUG) << "Call ascend api <" << api_func << "> in <" << fn_info.caller_func << "> at "
                  << fn_info.caller_file << ":" << fn_info.caller_line << " failed, error code [" << error_code << "].";
  } else {
    MS_LOG(ERROR) << "Call ascend api <" << api_func << "> in <" << fn_info.caller_func << "> at "
                  << fn_info.caller_file << ":" << fn_info.caller_line << " failed, error code [" << error_code << "].";
  }

  if ((error_type == UCEError::kDeviceMemError || error_type == UCEError::kHbmMultBitEccError) &&
      !get_has_throw_error()) {
    if (error_type == UCEError::kHbmMultBitEccError && fn_get_recent_err_msg != nullptr) {
      set_uce_occur_time(ExtractUceTime(fn_get_recent_err_msg()));
    }
    uce_error_type_ = error_type;
    if (throw_exception) {
      MS_LOG(EXCEPTION) << "UCEError error occurs when execute, error_code=" << error_code << ".";
    } else {
      MS_LOG(ERROR) << "UCEError error occurs when execute, error_code=" << error_code << ".";
    }
  }
  if (error_type == UCEError::kForceStopError) {
    set_force_stop_flag(true);
    MS_LOG(ERROR) << "ForceStopError error occurs when execute";
  }
}

// extract UCE occurs time from string "HBM MULTI BIT ECC, Uncorrectable ECC, device_id=3,
// event_id=0x80e01801, time us=67672363666.[FUNC:ProcHBMRas][FILE:stars_engine.cc]"
uint64_t UCEException::ExtractUceTime(const char *error_msg) {
  if (error_msg == nullptr) {
    return 0;
  }
  std::string message = error_msg;
  auto idx_begin = message.find(kStrUceTimeBegin);
  if (idx_begin == std::string::npos) {
    return 0;
  }
  auto idx_end = message.find(kStrUceTimeEnd, idx_begin);
  if (idx_end == std::string::npos) {
    return 0;
  }
  auto decimal_str = message.substr(idx_begin + kStrUceTimeBeginLen, idx_end - idx_begin - kStrUceTimeBeginLen);
  try {
    return std::stoull(decimal_str);
  } catch (std::logic_error const &ex) {
    MS_LOG(ERROR) << "Convert decimal string " << decimal_str << " to uint64_t value failed.";
    return 0;
  }
}

bool UCEException::IsEnableUCE() {
  static bool is_enable_uce = []() {
    auto tftEnv = common::GetEnv("MS_ENABLE_TFT");
    constexpr std::string_view optUCE = "UCE:1";
    if (!tftEnv.empty() && (tftEnv.find(optUCE) != std::string::npos)) {
      MS_LOG(WARNING) << "UCE enabled.";
      return true;
    }
    return false;
  }();
  return is_enable_uce;
}
}  // namespace mindspore
