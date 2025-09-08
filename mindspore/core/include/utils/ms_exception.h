/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_UTILS_MS_EXCEPTION_H_
#define MINDSPORE_CORE_UTILS_MS_EXCEPTION_H_
#include <sys/types.h>
#include <cstdint>
#include <exception>
#include <functional>
#include <set>
#include <mutex>
#include <string>
#include "utils/ms_utils.h"
#include "utils/log_adapter.h"
#include "mindapi/base/macros.h"

using FuncGetRecentErrMsg = std::function<const char *()>;

namespace mindspore {
class ExceptionListener {
 public:
  virtual void OnException() = 0;
  virtual ~ExceptionListener() = default;
};

class MS_CORE_API MsException {
 public:
  static MsException &Instance();

  void SetException(const std::exception_ptr &def = nullptr) {
    if (def != nullptr) {
      exception_ptr_ = def;
    } else {
      exception_ptr_ = std::current_exception();
    }
    if (exception_ptr_ != nullptr && listener_ != nullptr) {
      auto listener = listener_;
      listener_ = nullptr;
      listener->OnException();
    }
  }

  void CheckException() {
    if (exception_ptr_ != nullptr) {
      auto exception_ptr = exception_ptr_;
      exception_ptr_ = nullptr;
      MS_LOG(DEBUG) << "Find exception and rethrow";
      std::rethrow_exception(exception_ptr);
    }
  }

  void SetExceptionListener(ExceptionListener *listener) { listener_ = listener; }

  void ResetException() { exception_ptr_ = nullptr; }

 private:
  MsException() = default;
  ~MsException() { listener_ = nullptr; }
  DISABLE_COPY_AND_ASSIGN(MsException)
  ExceptionListener *listener_{nullptr};
  std::exception_ptr exception_ptr_{nullptr};
};

class MS_CORE_API StaticAnalysisException {
 public:
  static StaticAnalysisException &Instance();

  void ClearException() {
    std::lock_guard<std::mutex> lock(lock_);
    msg_ = "";
    exception_ptr_ = nullptr;
  }

  bool HasException() {
    std::lock_guard<std::mutex> lock(lock_);
    return exception_ptr_ != nullptr;
  }

  void SetException() {
    std::lock_guard<std::mutex> lock(lock_);
    if (exception_ptr_ != nullptr) {
      return;
    }
    exception_ptr_ = std::current_exception();
  }
  void AppendMsg(const std::string &msg) {
    std::lock_guard<std::mutex> lock(lock_);
    msg_ += msg;
  }
  std::string msg() {
    std::lock_guard<std::mutex> lock(lock_);
    return msg_;
  }

  void SetAndRethrowException() {
    std::lock_guard<std::mutex> lock(lock_);
    SetException();
    std::rethrow_exception(std::current_exception());
  }

  void CheckException() {
    std::lock_guard<std::mutex> lock(lock_);
    if (exception_ptr_ != nullptr) {
      auto tmp_exception_ptr = exception_ptr_;
      std::rethrow_exception(tmp_exception_ptr);
    }
  }

 private:
  StaticAnalysisException() = default;
  ~StaticAnalysisException() = default;
  DISABLE_COPY_AND_ASSIGN(StaticAnalysisException)

  std::exception_ptr exception_ptr_{nullptr};
  std::string msg_;
  std::mutex lock_;
};

struct FuncInfo {
  const char *caller_file;
  int caller_line;
  const char *caller_func;
  std::string api_msg;
};

enum class UCEError : int {
  kNoneError = 0,
  kDeviceMemError,
  kHbmMultBitEccError,
  kCommOpRetryFailError,
  kForceStopError,
  kUnknownError
};

class MS_CORE_API UCEException {
 public:
  static UCEException &GetInstance();
  static uint64_t ExtractUceTime(const char *error_msg);
  static bool IsEnableUCE();
  static bool IsEnableHCCE();
  bool get_has_throw_error() const { return force_stop_flag_ || get_uce_flag() || is_reboot_node_; }

  void set_force_stop_flag(bool flag) { force_stop_flag_ = flag; }
  bool get_force_stop_flag() const { return force_stop_flag_; }

  bool get_uce_flag() const { return uce_error_type_ != UCEError::kNoneError; }
  bool get_hcce_flag() const { return uce_error_type_ == UCEError::kCommOpRetryFailError; }
  void clear_uce_error() { uce_error_type_ = UCEError::kNoneError; }

  void set_reboot_node(bool flag) { is_reboot_node_ = flag; }
  bool is_reboot_node() const { return is_reboot_node_; }
  void set_is_arf(bool flag) { is_arf_ = flag; }
  bool is_arf() const { return is_arf_; }
  bool enable_arf();
  void set_rebuild_group_flag(bool flag) { rebuild_group_ = flag; }
  bool rebuild_group_flag() const { return rebuild_group_; }
  bool check_rootinfo_clean_flag() const { return had_clean_; }
  bool need_clean_rootinfo() const { return clean_rootinfo_; }
  void set_rootinfo_clean_flag(bool flag) { had_clean_ = flag; }

  void CheckUceARFEnv() {
    if (init_) {
      return;
    }
    static std::string tftEnv = common::GetEnv("MS_ENABLE_TFT");
    if (tftEnv.empty()) {
      init_ = true;
      return;
    }
    const std::string optARF = "ARF:1";
    const std::string optRSC = "RSC:1";
    if (tftEnv.find(optARF) != std::string::npos) {
      arf_env_ = true;
      clean_rootinfo_ = true;
      MS_LOG(WARNING) << "ARF enabled.";
    }
    if (tftEnv.find(optRSC) != std::string::npos) {
      clean_rootinfo_ = true;
      MS_LOG(WARNING) << "RSC enabled.";
    }
    init_ = true;
  }
  void set_uce_occur_time(uint64_t time) { uce_occur_time_ = time; }
  uint64_t get_uce_occur_time() { return uce_occur_time_; }

  void ProcessApiUceError(const FuncInfo &fn_info, int error_code, const FuncGetRecentErrMsg &fn_get_recent_err_msg,
                          UCEError error_type, bool throw_exception = false);

  void ProcessUceError(const FuncInfo &fn_info, int error_code, const FuncGetRecentErrMsg &fn_get_recent_err_msg,
                       UCEError error_type);

  void SetGraphPipelineCompiled(bool value) { is_graph_pipeline_compiled_ = value; }

  const char *GetUceErrorMsg() const {
    if (uce_error_type_ == UCEError::kDeviceMemError) {
      return "UCEError error occurs when execute, error_code=507053";
    } else if (uce_error_type_ == UCEError::kHbmMultBitEccError) {
      return "UCEError error occurs when execute, error_code=507054";
    } else if (uce_error_type_ == UCEError::kCommOpRetryFailError) {
      return "HCCEError error occurs when execute";
    } else if (uce_error_type_ == UCEError::kNoneError) {
      return "No uce error occurs.";
    } else {
      return "Unknown error occurs.";
    }
  }

  const char *GetForceStopErrorMsg() const { return "ForceStopError error occurs when execute"; }

 private:
  UCEException() = default;
  ~UCEException() = default;
  DISABLE_COPY_AND_ASSIGN(UCEException)
  bool force_stop_flag_{false};
  bool arf_env_{false};
  bool is_reboot_node_{false};
  bool is_arf_{false};
  bool rebuild_group_{false};
  bool had_clean_{false};
  bool clean_rootinfo_{false};
  bool init_{false};
  uint64_t uce_occur_time_{0};
  UCEError uce_error_type_{UCEError::kNoneError};
  bool is_graph_pipeline_compiled_{false};
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_UTILS_MS_EXCEPTION_H_
