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

#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include "utils/dlopen_macro.h"
#include "acl/error_codes/rt_error_codes.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_base_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "include/common/debug/common.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const std::map<uint32_t, std::string> error_msg = {
  {ACL_RT_SUCCESS, "success"},
  {ACL_ERROR_RT_PARAM_INVALID, "param invalid"},
  {ACL_ERROR_RT_INVALID_DEVICEID, "invalid device id"},
  {ACL_ERROR_RT_CONTEXT_NULL, "current context null"},
  {ACL_ERROR_RT_STREAM_CONTEXT, "stream not in current context"},
  {ACL_ERROR_RT_MODEL_CONTEXT, "model not in current context"},
  {ACL_ERROR_RT_STREAM_MODEL, "stream not in model"},
  {ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID, "event timestamp invalid"},
  {ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL, " event timestamp reversal"},
  {ACL_ERROR_RT_ADDR_UNALIGNED, "memory address unaligned"},
  {ACL_ERROR_RT_FILE_OPEN, "open file failed"},
  {ACL_ERROR_RT_FILE_WRITE, "write file failed"},
  {ACL_ERROR_RT_STREAM_SUBSCRIBE, "error subscribe stream"},
  {ACL_ERROR_RT_THREAD_SUBSCRIBE, "error subscribe thread"},
  {ACL_ERROR_RT_GROUP_NOT_SET, "group not set"},
  {ACL_ERROR_RT_GROUP_NOT_CREATE, "group not create"},
  {ACL_ERROR_RT_STREAM_NO_CB_REG, "callback not register to stream"},
  {ACL_ERROR_RT_INVALID_MEMORY_TYPE, "invalid memory type"},
  {ACL_ERROR_RT_INVALID_HANDLE, "invalid handle"},
  {ACL_ERROR_RT_INVALID_MALLOC_TYPE, "invalid malloc type"},
  {ACL_ERROR_RT_FEATURE_NOT_SUPPORT, "feature not support"},
  {ACL_ERROR_RT_MEMORY_ALLOCATION, "memory allocation error"},
  {ACL_ERROR_RT_MEMORY_FREE, "memory free error"},
  {ACL_ERROR_RT_AICORE_OVER_FLOW, "aicore over flow"},
  {ACL_ERROR_RT_NO_DEVICE, "no device"},
  {ACL_ERROR_RT_RESOURCE_ALLOC_FAIL, "resource alloc fail"},
  {ACL_ERROR_RT_NO_PERMISSION, "no permission"},
  {ACL_ERROR_RT_NO_EVENT_RESOURCE, "no event resource"},
  {ACL_ERROR_RT_NO_STREAM_RESOURCE, "no stream resource"},
  {ACL_ERROR_RT_NO_NOTIFY_RESOURCE, "no notify resource"},
  {ACL_ERROR_RT_NO_MODEL_RESOURCE, "no model resource"},
  {ACL_ERROR_RT_INTERNAL_ERROR, "runtime internal error"},
  {ACL_ERROR_RT_TS_ERROR, "ts internal error"},
  {ACL_ERROR_RT_STREAM_TASK_FULL, "task full in stream"},
  {ACL_ERROR_RT_STREAM_TASK_EMPTY, " task empty in stream"},
  {ACL_ERROR_RT_STREAM_NOT_COMPLETE, "stream not complete"},
  {ACL_ERROR_RT_END_OF_SEQUENCE, "end of sequence"},
  {ACL_ERROR_RT_EVENT_NOT_COMPLETE, "event not complete"},
  {ACL_ERROR_RT_CONTEXT_RELEASE_ERROR, "context release error"},
  {ACL_ERROR_RT_SOC_VERSION, "soc version error"},
  {ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT, "task type not support"},
  {ACL_ERROR_RT_LOST_HEARTBEAT, "ts lost heartbeat"},
  {ACL_ERROR_RT_MODEL_EXECUTE, " model execute failed"},
  {ACL_ERROR_RT_REPORT_TIMEOUT, "report timeout"},
  {ACL_ERROR_RT_SYS_DMA, "sys dma error"},
  {ACL_ERROR_RT_AICORE_TIMEOUT, "aicore timeout"},
  {ACL_ERROR_RT_AICORE_EXCEPTION, "aicore exception"},
  {ACL_ERROR_RT_AICORE_TRAP_EXCEPTION, " aicore trap exception"},
  {ACL_ERROR_RT_AICPU_TIMEOUT, " aicpu timeout"},
  {ACL_ERROR_RT_AICPU_EXCEPTION, "aicpu exception"},
  {ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR, " aicpu datadump response error"},
  {ACL_ERROR_RT_AICPU_MODEL_RSP_ERR, "aicpu model operate response error"},
  {ACL_ERROR_RT_PROFILING_ERROR, "profiling error"},
  {ACL_ERROR_RT_IPC_ERROR, "ipc error"},
  {ACL_ERROR_RT_MODEL_ABORT_NORMAL, "model abort normal"},
  {ACL_ERROR_RT_KERNEL_UNREGISTERING, "kernel unregistering"},
  {ACL_ERROR_RT_RINGBUFFER_NOT_INIT, "ringbuffer not init"},
  {ACL_ERROR_RT_RINGBUFFER_NO_DATA, "ringbuffer no data"},
  {ACL_ERROR_RT_KERNEL_LOOKUP, "kernel lookup error"},
  {ACL_ERROR_RT_KERNEL_DUPLICATE, "kernel register duplicate"},
  {ACL_ERROR_RT_DEBUG_REGISTER_FAIL, "debug register failed"},
  {ACL_ERROR_RT_DEBUG_UNREGISTER_FAIL, "debug unregister failed"},
  {ACL_ERROR_RT_LABEL_CONTEXT, "label not in current context"},
  {ACL_ERROR_RT_PROGRAM_USE_OUT, "program register num use out"},
  {ACL_ERROR_RT_DEV_SETUP_ERROR, "device setup error"},
  {ACL_ERROR_RT_DRV_INTERNAL_ERROR, "drv internal error"},
};

constexpr auto kUnknowErrorString = "Unknown error occurred";

bool g_acl_initialized = false;
std::mutex g_acl_init_mutex;
}  // namespace

std::mutex ErrorManagerAdapter::initialized_mutex_;
bool ErrorManagerAdapter::initialized_ = false;

bool ErrorManagerAdapter::Init() {
  std::unique_lock<std::mutex> lock(initialized_mutex_);
  if (initialized_) {
    MS_LOG(DEBUG) << "Ascend error manager has been initialized.";
    return true;
  }
  LogWriter::SetMessageHandler(&MessageHandler);
  initialized_ = true;
  return true;
}

std::string ErrorManagerAdapter::GetErrorMessage(bool add_title) {
  int32_t device_id;
  if (CALL_ASCEND_API(aclrtGetDevice, &device_id) != ACL_SUCCESS) {
    MS_LOG(INFO) << "The device is not set yet, no need to fetch error from device.";
    return "";
  }
  const char *message = CALL_ASCEND_API(aclGetRecentErrMsg);
  const string error_message = message == nullptr ? "" : message;
  if (error_message.empty() || error_message.find(kUnknowErrorString) != string::npos) {
    return "";
  }
  if (add_title) {
    return "#umsg#Ascend Error Message:#umsg#" + error_message +
           "\n(Please search \"CANN Common Error Analysis\" at https://www.mindspore.cn for error code description)";
  }
  return error_message;
}

void ErrorManagerAdapter::MessageHandler(std::ostringstream *oss) {
  const auto &error_message = GetErrorMessage(true);
  if (!error_message.empty()) {
    *oss << error_message;
  }
}

std::string GetErrorMsg(uint32_t rt_error_code) {
  auto find_iter = error_msg.find(rt_error_code);
  if (find_iter == error_msg.end()) {
    return "Return error code unknown, ret code: " + std::to_string(rt_error_code);
  }
  return find_iter->second;
}

void *callback_thread_func(void *data) {
  pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);
#ifdef WITH_BACKEND
  auto callback_thread = reinterpret_cast<CallbackThread *>(data);
  while (callback_thread->flag_.load()) {
    try {
      auto ret = CALL_ASCEND_API(aclrtProcessReport, callback_thread->default_timeout_);
      if (ret && ret != ACL_ERROR_WAIT_CALLBACK_TIMEOUT && ret != ACL_ERROR_RT_REPORT_TIMEOUT) {
        MS_LOG(DEBUG) << "aclrtProcessReport err : " << ret << ".";
      }
    } catch (const std::exception &ex) {
      MS_LOG(ERROR) << "aclrtProcessReport exception : " << ex.what() << ".";
      break;
    }
  }
  MS_LOG(INFO) << "Exit callback thread loop.";
#endif
  return data;
}

namespace {
bool GenerateAclInitJson(const string &json_file_path) {
  nlohmann::json acl_init_json;
  // generate err_msg_mode
  acl_init_json["err_msg_mode"] = "1";

  // write to file
  std::string json_file_str = acl_init_json.dump();
  std::ofstream json_file(json_file_path);
  if (!json_file.is_open()) {
    MS_LOG(WARNING) << "Open file [" << json_file_path << "] failed!";
    return False;
  }
  json_file << json_file_str;
  json_file.close();
  MS_LOG(INFO) << "Generate aclInit json to file : " << json_file_path;
  return True;
}
}  // namespace

bool EnableLccl() {
  auto ascend_soc_version = MsContext::GetInstance()->ascend_soc_version();
  if (ascend_soc_version != "ascend910b" && ascend_soc_version != "ascend910_93") {
    return false;
  }
  auto enable_infer_boost = MsContext::GetInstance()->IsEnableInferBoost();
  if (enable_infer_boost) {
    static bool disable_lccl = common::GetEnv("MS_ENABLE_LCCL") == "off";
    if (disable_lccl) {
      return false;
    }
    return true;
  } else {
    static bool enable_lccl = common::GetEnv("MS_ENABLE_LCCL") == "on";
    if (enable_lccl) {
      return true;
    }
    return false;
  }
}

void InitializeAcl() {
  std::lock_guard<std::mutex> lock(g_acl_init_mutex);
  if (g_acl_initialized) {
    return;
  }

  const char *acl_json_path = nullptr;

  std::string file_name = "./aclinit.json";
  auto realpath = Common::CreatePrefixPath(file_name);
  if (realpath.has_value()) {
    if (Common::FileExists(realpath.value()) || GenerateAclInitJson(realpath.value())) {
      acl_json_path = realpath.value().c_str();
    }
  } else {
    MS_LOG(WARNING) << "Failed to get real path: [" << file_name << "] in generate aclInit json file path.";
  }

  if (CALL_ASCEND_API(aclInit, acl_json_path) != ACL_ERROR_NONE) {
    MS_LOG(WARNING) << "Call aclInit failed, acl data dump function will be unusable.";
  } else {
    MS_LOG(INFO) << "Call aclInit successfully";
  }
  g_acl_initialized = true;
}

std::string GetFormatMode() {
  auto format_mode = common::GetEnv("MS_FORMAT_MODE");
  if (format_mode.empty()) {
    // default set "0" for 910a graph sink, otherwise "1"
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (ms_context->ascend_soc_version() == "ascend910" && ms_context->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK) &&
        ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
      format_mode = "0";
    } else {
      format_mode = "1";
    }
  }
  return format_mode;
}

void SavePrevStepWeight(const std::vector<AnfNodePtr> &weights, aclrtStream stream) {
  for (const auto &node : weights) {
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    if (common::AnfAlgo::IsParameterWeight(param)) {
      auto tensor = param->default_param()->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      auto out_addr = AnfAlgo::GetMutableOutputAddr(param, 0, false);
      if (out_addr == nullptr || out_addr->GetPtr() == nullptr || IsOneOfHWSpecialFormat(out_addr->format())) {
        // skip async copy if addr is nullptr.
        // special format need convert to default format at host, so skip async copy if format is a special format.
        continue;
      }
      auto size = tensor->Size();
      auto ret = CALL_ASCEND_API(aclrtMemcpyAsync, tensor->data_c(), size, out_addr->GetMutablePtr(), size,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        MS_LOG_WITH_NODE(EXCEPTION, param) << "Call aclrtMemcpyAsync failed, param: " << param->DebugString();
      }
      tensor->set_copy_done_flag(true);
    }
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
