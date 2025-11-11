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

#include "tools/error_handler/error_handler.h"
#include <string>
#include <utility>
#include <vector>
#include "error_handler/error_config.h"
#include "include/utils/callback.h"

namespace mindspore {
namespace tools {
namespace {
constexpr char kStrUceTimeBegin[] = "time us=";
constexpr size_t kStrUceTimeBeginLen = sizeof(kStrUceTimeBegin) - 1;

SNAPSHOT_MANAGER_REG(kCPUDevice, SnapshotMgr);
SNAPSHOT_MANAGER_REG(kGPUDevice, SnapshotMgr);
}  // namespace

ErrorHandler &ErrorHandler::GetInstance() {
  static ErrorHandler instance;
  return instance;
}

void ErrorHandler::ProcessError(const FuncInfo &fn_info, int error_code,
                                const FuncGetRecentErrMsg &fn_get_recent_err_msg, ErrorType error_type,
                                bool throw_exception) {
  const std::string &api_func = fn_info.api_msg;
  if (api_func == "aclrtProcessReport" || api_func == "acltdtReceiveTensor" || api_func == "aclDestroyDataBuffer") {
    MS_LOG(DEBUG) << "Call ascend api <" << api_func << "> in <" << fn_info.caller_func << "> at "
                  << fn_info.caller_file << ":" << fn_info.caller_line << " failed, error code [" << error_code << "].";
  } else {
    MS_LOG(ERROR) << "Call ascend api <" << api_func << "> in <" << fn_info.caller_func << "> at "
                  << fn_info.caller_file << ":" << fn_info.caller_line << " failed, error code [" << error_code << "].";
  }

  switch (error_type) {
    case ErrorType::kHbmMultBitEccError: {
      if (fn_get_recent_err_msg != nullptr && !HasThrownError()) {
        SetUceOccurTime(ExtractUceTime(fn_get_recent_err_msg()));
      }
    }
    case ErrorType::kDeviceMemError:
    case ErrorType::kCommOpRetryFailError:
    case ErrorType::kSuspectRemoteError: {
      if (!HasThrownError()) {
        error_type_ = error_type;
        if (throw_exception) {
          MS_LOG(EXCEPTION) << GetErrorMsg() << ". Error code is " << error_code;
        } else {
          MS_LOG(ERROR) << GetErrorMsg() << ". Error code is " << error_code;
        }
      }
      break;
    }
    case ErrorType::kForceStopError: {
      SetForceStopFlag(true);
      MS_LOG(ERROR) << "ForceStopError error occurs when execute";
      break;
    }
    default: {
      break;
    }
  }
}

// extract UCE occurs time from string "HBM MULTI BIT ECC, Uncorrectable ECC, device_id=3,
// event_id=0x80e01801, time us=67672363666.[FUNC:ProcHBMRas][FILE:stars_engine.cc]"
uint64_t ErrorHandler::ExtractUceTime(const char *error_msg) {
  MS_VLOG(VL_UCE_HBM_MUTLI_BIT_ECC) << "Error message is " << error_msg;
  if (error_msg == nullptr) {
    return 0;
  }
  std::string message = error_msg;
  auto idx_begin = message.find(kStrUceTimeBegin);
  if (idx_begin == std::string::npos) {
    return 0;
  }
  size_t num_digits = 0;
  for (auto idx = idx_begin + kStrUceTimeBeginLen; idx < message.size(); ++idx) {
    if (!isdigit(message[idx])) {
      break;
    }
    num_digits += 1;
  }
  if (num_digits == 0) {
    return 0;
  }
  auto decimal_str = message.substr(idx_begin + kStrUceTimeBeginLen, num_digits);
  try {
    auto time_us = std::stoull(decimal_str);
    MS_VLOG(VL_UCE_HBM_MUTLI_BIT_ECC) << "Extracted time is " << time_us << " us.";
    return time_us;
  } catch (std::logic_error const &ex) {
    MS_LOG(ERROR) << "Convert decimal string " << decimal_str << " to uint64_t value failed.";
    return 0;
  }
}

void ErrorHandler::SaveConstants(const std::vector<KernelGraphPtr> &graphs) {
  if (!TftConfig::GetInstance()->IsEnableUCE()) {
    MS_LOG(INFO) << "Not enable UCE, skip saving constants.";
    return;
  }
  MS_LOG(INFO) << "Save constants of graphs for UCE recovery";
  for (auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    for (const auto &value_node : graph->graph_value_nodes()) {
      auto node_value = value_node->value();
      MS_EXCEPTION_IF_NULL(node_value);
      if (node_value->isa<tensor::Tensor>()) {
        auto tensor = node_value->cast<tensor::TensorPtr>();
        const_values_[value_node] = std::make_shared<tensor::Tensor>(*tensor);
      } else {
        const_values_[value_node] = value_node->value();
      }
    }
  }
}

const ValuePtr &ErrorHandler::GetConstant(const AnfNodePtr &node) {
  auto iter = const_values_.find(node);
  if (iter == const_values_.end()) {
    MS_LOG(EXCEPTION) << "Can not find tensor for node " << node->fullname_with_scope();
  }
  return iter->second;
}

void ErrorHandler::Clear() { const_values_.clear(); }

SnapshotMgrPtr SnapshotMgr::GetInstance(const std::string &device) {
  auto iter = GetInstanceMap().find(device);
  if (iter == GetInstanceMap().end()) {
    MS_LOG(EXCEPTION) << "Can not find SnapshotMgr for device " << device;
  }
  auto snapshot_mgr = iter->second;
  MS_EXCEPTION_IF_NULL(snapshot_mgr);
  return snapshot_mgr;
}

std::map<std::string, SnapshotMgrPtr> &SnapshotMgr::GetInstanceMap() {
  static std::map<std::string, SnapshotMgrPtr> instance_map = {};
  return instance_map;
}

bool SnapshotMgr::Register(const std::string &device, const SnapshotMgrPtr &instance) {
  auto ret = GetInstanceMap().insert(std::pair<std::string, SnapshotMgrPtr>(device, instance));
  if (ret.second) {
    MS_LOG(INFO) << "SnapshotMgr for device " << device << " is registered successfully.";
  } else {
    MS_LOG(WARNING) << "SnapshotMgr for device " << device << " has already been registered.";
  }
  return true;
}

void SnapshotMgr::Clear() { GetInstanceMap().clear(); }

bool HasResumableError() { return tools::ErrorHandler::GetInstance().HasThrownError(); }

bool NeedRebuildGroup() { return mindspore::tools::ErrorHandler::GetInstance().GetRebuildGroupFlag(); }

REGISTER_COMMON_CALLBACK(HasResumableError);
REGISTER_COMMON_CALLBACK(NeedRebuildGroup);
}  // namespace tools
}  // namespace mindspore
