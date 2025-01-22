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

#include "plugin/device/ascend/hal/hardware/ascend_collective_comm/hccl_watch_dog_thread.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "utils/ms_exception.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
bool HcclWatchDogManager::InitHandler(uint32_t idx) {
  if (handles_.empty() || idx > handles_.size() || idx <= 0) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(handles_[idx - 1]);
  return handles_[idx - 1]->Initialize();
}

HcclWatchDogHandler::~HcclWatchDogHandler() {
  MS_LOG(DEBUG) << "HcclWatchDogHandler destructor start";
  terminate_.store(true, std::memory_order_acq_rel);
  if (thread_.joinable()) {
    thread_.join();
  }
  MS_LOG(INFO) << "HcclWatchDogHandler thread exit, rank id: " << rank_id_ << ", group name: " << group_name_;
}

HcclWatchDogHandler::HcclWatchDogHandler(uint32_t rank_id, const std::string &group_name, HcclComm hcom) {
  rank_id_ = rank_id;
  group_name_ = group_name;
  hcom_ = hcom;
}

bool HcclWatchDogHandler::Initialize() {
  MS_LOG(INFO) << "Initialize hccl watch dog handler. rank id: " << rank_id_ << ", group name: " << group_name_;
  thread_ = std::thread(&HcclWatchDogHandler::WatchDogProcess, this);
  return true;
}

void HcclWatchDogHandler::SetException(std::string *error_info, bool *disable) {
  MS_EXCEPTION_IF_NULL(error_info);
  MS_EXCEPTION_IF_NULL(disable);
  MS_EXCEPTION_IF_NULL(hcom_);
  if (exception_ != nullptr) {
    MS_LOG(WARNING) << "Already has an exception";
    return;
  }
  MS_LOG(DEBUG) << "Watch dog checking for hcom: " << hcom_ << ", group name: " << group_name_
                << ", rank id: " << rank_id_;
  auto ret = hccl::HcclAdapter::GetInstance().HcclWatchdogThread(hcom_, error_info, disable);
  if (!ret) {
    std::ostringstream param_oss;
    param_oss << "HcclWatchdogThread catch an error: " << *error_info << ", rank id: " << rank_id_
              << ", group name: " << group_name_;
    auto exception_ptr = std::make_exception_ptr(std::runtime_error(param_oss.str()));
    std::unique_lock<std::mutex> lock(mutex_);
    exception_ = exception_ptr;
  }
}

void HcclWatchDogHandler::DestroyHcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
  MS_LOG(INFO) << "Destroy hccl comm, group name: " << group_name_ << ", rank id: " << rank_id_;
  (void)HcclCommDestroy(hcom_);
}

void HcclWatchDogHandler::HandleException() {
  if (exception_) {
    std::rethrow_exception(exception_);
  }
}

void HcclWatchDogHandler::Terminate() { terminate_.store(true, std::memory_order_acq_rel); }

void HcclWatchDogHandler::DoProcess() {
  // check exception in every 2s
  constexpr int64_t kQueryFrequency = 2000;
  std::string error_info;
  while (!terminate_.load()) {
    MS_LOG(DEBUG) << "Start check watch dog thread in every 2s .";
    std::this_thread::sleep_for(std::chrono::milliseconds(kQueryFrequency));
    error_info.clear();
    bool disable = false;
    SetException(&error_info, &disable);
    if (disable) {
      MS_LOG(WARNING) << "Call HcclGetCommAsyncError failed, close watchdog, group: " << group_name_;
      Terminate();
      break;
    }
    if (exception_) {
      MS_LOG(ERROR) << "Watchdog thread got hccl error, try to stop training. rank: " << rank_id_
                    << ", group name:" << group_name_;
      DestroyHcclComm();
      HandleException();
      Terminate();
    }
  }
}

void HcclWatchDogHandler::WatchDogProcess() {
  MS_LOG(INFO) << "WatchDogProcess start, rank id: " << rank_id_ << ", group name: " << group_name_;
  if (!(common::GetEnv(kSimulationLevel).empty() && common::UseHostCollective() &&
        !hccl::HcclAdapter::GetInstance().UseHcclCM())) {
    MS_LOG(INFO) << "No need watch dog, return!";
    return;
  }
  try {
    DoProcess();
  } catch (const std::exception &e) {
    auto msg = e.what();
    MS_LOG(ERROR) << "HcclWatchDog thread catch exception: " << msg
                  << ".\n Try to destroy all streams by watch dog thread.";
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    (void)device_context->device_res_manager_->BindDeviceToCurrentThread(false);
    AscendStreamMng::GetInstance().ForceDestroyAllStreams();
    auto exp = std::make_exception_ptr(std::runtime_error(msg));
    MsException::Instance().SetException(exp);
  }
  MS_LOG(INFO) << "end";
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
