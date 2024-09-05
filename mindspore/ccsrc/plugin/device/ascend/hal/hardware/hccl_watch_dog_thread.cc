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

#include "plugin/device/ascend/hal/hardware/hccl_watch_dog_thread.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_context.h"
#include "utils/ms_exception.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
bool HcclWatchDogManager::InitHandler() {
  if (handles_ == nullptr) {
    return false;
  }
  return handles_->Initialize();
}

HcclWatchDogHandler::~HcclWatchDogHandler() {
  MS_LOG(DEBUG) << "HcclWatchDogHandler destructor start";
  terminate_.store(true, std::memory_order_acq_rel);
  if (thread_.joinable()) {
    thread_.join();
  }
  MS_LOG(INFO) << "HcclWatchDogHandler thread exit. global rank id: " << global_rank_id_
               << " local rank id: " << local_rank_id_ << ", global rank size: " << global_rank_size_;
}

HcclWatchDogHandler::HcclWatchDogHandler(uint32_t global_rank_id, uint32_t local_rank_id, uint32_t global_rank_size,
                                         const std::map<std::string, HcclComm> &hcoms)
    : global_rank_id_(global_rank_id),
      local_rank_id_(local_rank_id),
      global_rank_size_(global_rank_size),
      hcoms_(hcoms) {}

bool HcclWatchDogHandler::Initialize() {
  MS_LOG(INFO) << "Initialize hccl watch dog handler. global rank id: " << global_rank_id_
               << " local rank id: " << local_rank_id_ << ", global rank size: " << global_rank_size_;
  thread_ = std::thread(&HcclWatchDogHandler::WatchDogProcess, this);
  return true;
}

void HcclWatchDogHandler::SetException(HcclComm hcom, std::string *error_info) {
  MS_EXCEPTION_IF_NULL(error_info);
  MS_EXCEPTION_IF_NULL(hcom);
  if (exception() != nullptr) {
    MS_LOG(WARNING) << "Already has an exception";
    return;
  }
  MS_LOG(DEBUG) << "Check watch dog for: " << hcom;
  if (!hccl::HcclAdapter::GetInstance().HcclWatchdogThread(hcom, error_info)) {
    std::ostringstream param_oss;
    param_oss << "[rank " << local_rank_id_ << "] HcclWatchdogThread catch an error: " << *error_info
              << ". Global rank id: [" << global_rank_id_ << "]. Global rank size: [" << global_rank_size_ << "].";
    auto exception_ptr = std::make_exception_ptr(std::runtime_error(param_oss.str()));
    std::unique_lock<std::mutex> lock(mutex_);
    exception_ = exception_ptr;
  }
}

void HcclWatchDogHandler::DestroyHcclComm() {
  std::unique_lock<std::mutex> lock(mutex_);
  for (auto &pair : hcoms_) {
    auto name = pair.first;
    MS_LOG(INFO) << "Destroy hccl comm: " << name;
    if (pair.second == nullptr) {
      continue;
    }
    (void)HcclCommDestroy(pair.second);
  }
}

void HcclWatchDogHandler::HandleException() {
  if (exception()) {
    std::rethrow_exception(exception_);
  }
}

void HcclWatchDogHandler::Terminate() { terminate_.store(true, std::memory_order_acq_rel); }

void HcclWatchDogHandler::DoProcess() {
  // check exception in every 2s
  constexpr int64_t kQueryFrequency = 2000;
  while (!terminate_.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kQueryFrequency));
    std::string error_info;
    for (const auto &hcom_pair : hcoms_) {
      error_info.clear();
      SetException(hcom_pair.second, &error_info);
      if (exception()) {
        DestroyHcclComm();
        HandleException();
        Terminate();
        break;
      }
    }
  }
}

void HcclWatchDogHandler::WatchDogProcess() {
  MS_LOG(INFO) << "WatchDogProcess start";
  if (!(common::GetEnv(kSimulationLevel).empty() && common::UseHostCollective() &&
        !hccl::HcclAdapter::GetInstance().UseHcclCM())) {
    return;
  }
  try {
    DoProcess();
  } catch (const std::exception &e) {
    auto msg = e.what();
    MS_LOG(ERROR) << "HcclWatchDog thread catch exception: " << msg;
    auto exp = std::make_exception_ptr(std::runtime_error(msg));
    MsException::Instance().SetException(exp);
  }
  MS_LOG(DEBUG) << "end";
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
