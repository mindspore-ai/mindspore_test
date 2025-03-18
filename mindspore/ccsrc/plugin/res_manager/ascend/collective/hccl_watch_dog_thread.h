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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_WATCH_DOG_THREAD_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_WATCH_DOG_THREAD_H_

#include <atomic>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <thread>
#include "hccl/hccl.h"

namespace mindspore {
namespace device {
namespace ascend {
class HcclWatchDogHandler {
 public:
  HcclWatchDogHandler(uint32_t global_rank_id, const std::string &group_name, HcclComm hcom);
  ~HcclWatchDogHandler();
  bool Initialize();
  void Terminate();
  uint32_t rank_id() const { return rank_id_; }
  std::string group_name() const { return group_name_; }
  bool can_stop(bool stop = false);
  bool exit() const { return exit_; }

 private:
  void WatchDogProcess();
  void SetException(std::string *error_info, bool *disable);
  void HandleException();
  void DestroyHcclComm();
  void DoProcess();
  uint32_t rank_id_;
  std::string group_name_;
  HcclComm hcom_;
  std::thread thread_;
  std::mutex mutex_;
  std::exception_ptr exception_{nullptr};
  std::atomic<bool> terminate_{false};
  std::atomic<bool> can_stop_{false};
  std::atomic<bool> stop_request_{false};
  std::atomic<bool> exit_{false};
};

class HcclWatchDogManager {
 public:
  static HcclWatchDogManager &GetInstance() {
    static HcclWatchDogManager instance;
    return instance;
  }

  void AddHandler(std::unique_ptr<HcclWatchDogHandler> handler) { (void)handles_.emplace_back(std::move(handler)); }
  uint32_t HandleSize() { return handles_.size(); }
  bool InitHandler(uint32_t idx);
  void DestroyHandlerByName(const std::string &name);
  void DestoryHandler() {
    std::unique_lock<std::mutex> lock(handle_mutex_);
    if (handles_.empty()) {
      return;
    }
    for (const auto &handle : handles_) {
      if (handle != nullptr) {
        handle->Terminate();
      }
    }
  }

 private:
  HcclWatchDogManager() = default;
  ~HcclWatchDogManager();
  HcclWatchDogManager(const HcclWatchDogManager &) = delete;
  HcclWatchDogManager &operator=(const HcclWatchDogManager &) = delete;
  std::mutex handle_mutex_;
  std::vector<std::unique_ptr<HcclWatchDogHandler>> handles_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_WATCH_DOG_THREAD_H_
