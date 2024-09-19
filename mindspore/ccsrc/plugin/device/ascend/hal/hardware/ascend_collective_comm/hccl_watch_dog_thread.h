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
  HcclWatchDogHandler(uint32_t global_rank_id, uint32_t local_rank_id, uint32_t global_rank_size,
                      const std::map<std::string, HcclComm> &hcoms);
  ~HcclWatchDogHandler();
  bool Initialize();
  void Terminate();
  const uint32_t global_rank() const { return global_rank_id_; }
  const uint32_t local_rank() const { return local_rank_id_; }
  const uint32_t global_rank_size() const { return global_rank_size_; }
  const std::exception_ptr &exception() const { return exception_; }

 private:
  void WatchDogProcess();
  void SetException(HcclComm hcom, std::string *error_info);
  void HandleException();
  void DestroyHcclComm();
  void DoProcess();
  uint32_t global_rank_id_;
  uint32_t local_rank_id_;
  uint32_t global_rank_size_;
  std::map<std::string, HcclComm> hcoms_;
  std::thread thread_;
  std::mutex mutex_;
  std::exception_ptr exception_{nullptr};
  std::atomic<bool> terminate_{false};
};

class HcclWatchDogManager {
 public:
  static HcclWatchDogManager &GetInstance() {
    static HcclWatchDogManager instance;
    return instance;
  }
  ~HcclWatchDogManager() = default;
  HcclWatchDogManager(const HcclWatchDogManager &) = delete;
  HcclWatchDogManager &operator=(const HcclWatchDogManager &) = delete;

  void AddHandler(std::unique_ptr<HcclWatchDogHandler> handler) { handles_ = std::move(handler); }
  bool InitHandler();

  void DestoryHandler() {
    if (handles_ == nullptr) {
      return;
    }
    handles_->Terminate();
    handles_ = nullptr;
  }

 private:
  HcclWatchDogManager() = default;
  std::unique_ptr<HcclWatchDogHandler> handles_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_ASCEND_WATCH_DOG_THREAD_H_
