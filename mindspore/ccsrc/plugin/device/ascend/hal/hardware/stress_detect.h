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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_STRESS_DETECT_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_STRESS_DETECT_H_
#include <thread>
#include <future>
#include <utility>
#include "runtime/hardware/device_context.h"
#include "runtime/pipeline/task/task.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class StressDetectTask : public runtime::AsyncTask {
 public:
  StressDetectTask(std::function<int(int32_t, void *, uint64_t)> run_func, uint32_t device_id, void *workspace_addr,
                   uint64_t workspace_size, std::promise<int> &&p)
      : AsyncTask(runtime::kStressDetectTask),
        run_func_(std::move(run_func)),
        device_id_(device_id),
        workspace_addr_(workspace_addr),
        workspace_size_(workspace_size),
        p_(std::move(p)) {}
  ~StressDetectTask() override = default;
  void Run() override;

 private:
  std::function<int(int32_t, void *, uint64_t)> run_func_;
  uint32_t device_id_;
  void *workspace_addr_;
  uint64_t workspace_size_;
  std::promise<int> p_;
};
int StressDetectKernel(const device::DeviceContext *device_context);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_STRESS_DETECT_H_
