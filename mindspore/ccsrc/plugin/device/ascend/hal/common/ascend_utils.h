/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_COMMON_ASCEND_UTILS_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_COMMON_ASCEND_UTILS_H_

#include <pthread.h>

#include <atomic>
#include <memory>
#include <string>
#include <set>
#include <vector>

#include "include/backend/kernel_graph.h"
#include "utils/ms_context.h"
#include "acl/acl_rt.h"

namespace mindspore {
namespace device {
namespace ascend {
class ErrorManagerAdapter {
 public:
  ErrorManagerAdapter() = default;
  ~ErrorManagerAdapter() = default;
  static bool Init();
  static std::string GetErrorMessage(bool add_title = false);

 private:
  static void MessageHandler(std::ostringstream *oss);

 private:
  static std::mutex initialized_mutex_;
  static bool initialized_;
};

std::string GetErrorMsg(uint32_t rt_error_code);

bool EnableLccl();

void InitializeAcl();

BACKEND_EXPORT std::string GetFormatMode();

void SavePrevStepWeight(const std::vector<AnfNodePtr> &weights, aclrtStream stream);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_COMMON_ASCEND_UTILS_H_
