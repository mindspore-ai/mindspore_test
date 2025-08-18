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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_HAL_MANAGER_ASCEND_ERR_MANAGER_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_HAL_MANAGER_ASCEND_ERR_MANAGER_H_

#include <string>
#include <mutex>
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/visible.h"

namespace mindspore {
namespace device {
namespace ascend {
class ASCEND_RES_MANAGER_EXPORT ErrorManagerAdapter {
 public:
  ErrorManagerAdapter() = default;
  ~ErrorManagerAdapter() = default;
  static bool Init();
  static bool Finalize();
  static std::string GetErrorMessage(bool add_title = false);

 private:
  static void MessageHandler(std::ostringstream *oss);
  static void TaskExceptionCallback(aclrtExceptionInfo *task_fail_info);
  static std::string GetErrorMsgFromErrorCode(uint32_t rt_error_code);

  static std::mutex initialized_mutex_;
  static bool initialized_;
};

}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_HAL_MANAGER_ASCEND_ERR_MANAGER_H_
