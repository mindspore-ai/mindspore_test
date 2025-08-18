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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_HAL_MANAGER_ASCEND_HAL_MANAGER_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_HAL_MANAGER_ASCEND_HAL_MANAGER_H_

#include <map>
#include <mutex>
#include <set>
#include "acl/acl_rt.h"
#include "plugin/ascend/res_manager/visible.h"

namespace mindspore {
namespace device {
namespace ascend {
class ASCEND_RES_MANAGER_EXPORT AscendHalManager {
 public:
  static AscendHalManager &GetInstance();

  ~AscendHalManager() {}
  // init

  // device
  uint32_t GetDeviceCount();
  void InitDevice(uint32_t device_id);
  void ResetDevice(uint32_t device_id);
  void SetDeviceSatMode(const aclrtFloatOverflowMode &overflow_mode);
  void SetOpWaitTimeout(uint32_t op_wait_timeout);
  void SetOpExecuteTimeOut(uint32_t op_execute_timeout);
  void InitializeAcl();
  bool EnableLccl();

  // context
  aclrtContext CreateContext(uint32_t device_id);
  // reset the default context of device_id
  void ResetContext(uint32_t device_id);
  void SetContext(uint32_t device_id);
  void SetContextForce(uint32_t device_id);
  void DestroyContext(aclrtContext context);
  void DestroyAllContext();

 private:
  static AscendHalManager instance_;
  std::set<uint32_t> initialized_device_set_{};
  // default <device_id, aclrtcontext> pair
  std::map<uint32_t, aclrtContext> default_device_context_map_;

  // rt_contexts by aclrtCreateContext, to destroy
  std::set<aclrtContext> rt_contexts_;

  bool acl_initialized_ = false;
  std::mutex acl_init_mutex_;
};

}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_ASCEND_HAL_MANAGER_ASCEND_HAL_MANAGER_H_
