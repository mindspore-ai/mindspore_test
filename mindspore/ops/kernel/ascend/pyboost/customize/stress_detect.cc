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

#include "kernel/ascend/pyboost/customize/stress_detect.h"
#include <string>
#include <thread>
#include <future>
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

int LaunchAclnnWithNoInput(const std::string &aclnn_name, const device::DeviceContext *device_context) {
  runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);
  uint64_t workspace_size = 2;
  workspace_size = workspace_size * 1024 * 1024 * 1024;
  void *workspace_addr = nullptr;
  auto workspace_device_address = runtime::DeviceAddressUtils::CreateWorkspaceAddressWithoutKernelTensor(
    device_context, device_context->device_res_manager_->DefaultStream(), workspace_size, true);
  if (workspace_device_address->GetMutablePtr() == nullptr) {
    MS_LOG(WARNING) << " Can't allocate workspace memory size: " << workspace_size << " for " << aclnn_name;
    return 0;
  }
  workspace_addr = workspace_device_address->GetMutablePtr();
  const auto op_api_func = transform::GetOpApiFunc(aclnn_name.c_str());
  if (op_api_func == nullptr) {
    MS_LOG(EXCEPTION) << aclnn_name << " not in " << transform::GetOpApiLibName() << ", please check!";
  }
  auto run_api_func = reinterpret_cast<int (*)(int32_t, void *, uint64_t)>(op_api_func);
  std::future<int> result = std::async(std::launch::async, run_api_func,
                                       device_context->device_context_key().device_id_, workspace_addr, workspace_size);
  int api_ret = result.get();
  return api_ret;
}

int StressDetectKernel(const device::DeviceContext *device_context) {
  auto ret = LaunchAclnnWithNoInput("StressDetect", device_context);
  constexpr int clear_device_state_fail = 100010;
  if (ret == clear_device_state_fail) {
    MS_LOG(EXCEPTION) << "Stress detect: clear device state fail!";
  }
  return ret;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
