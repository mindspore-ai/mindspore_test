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

#include "plugin/device/ascend/hal/hardware/stress_detect.h"
#include <string>
#include <thread>
#include <future>
#include <memory>
#include <utility>
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace kernel {
namespace pyboost {

void StressDetectTask::Run() {
  auto ret = run_func_(device_id_, workspace_addr_, workspace_size_);
  p_.set_value(ret);
}

void AmlAicoreDetectTask::Run() {
  auto ret = run_func_(device_id_, attr_.get());
  p_.set_value(ret);
}

int LaunchAclnnWithNoInput(const std::string &aclnn_name, const device::DeviceContext *device_context) {
  runtime::ProfilerRecorder aclnn_profiler(runtime::ProfilerModule::kPynative,
                                           runtime::ProfilerEvent::kPyBoostLaunchAclnn, aclnn_name, false);
  uint64_t workspace_size = 10;
  constexpr uint64_t kSize = 1024;
  workspace_size = workspace_size * kSize * kSize * kSize;
  void *workspace_addr = nullptr;
  auto workspace_device_address = runtime::DeviceAddressUtils::CreateWorkspaceAddressWithoutKernelTensor(
    device_context, device_context->device_res_manager_->DefaultStream(), workspace_size, true);
  if (workspace_device_address->GetMutablePtr() == nullptr) {
    MS_LOG(WARNING) << " Can't allocate workspace memory size: " << workspace_size << " for " << aclnn_name;
    return 0;
  }
  workspace_addr = workspace_device_address->GetMutablePtr();
  std::promise<int> p;
  std::future<int> f = p.get_future();
  std::shared_ptr<runtime::AsyncTask> task;

  if (aclnn_name == "AmlAicoreDetectOnline") {
    auto ascend_path = mindspore::device::ascend::GetAscendPath();
    auto lib_path = ascend_path + GetLibAscendMLName();
    void *lib_handle = dlopen(lib_path.c_str(), RTLD_LAZY);
    if (lib_handle == nullptr) {
      MS_LOG(EXCEPTION) << lib_path << " was not found. Exiting stress detect";
    }
    const auto *op_api_func = dlsym(lib_handle, aclnn_name.c_str());
    if (op_api_func == nullptr) {
      MS_LOG(EXCEPTION) << aclnn_name << " not in " << GetLibAscendMLName() << ", please check!";
    }
    auto run_api_func = reinterpret_cast<int (*)(int32_t, const AmlAicoreDetectAttr *)>(op_api_func);

    auto aml_attr = std::make_shared<AmlAicoreDetectAttr>();
    aml_attr->mode = AML_DETECT_RUN_MODE_ONLINE;
    aml_attr->workspaceSize = workspace_size;
    aml_attr->workspace = workspace_addr;

    task = std::make_shared<AmlAicoreDetectTask>(
      std::move(run_api_func), device_context->device_context_key().device_id_, aml_attr, std::move(p));
    auto aml_task = std::dynamic_pointer_cast<AmlAicoreDetectTask>(task);
    MS_LOG(DEBUG) << "aml_task created with device_id: " << aml_task->device_id()
                  << ", attr.runmode: " << aml_task->attr()->mode
                  << ", attr.workspaceSize: " << aml_task->attr()->workspaceSize
                  << ", attr.workspace: " << aml_task->attr()->workspace;
  } else {
    const auto op_api_func = device::ascend::GetOpApiFunc(aclnn_name.c_str());
    if (op_api_func == nullptr) {
      MS_LOG(EXCEPTION) << aclnn_name << " not in " << device::ascend::GetOpApiLibName() << ", please check!";
    }
    auto run_api_func = reinterpret_cast<int (*)(int32_t, void *, uint64_t)>(op_api_func);
    task = std::make_shared<StressDetectTask>(std::move(run_api_func), device_context->device_context_key().device_id_,
                                              workspace_addr, workspace_size, std::move(p));
  }
  runtime::Pipeline::Get().stress_detect()->Push(task);
  runtime::Pipeline::Get().stress_detect()->Wait();
  int api_ret = f.get();
  return api_ret;
}

int StressDetectKernel(const device::DeviceContext *device_context) {
  auto ascend_path = mindspore::device::ascend::GetAscendPath();
  auto lib_path = ascend_path + GetLibAscendMLName();
  int ret;

  void *lib_handle = dlopen(lib_path.c_str(), RTLD_LAZY);
  if (lib_handle) {
    // Try to find the function
    void *func_ptr = dlsym(lib_handle, kNameAmlAicoreDetectOnline);
    if (func_ptr) {
      MS_LOG(INFO) << "Using new API AmlAicoreDetectOnline from " << lib_path;
      ret = LaunchAclnnWithNoInput("AmlAicoreDetectOnline", device_context);
    } else {
      MS_LOG(INFO) << "AmlAicoreDetectOnline not found in " << lib_path << ". Using the StressDetect api instead.";
      ret = LaunchAclnnWithNoInput("StressDetect", device_context);
    }
    dlclose(lib_handle);
  } else {
    MS_LOG(INFO) << lib_path << " not found. Using the StressDetect api instead.";
    ret = LaunchAclnnWithNoInput("StressDetect", device_context);
  }
  constexpr int clear_device_state_fail = 574007;
  if (ret == clear_device_state_fail) {
    MS_LOG(EXCEPTION) << "Stress detect: clear device state fail!";
  }
  return ret;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
