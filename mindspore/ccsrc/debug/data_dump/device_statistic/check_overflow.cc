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
#include "debug/data_dump/device_statistic/check_overflow.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "debug/debugger/debugger_utils.h"
#include "include/common/debug/common.h"
#include "debug/data_dump/device_statistic/kernel_factory.h"

namespace mindspore {
namespace datadump {

vector<KernelTensor *> CheckOverflowKernel::CheckInputs(vector<KernelTensor *> inputs) {
  std::vector<KernelTensor *> check_kernel_tensors;
  static std::set<TypeId> warning_once;

  for (size_t i = 0; i < inputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    auto type = inputs[i]->dtype_id();
    if (supported_dtype_.find(type) != supported_dtype_.end()) {
      check_kernel_tensors.emplace_back(inputs[i]);
    } else {
      if (warning_once.find(type) != warning_once.end()) {
        break;
      } else {
        warning_once.insert(type);
        MS_LOG(WARNING) << "Overflow detection does not support " << type << " !";
      }
    }
  }
  return check_kernel_tensors;
}

DeviceAddressPtr CheckOverflowKernel::LaunchKernelAsync(vector<KernelTensor *> inputs, const std::uint32_t stream_id) {
  stream_id_ = stream_id;
  vector<KernelTensor *> selected_inputs = CheckInputs(inputs);
  if (selected_inputs.empty()) {
    return nullptr;
  }

  auto output_addr = GetOutputDeviceAddress(kNumberTypeBool);
  vector<KernelTensor *> outputs{output_addr->kernel_tensor().get()};

  MS_EXCEPTION_IF_NULL(output_addr);
  MS_EXCEPTION_IF_NULL(kernel_mod_);

  void *stream_ptr = device_context_->device_res_manager_->GetStream(stream_id_);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  bool ret = kernel_mod_->Launch(selected_inputs, {}, outputs, stream_ptr);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Device cal overflow check, launch " << kernel_name_ << "error";
  }
  return output_addr;
}

REGISTER_KERNEL(KCheckOverflow, CheckOverflowKernel);

}  // namespace datadump
}  // namespace mindspore
