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
#include "debug/data_dump/device_statistic/statistic_kernel.h"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "debug/debugger/debugger_utils.h"
#include "include/common/debug/common.h"

namespace mindspore {

namespace datadump {

TensorPtr SyncDeviceToHostTensor(DeviceAddressPtr device_addr) {
  if (!device_addr) {
    return nullptr;
  }
  auto kernel_tensor = device_addr->kernel_tensor();
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto dtype_id = kernel_tensor->dtype_id();
  const auto &shape_vec = kernel_tensor->GetShapeVector();

  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(dtype_id, shape_vec);
  auto ret_sync = device_addr->SyncDeviceToHost(UnitSizeInBytes(dtype_id), out_tensor->data_c());
  if (!ret_sync) {
    MS_LOG(EXCEPTION) << "Convert format or Copy device mem to host failed";
  }
  return out_tensor;
}

DeviceAddressPtr StatisticKernel::GenerateDeviceAddress(const size_t &mem_size, const TypeId &dtype_id,
                                                        const ShapeVector &shape, const ValuePtr &value) {
  auto addr = device_context_->device_res_manager_->AllocateMemory(mem_size, kDefaultStreamIndex);
  MS_EXCEPTION_IF_NULL(addr);

  auto tensor = std::make_shared<kernel::KernelTensor>(addr, mem_size, Format::DEFAULT_FORMAT, dtype_id, shape,
                                                       device_context_->device_context_key().device_name_,
                                                       device_context_->device_context_key().device_id_);
  tensor->set_stream_id(kDefaultStreamIndex);
  tensor->SetType(std::make_shared<TensorType>(TypeIdToType(dtype_id)));
  tensor->SetShape(std::make_shared<abstract::TensorShape>(shape));
  if (value) {
    tensor->SetValue(value);
  }
  return device_context_->device_res_manager_->CreateDeviceAddress(tensor);
}

DeviceAddressPtr StatisticKernel::GetWorkSpaceDeviceAddress(const vector<KernelTensor *> &inputs,
                                                            const vector<KernelTensor *> &outputs) {
  auto ret = kernel_mod_->Resize(inputs, outputs);
  if (ret) {
    MS_LOG(EXCEPTION) << "Call Resize error, error id is " << ret;
  }
  auto work_space = kernel_mod_->GetWorkspaceSizeList();
  if (!work_space.empty() && work_space[0] != 0) {
    MS_VLOG(VL_DUMP) << "Statistic kernel name is " << kernel_name_ << ", workspace size is " << work_space[0]
                     << ", input shape is " << inputs[0]->GetShapeVector() << ", dtype is "
                     << TypeIdToString(inputs[0]->dtype_id());
    return runtime::DeviceAddressUtils::CreateWorkspaceAddress(device_context_, kDefaultStreamIndex, work_space[0]);
  }
  return nullptr;
}

DeviceAddressPtr StatisticKernel::GetOutputDeviceAddress(TypeId dtype_id) {
  ShapeVector shape_vec = {};
  return GenerateDeviceAddress(UnitSizeInBytes(dtype_id), dtype_id, shape_vec);
}

vector<KernelTensorPtr> StatisticKernel::GetExtraInputsDeviceAddress(KernelTensor *) {
  return vector<KernelTensorPtr>();
}

vector<DeviceAddressPtr> StatisticKernel::LaunchKernelAsync(KernelTensor *input, const uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(input);
  stream_id_ = stream_id;
  vector<KernelTensor *> inputs{input};
  auto extra_inputs = GetExtraInputsDeviceAddress(input);
  vector<DeviceAddressPtr> res;
  std::transform(extra_inputs.begin(), extra_inputs.end(), std::back_inserter(inputs),
                 [](const auto &extra_input) { return extra_input.get(); });
  auto output_addr = GetOutputDeviceAddress(input->dtype_id());
  vector<KernelTensor *> outputs{output_addr->kernel_tensor().get()};

  MS_EXCEPTION_IF_NULL(output_addr);
  MS_EXCEPTION_IF_NULL(kernel_mod_);

  void *stream_ptr = device_context_->device_res_manager_->GetStream(stream_id_);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto workspace_addr = GetWorkSpaceDeviceAddress(inputs, outputs);
  res.emplace_back(workspace_addr);
  res.emplace_back(output_addr);
  vector<KernelTensor *> workspace;
  if (workspace_addr) {
    workspace.emplace_back(workspace_addr->kernel_tensor().get());
  }
  MS_VLOG(VL_DUMP) << "Start launch statistic kernel, kernel name is " << kernel_name_ << ", stream id is "
                   << stream_id_;
  bool ret = kernel_mod_->Launch(inputs, workspace, outputs, stream_ptr);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Device cal statistic, launch " << kernel_name_ << "error";
  }
  return res;
}

}  // namespace datadump
}  // namespace mindspore
