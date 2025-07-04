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
#include "backend/common/backend_common_callback.h"
#include "debug/debugger/debugger_utils.h"
#include "debug/data_dump/device_statistic/mem_manager.h"
#include "include/common/debug/common.h"
#include "include/backend/mem_reuse/mem_tracker.h"

namespace mindspore {

namespace datadump {

TensorPtr SyncDeviceToHostTensor(KernelTensorPtr kernel_tensor) {
  if (!kernel_tensor) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto device_addr = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(device_addr);
  auto dtype_id = kernel_tensor->dtype_id();
  const auto &shape_vec = kernel_tensor->GetShapeVector();

  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(dtype_id, shape_vec);
  auto ret_sync = device_addr->SyncDeviceToHost(UnitSizeInBytes(dtype_id), out_tensor->data_c());
  if (!ret_sync) {
    MS_LOG(EXCEPTION) << "Convert format or Copy device mem to host failed";
  }
  return out_tensor;
}

KernelTensorPtr StatisticKernel::GetWorkSpaceDeviceAddress(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  auto ret = kernel_mod_->Resize(inputs, outputs);
  if (ret) {
    MS_LOG(EXCEPTION) << "Call Resize error, error id is " << ret;
  }
  auto work_space = kernel_mod_->GetWorkspaceSizeList();
  if (!work_space.empty() && work_space[0] != 0) {
    MS_VLOG(VL_DUMP) << "Statistic kernel name is " << kernel_name_ << ", workspace size is " << work_space[0]
                     << ", input shape is " << inputs[0]->GetShapeVector() << ", dtype is "
                     << TypeIdToString(inputs[0]->dtype_id());
    return DumpMemManager::GetInstance().GetWorkSpaceTensor(device_context_, stream_id_, work_space[0]);
  }
  return nullptr;
}

KernelTensorPtr StatisticKernel::GetOutputDeviceAddress(TypeId dtype_id) {
  return DumpMemManager::GetInstance().GetOutputTensor(device_context_, stream_id_, dtype_id);
}

std::vector<KernelTensorPtr> StatisticKernel::GetExtraInputsDeviceAddress(KernelTensor *) {
  return std::vector<KernelTensorPtr>();
}

std::vector<KernelTensorPtr> StatisticKernel::LaunchKernelAsync(KernelTensor *input, const uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(input);
  stream_id_ = stream_id;
  std::vector<KernelTensor *> inputs{input};
  auto extra_inputs = GetExtraInputsDeviceAddress(input);
  std::vector<KernelTensorPtr> res;
  std::transform(extra_inputs.begin(), extra_inputs.end(), std::back_inserter(inputs),
                 [](const auto &extra_input) { return extra_input.get(); });
  auto output_kernel_tensor = GetOutputDeviceAddress(input->dtype_id());
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  std::vector<KernelTensor *> outputs{output_kernel_tensor.get()};

  MS_EXCEPTION_IF_NULL(kernel_mod_);

  void *stream_ptr = device_context_->device_res_manager_->GetStream(stream_id_);
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto workspace_kernel_tensor = GetWorkSpaceDeviceAddress(inputs, outputs);
  // in low precision mode, workspace is about 1-13KB.
  // don't use memreuse capture workspace mem.
  if (!DumpJsonParser::GetInstance().IsDeviceStatHighPrecisionMode()) {
    res.emplace_back(workspace_kernel_tensor);
  }
  res.emplace_back(output_kernel_tensor);
  std::vector<KernelTensor *> workspace;
  if (workspace_kernel_tensor) {
    workspace.emplace_back(workspace_kernel_tensor.get());
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
