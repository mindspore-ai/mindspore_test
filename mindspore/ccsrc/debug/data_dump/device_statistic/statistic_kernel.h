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

#ifndef MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_STATISTIC_KERNEL_H_
#define MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_STATISTIC_KERNEL_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "debug/data_dump/device_statistic/common.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/common/debug/common.h"
#include "ir/dtype/tensor_type.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "runtime/hardware/device_context.h"
#include "utils/log_adapter.h"

namespace mindspore {

namespace datadump {
using device::DeviceAddressPtr;
using kernel::KernelTensor;
using kernel::KernelTensorPtr;
using mindspore::device::DeviceContext;
using TensorPtr = tensor::TensorPtr;

class StatisticKernel {
 public:
  StatisticKernel(const DeviceContext *device_context, const string &kernel_name, std::set<TypeId> dtype_id)
      : device_context_(device_context), kernel_name_(kernel_name), supported_dtype_(dtype_id) {
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context_->device_res_manager_);
    MS_VLOG(VL_DUMP) << "Statistic kernel mod " << kernel_name_ << " construct.";
    kernel_mod_ = device_context_->GetKernelExecutor(false)->CreateKernelMod(kernel_name);
    MS_EXCEPTION_IF_NULL(kernel_mod_);
  }
  std::vector<DeviceAddressPtr> LaunchKernelAsync(KernelTensor *input, const uint32_t stream_id);
  virtual DeviceAddressPtr LaunchKernelAsync(std::vector<KernelTensor *> inputs, const uint32_t stream_id) {
    return nullptr;
  }

  bool CheckDataType(const TypeId &dtype_id) { return supported_dtype_.find(dtype_id) != supported_dtype_.end(); }

 protected:
  DeviceAddressPtr GenerateDeviceAddress(const size_t &mem_size, const TypeId &dtype_id, const ShapeVector &shape);
  DeviceAddressPtr GetWorkSpaceDeviceAddress(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs);
  virtual DeviceAddressPtr GetOutputDeviceAddress(TypeId dtype_id);
  virtual std::vector<KernelTensorPtr> GetExtraInputsDeviceAddress(KernelTensor *);
  const DeviceContext *device_context_{nullptr};
  string kernel_name_;
  std::set<TypeId> supported_dtype_;
  uint32_t stream_id_ = kDefaultStreamIndex;
  kernel::KernelModPtr kernel_mod_;
};

TensorPtr SyncDeviceToHostTensor(DeviceAddressPtr device_addr);

}  // namespace datadump

}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_STATISTIC_KERNEL_H_
