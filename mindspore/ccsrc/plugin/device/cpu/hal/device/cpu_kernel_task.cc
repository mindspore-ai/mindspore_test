/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/hal/device/cpu_kernel_task.h"
#include "plugin/device/cpu/kernel/contiguous_cpu_kernel.h"
#include "plugin/device/cpu/kernel/copy_with_slice_cpu_kernel.h"
#include "include/backend/mem_reuse/mem_tracker.h"

namespace mindspore::device::cpu {
void MallocMemoryForDeviceAddress(const device::DeviceAddressPtr &device_address,
                                  const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(device_address);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "PyNative", "Contiguous", "");
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                 device_address->GetSize(), device_address.get());
  if (device_address->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(device_address.get())) {
      MS_LOG(EXCEPTION) << "Allocate device memory failed!";
    }
  }
}

bool CpuContiguousKernelTask::RunWithRet() {
  MS_LOG(DEBUG) << "Start";
  auto device_context = context_->device_context();
  MS_EXCEPTION_IF_NULL(device_context);

  const auto &input_address = context_->GetInputAddr(0);
  const auto &output_address = context_->GetOutputAddr(0);
  const auto &input_storage_info = context_->GetInputAddr(0)->GetTensorStorageInfo();
  MS_LOG(DEBUG) << "Input_storage_info:" << (input_storage_info == nullptr ? "" : input_storage_info->ToString())
                << ", input_address size:" << input_address->GetSize()
                << ", output_address size:" << output_address->GetSize();

  MallocMemoryForDeviceAddress(input_address, device_context);
  MallocMemoryForDeviceAddress(output_address, device_context);

  kernel::ContiguousCpuKernel contiguous_kernel;
  auto ret = contiguous_kernel.LaunchContiguous(input_address->type_id(), input_address, input_storage_info,
                                                output_address->type_id(), output_address);
  if (!ret) {
    MS_LOG(EXCEPTION) << "GpuContiguous failed";
  }

  MS_LOG(DEBUG) << "End";
  return true;
}

bool CpuCopyWithSliceKernelTask::RunWithRet() {
  MS_LOG(DEBUG) << "Start";
  auto device_context = context_->device_context();
  MS_EXCEPTION_IF_NULL(device_context);

  const auto &dst_device_address = context_->GetInputAddr(0);
  const auto &src_device_address = context_->GetInputAddr(1);
  MS_EXCEPTION_IF_NULL(dst_device_address);

  const auto &dst_storage_info = context_->GetInputAddr(0)->GetTensorStorageInfo();
  const auto &src_storage_info = context_->GetInputAddr(1)->GetTensorStorageInfo();
  MS_LOG(DEBUG) << "Src_storage_info:" << (src_storage_info == nullptr ? "" : src_storage_info->ToString())
                << ", dst_storage_info:" << (dst_storage_info == nullptr ? "" : dst_storage_info->ToString())
                << ", src address size:" << src_device_address->GetSize()
                << ", dst address size:" << dst_device_address->GetSize();

  MallocMemoryForDeviceAddress(dst_device_address, device_context);
  MallocMemoryForDeviceAddress(src_device_address, device_context);

  kernel::CopyWithSliceCpuKernel copy_kernel;
  auto ret = copy_kernel.LaunchCopyWithSlice(dst_device_address->type_id(), src_storage_info, src_device_address,
                                             dst_storage_info, dst_device_address);
  if (!ret) {
    MS_LOG(EXCEPTION) << "LaunchCopyWithSlice failed";
  }

  MS_LOG(DEBUG) << "End";
  return true;
}
}  // namespace mindspore::device::cpu
