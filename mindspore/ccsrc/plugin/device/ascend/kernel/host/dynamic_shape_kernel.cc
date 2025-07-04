/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/host/dynamic_shape_kernel.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
void TensorShapeKernelMod::Execute(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) const {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid TensorShape input or output size (" << inputs.size() << ", " << outputs.size()
                      << ").";
  }

  MS_EXCEPTION_IF_NULL(inputs[0]);
  MS_EXCEPTION_IF_NULL(outputs[0]);
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto shape = inputs[0]->GetShapeVector();
  auto size = shape.size() > 0 ? shape.size() * sizeof(shape[0]) : 0;
  if (device::GetDeviceTypeByName(outputs[0]->device_name()) == device::DeviceType::kCPU) {
    auto ret = memcpy_s(outputs[0]->GetHostData()->addr, outputs[0]->size(), shape.data(), LongToSize(size));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "Execute TensorShapeKernel memcpy_s failed!";
    }
  } else {
    // cppcheck-suppress unreadVariable
    auto lock = device::KernelRuntime::LockRuntime(stream_ptr);
    // Memcpy needs to be synchronized first, if tensor data is from numpy.
    if (!device::ascend::AscendStreamMng::GetInstance().SyncStream(stream_ptr)) {
      MS_EXCEPTION(DeviceProcessError) << "Sync stream error!";
    }

    auto ret_rt_memcpy = CALL_ASCEND_API(aclrtMemcpy, outputs[0]->device_ptr(), outputs[0]->size(), shape.data(),
                                         LongToSize(size), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret_rt_memcpy != ACL_SUCCESS) {
      MS_EXCEPTION(DeviceProcessError) << "aclrtMemcpy failed";
    }
  }
}

bool TensorShapeKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  try {
    Execute(inputs, outputs, stream_ptr);
  } catch (const std::exception &e) {
    MS_LOG(ERROR) << "TensorShapeKernelMod Launch failed. kernel: " << kernel_name_ << ", Error message is "
                  << e.what();
    return false;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
