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

#include "plugin/device/ascend/kernel/rts/copy_to_remote.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "kernel/framework_utils.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"

namespace mindspore {
namespace kernel {
bool CopyToRemoteKernel::Init(const AnfNodePtr &anf_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(anf_node);
  std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(anf_node);

  auto prim = common::AnfAlgo::GetCNodePrimitive(anf_node);
  MS_EXCEPTION_IF_NULL(prim);
  primitive_ = prim;
  kernel_name_ = prim->name();
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (CheckResizeCondition(cnode)) {
    if (Resize(input_kernel_tensors, output_kernel_tensors) == KRET_RESIZE_FAILED) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode) << "#dmsg#Kernel build failed:#dmsg#rts kernel op["
                                         << cnode->fullname_with_scope() << "] Resize failed.";
    }
  }
  return true;
}

bool CopyToRemoteKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(ERROR) << "Begin to call CopyToRemote kernel.";
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid TensorShape input or output size (" << inputs.size() << ", " << outputs.size()
                      << ").";
  }
  const auto input = inputs[0];
  MS_EXCEPTION_IF_NULL(input);
  const auto output = outputs[0];
  MS_EXCEPTION_IF_NULL(output);
  MS_EXCEPTION_IF_NULL(input->device_address());
  MS_EXCEPTION_IF_NULL(output->device_address());
  const auto input_device = input->device_address()->GetDeviceType();
  const auto output_device = output->device_address()->GetDeviceType();
  if (input_device != device::DeviceType::kAscend || output_device != device::DeviceType::kCPU) {
    MS_LOG(EXCEPTION) << "For Primitive '" << kernel_name_
                      << "', the device type of the first input and output must be Ascend(2) and CPU(1)."
                      << "But got input device type: " << input_device << ", output device type: " << output_device;
  }

  // Get input ptr.
  const auto &input_device_ptr = input->device_ptr();
  if (input_device_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "The first input of Primitive '" << kernel_name_
                      << "' has been released before. Please Check!";
  }
  // Get output device ptr.
  const auto output_device_ptr = output->device_ptr();
  MS_EXCEPTION_IF_NULL(output_device_ptr);
  const auto size = input->size();
  auto sync = device::ascend::ConvertKernelTensor<bool>(inputs[inputs.size() - 1]);
  const auto mem_cpy_status = CALL_ASCEND_API(aclrtMemcpyAsync, output_device_ptr, size, input_device_ptr, size,
                                              ACL_MEMCPY_DEVICE_TO_HOST, stream_ptr);
  if (mem_cpy_status != ACL_SUCCESS) {
    MS_LOG(ERROR) << "CopyToRemote kernel aclrtMemcpyAsync device to host failed! src ptr: " << input_device_ptr
                  << ", dst ptr: " << output_device_ptr << ", size: " << size << ", stream: " << stream_ptr;
    return false;
  }
  if (sync) {
    const auto sync_status = CALL_ASCEND_API(aclrtSynchronizeStreamWithTimeout, stream_ptr, -1);
    if (sync_status != ACL_SUCCESS) {
      MS_LOG(ERROR) << "Failed to synchronize stream, ret = " << sync_status << ".";
      return false;
    }
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
