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

#include "plugin/device/ascend/kernel/rts/free_device.h"

#include <string>
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "kernel/framework_utils.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "runtime/device/res_manager/utils/utils.h"

namespace mindspore {
namespace kernel {
bool FreeDeviceKernel::Init(const AnfNodePtr &anf_node) {
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

bool FreeDeviceKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(ERROR) << "Begin to call FreeDevice kernel.";
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid TensorShape input or output size (" << inputs.size() << ", " << outputs.size()
                      << ").";
  }
  const auto input = inputs[0];
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(input->device_address());
  const auto input_device = input->device_address()->GetDeviceType();
  if (input_device != device::DeviceType::kAscend) {
    MS_LOG(EXCEPTION) << "For Primitive '" << kernel_name_ << "', the device type of the first input must be CPU(1)."
                      << "But got input device type: " << input_device;
  }

  // Get src input ptr.
  const auto &input_device_ptr = input->device_ptr();
  if (input_device_ptr == nullptr) {
    MS_LOG(WARNING) << "the input of Primitive '" << kernel_name_ << "' has been released before. Please Check!";
    return true;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::ResKey res_key{device::GetDeviceTypeByName(device_name), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);
  MS_EXCEPTION_IF_NULL(input->device_address());
  MS_LOG(ERROR) << "Try to release input memory, the addr: " << input_device_ptr;
  res_manager->FreeMemory(input->device_address().get());
  return true;
}
}  // namespace kernel
}  // namespace mindspore
