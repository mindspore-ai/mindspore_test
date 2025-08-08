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

#include "plugin/device/ascend/kernel/rts/prefetch.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "kernel/framework_utils.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
bool PrefetchKernel::Init(const AnfNodePtr &anf_node) {
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

bool PrefetchKernel::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_LOG(ERROR) << "call prefetch kernel start.";
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid TensorShape input or output size (" << inputs.size() << ", " << outputs.size()
                      << ").";
  }

  MS_EXCEPTION_IF_NULL(inputs[0]);

  const auto device_ptr = inputs[0]->device_ptr();
  MS_EXCEPTION_IF_NULL(device_ptr);
  auto src_addr = inputs[0]->device_address();
  MS_EXCEPTION_IF_NULL(src_addr);

  auto input_device_addr = std::dynamic_pointer_cast<device::DeviceAddress>(src_addr);
  constexpr size_t offset = 0;
  auto sync = device::ascend::ConvertKernelTensor<bool>(inputs[inputs.size() - 1]);
  MS_EXCEPTION_IF_NULL(input_device_addr);
  return input_device_addr->UpdateRemoteToDevice(device_id_, device_ptr, inputs[0]->size(), stream_ptr, offset, sync);
}
}  // namespace kernel
}  // namespace mindspore
