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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_MEMORY_ALLOCATOR_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_MEMORY_ALLOCATOR_H_

#include <memory>
#include <string>
#include <set>
#include "include/backend/kernel_graph.h"
#include "include/transform/graph_ir/types.h"
#include "runtime/hardware/device_context.h"
#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_summary.h"

namespace mindspore {
namespace device {
namespace ascend {
class GEMemoryAllocator {
 public:
  static void ProcessGraphDeviceAddress(const KernelGraphPtr &kernel_graph, DeviceContext *device_context,
                                        GeDeviceResManager *res_manager);
  static void AllocInputHostMemory(const KernelGraphPtr &kernel_graph, DeviceContext *device_context);
  static void AllocOutputHostMemory(const KernelGraphPtr &kernel_graph, DeviceContext *device_context);
  static void AllocGraphMemory(const transform::RunOptions &options, const KernelGraphPtr &graph,
                               const GraphSummary &summary, size_t stream_id, GeDeviceResManager *res_manager);
  static void AllocUnuseInput(const KernelGraphPtr &kernel_graph, const AnfNodePtr &input_node,
                              DeviceAddress *output_addr, GeDeviceResManager *res_manager);
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_MEMORY_ALLOCATOR_H_
