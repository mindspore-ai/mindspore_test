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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HIERARCHICAL_MEMORY_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HIERARCHICAL_MEMORY_H_
#include "include/backend/common/kernel_graph/kernel_graph.h"
namespace mindspore {
namespace device {
namespace ascend {
namespace hierarchical_memory {
void AdjustExecutionOrderForHierarchicalMemoryOps(const KernelGraphPtr &kernel_graph);
void AddEventToHierarchicalMemoryOps(const KernelGraphPtr &kernel_graph);
void ExecutionOrderOptimizeWithHierarchicalMemory(const KernelGraphPtr &kernel_graph);
}  // namespace hierarchical_memory
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_HARDWARE_HIERARCHICAL_MEMORY_H_
