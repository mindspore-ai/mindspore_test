/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_UTILS_H_
#define MINDSPORE_CCSRC_DEBUG_DEBUGGER_DEBUGGER_UTILS_H_

#include <iostream>
#include <string>
#include <vector>
#include "include/backend/debug/debugger/debugger.h"
#include "common/kernel.h"
#include "runtime/hardware/device_context.h"
#include "include/backend/visible.h"

using mindspore::device::DeviceContext;
using mindspore::kernel::KernelLaunchAddr;
using mindspore::kernel::KernelTensor;

namespace mindspore {

std::vector<size_t> GetValidDumpIndex(const CNodePtr &cnode, size_t index_size, bool is_input,
                                      const DeviceContext *device_context,
                                      const std::vector<KernelTensor *> &tensors = {});

// when used in abnormal dump, the async_copy should set to false
void LoadInputs(const CNodePtr &cnode, std::vector<KernelTensor *> device_tensors, uint32_t exec_order,
                uint32_t root_graph_id, const DeviceContext *device_context, const bool trans_flag,
                const uint32_t sample_mode, const uint32_t sample_num, const bool async_copy = true);

void LoadOutputs(const CNodePtr &cnode, std::vector<KernelTensor *> device_tensors, uint32_t exec_order,
                 uint32_t root_graph_id, const DeviceContext *device_context, const bool trans_flag,
                 const uint32_t sample_mode, const uint32_t sample_num);

BACKEND_COMMON_EXPORT bool CheckReadData(const CNodePtr &cnode);

BACKEND_COMMON_EXPORT void ReadDataAndDump(const CNodePtr &cnode,
                                           std::vector<kernel::KernelTensor *> input_kernel_tensors,
                                           std::vector<kernel::KernelTensor *> output_kernel_tensors,
                                           uint32_t exec_order, const DeviceContext *device_context,
                                           const bool abnormal_dump = false);

BACKEND_COMMON_EXPORT void DumpDataViaCallback(const CNodePtr &cnode,
                                               const std::vector<KernelTensor *> &input_device_tensors,
                                               const std::vector<KernelTensor *> &output_kernel_tensors,
                                               const DeviceContext *device_context);

BACKEND_COMMON_EXPORT std::string CheckDatasetSinkMode(const KernelGraphPtr &graph_ptr);

// get the full name of a tensor, which is the name used in TensorLoader
std::string GetTensorFullName(const debugger::TensorProto &tensor);
}  // namespace mindspore
#endif
