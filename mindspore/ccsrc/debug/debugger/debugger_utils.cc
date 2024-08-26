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

#include "debug/debugger/debugger_utils.h"
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "backend/common/session/session_basic.h"
#include "debug/data_dump/tensor_statistic.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#include "include/backend/debug/debugger/debugger.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/config_manager.h"
#include "kernel/kernel.h"

constexpr int kFailure = 1;

using mindspore::kernel::AddressPtr;
using mindspore::kernel::KernelLaunchAddr;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;
using KernelGraph = mindspore::session::KernelGraph;
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

namespace mindspore {
/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Returns a vector containing real output number.
 */
std::vector<size_t> CheckRealOutput(const std::string &node_name, const size_t &output_size) {
  std::vector<size_t> real_outputs;
  // P.BatchNorm is used for training and inference
  // can add the filter list for more operators here....
  if (node_name == "BatchNorm") {
    MS_LOG(INFO) << "loading node named " << node_name;
    (void)real_outputs.insert(real_outputs.cend(), {0, 3, 4});
  } else {
    // by default, TensorLoader will load all outputs
    for (size_t j = 0; j < output_size; ++j) {
      real_outputs.push_back(j);
    }
  }
  return real_outputs;
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU, Ascend.
 * Runtime category: MindRT.
 * Description: Get Valid Tensor indexes.
 */
vector<size_t> GetValidDumpIndex(const CNodePtr &cnode, size_t index_size, bool is_input) {
  std::vector<size_t> valid_indexes;
  valid_indexes.reserve(index_size);
  if (is_input) {
    std::vector<size_t> ignored_address;
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    if (kernel_mod != nullptr) {
      ignored_address = kernel_mod->GetLaunchIgnoredInputAddressIdx();
    }
    std::set<size_t> ignored_address_set(ignored_address.begin(), ignored_address.end());
    for (size_t index = 0; index < index_size; ++index) {
      if (ignored_address_set.find(index) != ignored_address_set.end()) {
        continue;
      }
      valid_indexes.push_back(index);
    }
  } else {
    auto node_name = common::AnfAlgo::GetCNodeName(cnode);
    valid_indexes = CheckRealOutput(node_name, index_size);
  }
  return valid_indexes;
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU, Ascend.
 * Runtime category: MindRT.
 * Description: Get kernel inputs from device_tensors and load the inputs from device to host.
 */
void LoadInputs(const CNodePtr &cnode, std::vector<device::DeviceAddress *> device_tensors, uint32_t exec_order,
                uint32_t root_graph_id, const DeviceContext *device_context, const bool trans_flag,
                const uint32_t sample_mode, const uint32_t sample_num, const bool async_copy) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(device_context);
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  std::vector<size_t> ignored_address;
  if (kernel_mod != nullptr) {
    ignored_address = kernel_mod->GetLaunchIgnoredInputAddressIdx();
  }

  auto input_size = device_tensors.size();
  for (size_t j = 0; j < input_size; ++j) {
    // Ignore the input address that is not used in the kernel launch.
    if (std::find(ignored_address.begin(), ignored_address.end(), j) != ignored_address.end()) {
      MS_LOG(INFO) << "Ignore dump input data for kernel:" << cnode->fullname_with_scope() << " with input index:" << j;
      continue;
    }
    auto input_kernel = cnode->input(j + 1);
    std::string input_kernel_name = GetKernelNodeName(input_kernel);
    auto device_type = AnfAlgo::GetOutputDeviceDataType(input_kernel, kParameterOutputIndex);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(input_kernel, kParameterOutputIndex);
    auto type = trans_flag ? host_type : device_type;
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }
    auto host_format = kOpFormat_DEFAULT;
    auto device_format =
      E2eDump::IsDeviceTargetGPU() ? kOpFormat_DEFAULT : AnfAlgo::GetOutputFormat(input_kernel, kParameterOutputIndex);

    string input_tensor_name = input_kernel_name + ':' + "0";
    auto device_addr = device_tensors[j];

    auto dump_shape = device_addr->kernel_tensor()->GetShapeVector();  // host shape
    if (!trans_flag) {
      dump_shape = AnfAlgo::GetOutputDeviceShape(input_kernel, kParameterOutputIndex, dump_shape);  // device_shape
    }
    if (sample_mode == DumpJsonParser::DUMP_HEAD_AND_TAIL && SizeOf(dump_shape) > sample_num) {
      dump_shape = {sample_num};
    }
    auto ret = device_addr->LoadMemToHost(input_tensor_name, UintToInt(exec_order), host_format, dump_shape, type, 0,
                                          true, root_graph_id, false, trans_flag, async_copy);
    if (!ret) {
      MS_LOG(WARNING) << "LoadMemToHost failed: tensor_name:" << input_tensor_name << ", host_format:" << host_format
                      << ", device_format:" << device_format << ".";
    }
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU, Ascend.
 * Runtime category: MindRT.
 * Description: Get kernel outputs from device_tensors and load the inputs from device to host.
 */
void LoadOutputs(const CNodePtr &cnode, std::vector<device::DeviceAddress *> device_tensors, uint32_t exec_order,
                 uint32_t root_graph_id, const DeviceContext *device_context, const bool trans_flag,
                 const uint32_t sample_mode, const uint32_t sample_num) {
  auto output_size = AnfAlgo::GetOutputTensorNum(cnode);
  auto node_name = common::AnfAlgo::GetCNodeName(cnode);
  std::string kernel_name = GetKernelNodeName(cnode);
  std::vector<size_t> real_outputs = CheckRealOutput(node_name, output_size);
  for (size_t j : real_outputs) {
    auto device_type = AnfAlgo::GetOutputDeviceDataType(cnode, j);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(cnode, j);
    auto type = trans_flag ? host_type : device_type;
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }

    auto host_format = kOpFormat_DEFAULT;
    auto device_format = E2eDump::IsDeviceTargetGPU() ? kOpFormat_DEFAULT : AnfAlgo::GetOutputFormat(cnode, j);

    string tensor_name = kernel_name + ':' + std::to_string(j);
    auto device_addr = device_tensors[j];
    auto dump_shape = device_addr->kernel_tensor()->GetShapeVector();
    if (!trans_flag) {
      dump_shape = AnfAlgo::GetOutputDeviceShape(cnode, j, dump_shape);
    }
    if (sample_mode == DumpJsonParser::DUMP_HEAD_AND_TAIL && SizeOf(dump_shape) > sample_num) {
      dump_shape = {sample_num};
    }
    auto ret = device_addr->LoadMemToHost(tensor_name, UintToInt(exec_order), host_format, dump_shape, type, j, false,
                                          root_graph_id, false, trans_flag);
    if (!ret) {
      MS_LOG(WARNING) << "LoadMemToHost failed: tensor_name:" << tensor_name << ", host_format:" << host_format
                      << ", device_format:" << device_format << ".!";
    }
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Returns true if the node needs to be read for Dump or online debugger. This function is used by GPU
 * and Ascend kernel-by-kernel mindRT.
 */
bool CheckReadData(const CNodePtr &cnode) {
  auto debugger = Debugger::GetInstance();
  if (!debugger) {
    return false;
  }
  bool read_data = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool dump_enabled = dump_json_parser.DumpEnabledForIter();
  MS_LOG(DEBUG) << "dump_enabled: " << dump_enabled;
  std::string kernel_name = GetKernelNodeName(cnode);
  if (dump_enabled) {
    if (dump_json_parser.NeedDump(kernel_name)) {
      read_data = true;
    }
  }
  return read_data;
}

bool IsDeviceTargetGPU() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice;
}

bool GetTransFlag() {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (IsDeviceTargetGPU()) {
    return true;
  }
  return DumpJsonParser::GetInstance().trans_flag();
}

uint32_t GetSampleMode() {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (IsDeviceTargetGPU()) {
    return 0;
  }
  return DumpJsonParser::GetInstance().sample_mode();
}

uint32_t GetSampleNum() {
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (IsDeviceTargetGPU()) {
    return 0;
  }
  return DumpJsonParser::GetInstance().sample_num();
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Load inputs and outputs of the given node if needed and dump them if dump is enabled, then it performs
 * PostExecuteNode function on the given node for GPU.
 */
void ReadDataAndDump(const CNodePtr &cnode, std::vector<device::DeviceAddress *> input_device_tensors,
                     std::vector<device::DeviceAddress *> output_device_tensors, uint32_t exec_order,
                     const DeviceContext *device_context, const bool abnormal_dump) {
  auto debugger = Debugger::GetInstance();
  if (!debugger) {
    return;
  }
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool dump_enabled = dump_json_parser.DumpEnabledForIter();
  MS_LOG(DEBUG) << "dump_enabled: " << dump_enabled;
  auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(cnode->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto root_graph_id = kernel_graph->root_graph_id();
  bool trans_flag = GetTransFlag();
  uint32_t sample_mode = GetSampleMode();
  uint32_t sample_num = GetSampleNum();
  if (dump_json_parser.InputNeedDump()) {
    if (DumpJsonParser::GetInstance().IsDeviceCalcStats() && dump_enabled) {
      datadump::DumpKernelTensorStats(device_context, input_device_tensors, true, cnode, root_graph_id);
    } else {
      bool async_copy = !abnormal_dump;
      LoadInputs(cnode, input_device_tensors, exec_order, root_graph_id, device_context, trans_flag, sample_mode,
                 sample_num, async_copy);
    }
  }
  if (dump_json_parser.OutputNeedDump()) {
    if (DumpJsonParser::GetInstance().IsDeviceCalcStats() && dump_enabled) {
      datadump::DumpKernelTensorStats(device_context, output_device_tensors, false, cnode, root_graph_id);
    } else if (!abnormal_dump) {
      LoadOutputs(cnode, output_device_tensors, exec_order, root_graph_id, device_context, trans_flag, sample_mode,
                  sample_num);
    }
  }
  // Dump kernel
  if (dump_enabled && !DumpJsonParser::GetInstance().IsDeviceCalcStats()) {
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto graph_id = kernel_graph->graph_id();
    // for GPU, nodes are dumped in graph_id directory.
    if (IsDeviceTargetGPU()) {
      debugger->DumpSingleNode(cnode, graph_id);
    } else {
      // for Ascend, node are dumped in root_graph_id directory.
      debugger->DumpSingleNode(cnode, root_graph_id);
    }
    debugger->ClearCurrentData();
  }
}

/*
 * Feature group: Dump, Online Debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Returns the error_info when sink_mode is true and we are in online debugger mode or dump mode for
 * GPU, if everything is normal the error_info string will be empty.
 */
std::string CheckDatasetSinkMode(const KernelGraphPtr &graph_ptr) {
  std::string error_info = "";
  bool sink_mode =
    ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE || graph_ptr->IsDatasetGraph();
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (debugger->CheckDebuggerDumpEnabled() && sink_mode && IsDeviceTargetGPU()) {
    error_info = "e2e_dump is not supported on GPU with dataset_sink_mode=True. Please set dataset_sink_mode=False";
  }
  return error_info;
}

void Dump(const KernelGraphPtr &graph, uint32_t rank_id) {
  MS_LOG(DEBUG) << "Start!";
  MS_EXCEPTION_IF_NULL(graph);
  E2eDump::DumpData(graph.get(), rank_id);
  MS_LOG(DEBUG) << "Finish!";
}

uint32_t GetRankID() {
  uint32_t rank_id = 0;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto env_rank_id = common::GetEnv("RANK_ID");
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) && !env_rank_id.empty()) {
    // get actual rank id if it's distribution training case.
    rank_id = GetRankId();
  }
  return rank_id;
}

std::string GetTensorFullName(const debugger::TensorProto &tensor) {
  string node_name = tensor.node_name();
  if (tensor.truncate()) {
    // scopes in node name are separated by '/'
    // use the name without scope if truncate is true
    std::size_t found = node_name.find_last_of("/");
    node_name = node_name.substr(found + 1);
  }
  return node_name + ":" + tensor.slot() + (tensor.iter() == "" ? "" : ":" + tensor.iter());
}
}  // namespace mindspore
