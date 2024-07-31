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
#include "include/backend/debug/data_dump/tensor_stat_dump.h"
#include "debug/data_dump/tensor_info_collect.h"

constexpr int kFailure = 1;
constexpr auto kInput = "input";
constexpr auto kOutput = "output";

using mindspore::kernel::AddressPtr;
using mindspore::kernel::KernelLaunchAddr;
using AddressPtrList = std::vector<mindspore::kernel::AddressPtr>;
using KernelGraph = mindspore::session::KernelGraph;
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;

namespace mindspore {
using mindspore::TensorInfoCommForDump;
using mindspore::TensorInfoForDump;

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

bool IsDeviceTargetGPU() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice;
}

bool GetTransFlag() {
  if (IsDeviceTargetGPU()) {
    return true;
  }
  return DumpJsonParser::GetInstance().trans_flag();
}

uint32_t GetSampleMode() {
  if (IsDeviceTargetGPU()) {
    return 0;
  }
  return DumpJsonParser::GetInstance().sample_mode();
}

uint32_t GetSampleNum() {
  if (IsDeviceTargetGPU()) {
    return 0;
  }
  return DumpJsonParser::GetInstance().sample_num();
}

inline TypeId GetInputKernelType(const AnfNodePtr &input_kernel, bool trans_flag) {
  auto device_type = AnfAlgo::GetOutputDeviceDataType(input_kernel, kParameterOutputIndex);
  auto host_type = common::AnfAlgo::GetOutputInferDataType(input_kernel, kParameterOutputIndex);
  auto type = trans_flag ? host_type : device_type;
  return type;
}

inline TypeId GetOutputKernelType(const CNodePtr &cnode, size_t j, bool trans_flag) {
  auto device_type = AnfAlgo::GetOutputDeviceDataType(cnode, j);
  auto host_type = common::AnfAlgo::GetOutputInferDataType(cnode, j);
  auto type = trans_flag ? host_type : device_type;
  return type;
}

inline ShapeVector SampleDumpShape(const ShapeVector &dump_shape) {
  auto sample_mode = GetSampleMode();
  auto sample_num = GetSampleNum();
  if (sample_mode == DumpJsonParser::DUMP_HEAD_AND_TAIL && SizeOf(dump_shape) > sample_num) {
    ShapeVector sample_shape = {sample_num};
    return sample_shape;
  }
  return dump_shape;
}

inline ShapeVector GetOutputKernelShapeVec(const CNodePtr &cnode, device::DeviceAddress *device_tensor, size_t j,
                                           bool trans_flag) {
  auto dump_shape = device_tensor->kernel_tensor()->GetShapeVector();
  if (!trans_flag) {
    dump_shape = AnfAlgo::GetOutputDeviceShape(cnode, j, dump_shape);
  }
  dump_shape = SampleDumpShape(dump_shape);
  return dump_shape;
}

inline ShapeVector GetInputKernelShapeVec(const AnfNodePtr &input_kernel, device::DeviceAddress *device_tensor,
                                          size_t j, bool trans_flag) {
  auto dump_shape = device_tensor->kernel_tensor()->GetShapeVector();
  if (!trans_flag) {
    dump_shape = AnfAlgo::GetOutputDeviceShape(input_kernel, kParameterOutputIndex, dump_shape);
  }
  dump_shape = SampleDumpShape(dump_shape);
  return dump_shape;
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
    auto type = GetInputKernelType(input_kernel, trans_flag);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }
    auto host_format = kOpFormat_DEFAULT;
    auto device_format =
      E2eDump::IsDeviceTargetGPU() ? kOpFormat_DEFAULT : AnfAlgo::GetOutputFormat(input_kernel, kParameterOutputIndex);

    string input_tensor_name = input_kernel_name + ':' + "0";
    auto device_addr = device_tensors[j];

    auto dump_shape = GetInputKernelShapeVec(input_kernel, device_addr, j, trans_flag);

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
    auto type = GetOutputKernelType(cnode, j, trans_flag);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }

    auto host_format = kOpFormat_DEFAULT;
    auto device_format = E2eDump::IsDeviceTargetGPU() ? kOpFormat_DEFAULT : AnfAlgo::GetOutputFormat(cnode, j);

    string tensor_name = kernel_name + ':' + std::to_string(j);
    auto device_addr = device_tensors[j];
    auto dump_shape = GetOutputKernelShapeVec(cnode, device_addr, j, trans_flag);

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

inline std::shared_ptr<TensorData> PrepareStatTensorData(mindspore::tensor::TensorPtr out_tensor,
                                                         const TensorInfoForDump &tensor_info) {
  std::shared_ptr<TensorData> tensor_data = std::make_shared<TensorData>();
  tensor_data->SetTensor(out_tensor);
  tensor_data->SetDataPtr(static_cast<char *>(out_tensor->data_c()));
  tensor_data->SetByteSize(LongToSize(out_tensor->data().nbytes()));
  tensor_data->SetType(tensor_info.host_type);
  tensor_data->SetShape(out_tensor->shape());
  tensor_data->SetFormat(tensor_info.format);
  return tensor_data;
}

void LaunchDumpCallback(const std::vector<TensorInfoForDump> &tensor_info_list, const DeviceContext *device_context,
                        uint32_t stream_id, const TensorInfoCommForDump &tensor_info_comm) {
  bool dump_tensor = DumpJsonParser::GetInstance().IsTensorDump();
  bool dump_host_stat =
    (DumpJsonParser::GetInstance().IsStatisticDump() && !DumpJsonParser::GetInstance().IsDeviceCalcStats());
  if (!dump_tensor && !dump_host_stat) {
    return;
  }
  device::CallbackFunc callback_func = [tensor_info_list, device_context, stream_id, tensor_info_comm, dump_tensor,
                                        dump_host_stat]() {
    for (const auto &tensor_info : tensor_info_list) {
      MS_EXCEPTION_IF_NULL(tensor_info.device_tensor);

      auto host_type = tensor_info.host_type;
      if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin ||
          host_type == kNumberTypeComplex64) {
        MS_LOG(INFO) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
        continue;
      }

      uint64_t timestamp = Common::GetTimeStamp();
      std::string file_path = tensor_info_comm.file_path_prefix + '.' + std::to_string(timestamp) + '.' +
                              tensor_info.io + '.' + std::to_string(tensor_info.io_index) + '.' + tensor_info.format;

      auto host_shape = tensor_info.host_shape;
      mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
      MS_EXCEPTION_IF_NULL(out_tensor);
      size_t host_size = LongToSize(out_tensor->data().nbytes());
      if (host_size == 0) {
        MS_LOG(WARNING) << "Dump tensor size is 0 for tensor: " << file_path << ". Skip it";
        continue;
      }
      size_t device_size = tensor_info.device_size;
      if (host_size > device_size) {
        MS_LOG(ERROR) << "Dump host size " << host_size << " greater than device size " << device_size;
        continue;
      }
      auto ret_rt_memcpy = tensor_info.device_tensor->CallAclrtMemcpy(out_tensor->data_c(), host_size,
                                                                      tensor_info.device_ptr, tensor_info.device_size);
      MS_LOG(DEBUG) << "Callback aclrtmemcpy for " << file_path << ". result is: " << ret_rt_memcpy << file_path;

      if (dump_tensor) {
        DumpJsonParser::DumpToFile(file_path, out_tensor->data_c(), host_size, host_shape, host_type);
      }

      if (dump_host_stat) {
        auto tensor_data = PrepareStatTensorData(out_tensor, tensor_info);

        bool is_input = (tensor_info.io == kInput);
        TensorStatDump stat_dump(tensor_info_comm.op_type, tensor_info_comm.op_name, tensor_info_comm.task_id,
                                 stream_id, timestamp, is_input, tensor_info.io_index, 0);
        stat_dump.DumpTensorStatsToFile(tensor_info_comm.dump_path, tensor_data);
      }
    }
  };

  auto callback_ret = device_context->GetKernelExecutor(false)->LaunchCallback(callback_func, stream_id, true);
  if (!callback_ret) {
    MS_LOG(ERROR) << "Async dump callback launch fail.";
  }
}

void PrepareInputDataViaCallback(const CNodePtr &cnode,
                                 const std::vector<device::DeviceAddress *> &input_device_tensors,
                                 std::vector<TensorInfoForDump> *tensor_info_list) {
  auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
  bool trans_flag = GetTransFlag();

  std::vector<size_t> ignored_address;
  if (kernel_mod != nullptr) {
    ignored_address = kernel_mod->GetLaunchIgnoredInputAddressIdx();
  }

  for (size_t j = 0; j < input_device_tensors.size(); ++j) {
    // Ignore the input address that is not used in the kernel launch.
    if (std::find(ignored_address.begin(), ignored_address.end(), j) != ignored_address.end()) {
      MS_LOG(INFO) << "Ignore dump input data for kernel:" << cnode->fullname_with_scope() << " with input index:" << j;
      continue;
    }
    auto input_kernel = cnode->input(j + 1);
    auto &device_tensor = input_device_tensors[j];
    MS_EXCEPTION_IF_NULL(device_tensor);

    auto type = GetInputKernelType(input_kernel, trans_flag);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }

    auto dump_shape = GetInputKernelShapeVec(input_kernel, device_tensor, j, trans_flag);
    auto host_format = kOpFormat_DEFAULT;
    auto format = trans_flag ? host_format : device_tensor->format();

    tensor_info_list->emplace_back(
      TensorInfoForDump(kInput, j, format, type, dump_shape, device_tensor->GetSize(), device_tensor));
  }
}

void PrepareOutputDataViaCallback(const CNodePtr &cnode,
                                  const std::vector<device::DeviceAddress *> &output_device_tensors,
                                  std::vector<TensorInfoForDump> *tensor_info_list) {
  auto output_size = AnfAlgo::GetOutputTensorNum(cnode);
  auto node_name = common::AnfAlgo::GetCNodeName(cnode);
  bool trans_flag = GetTransFlag();

  std::string kernel_name = GetKernelNodeName(cnode);
  std::vector<size_t> real_outputs = CheckRealOutput(node_name, output_size);
  for (size_t j : real_outputs) {
    auto type = GetOutputKernelType(cnode, j, trans_flag);
    // For example, this happens with the Depend op
    if (type == kMetaTypeNone) {
      continue;
    }

    auto &device_tensor = output_device_tensors[j];
    MS_EXCEPTION_IF_NULL(device_tensor);

    auto dump_shape = GetOutputKernelShapeVec(cnode, device_tensor, j, trans_flag);

    auto host_format = kOpFormat_DEFAULT;
    auto format = trans_flag ? host_format : device_tensor->format();
    tensor_info_list->emplace_back(
      TensorInfoForDump(kOutput, j, format, type, dump_shape, device_tensor->GetSize(), device_tensor));
  }
}

TensorInfoCommForDump GetTensorInfoCommFromCnode(const CNodePtr &cnode) {
  auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(cnode->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto root_graph_id = kernel_graph->root_graph_id();

  uint32_t rank_id = GetRankId();
  std::string dump_path = GenerateDumpPath(root_graph_id, rank_id);
  std::string op_type = common::AnfAlgo::GetCNodeName(cnode);
  std::string op_name = GetKernelNodeName(cnode);
  GetFileKernelName(NOT_NULL(&op_name));

  uint32_t task_id = 0;
  auto stream_id = AnfAlgo::GetStreamId(cnode);

  TensorInfoCommForDump tensor_info_comm(dump_path, op_type, op_name, task_id, stream_id);
  return tensor_info_comm;
}

void DumpDataViaCallback(const CNodePtr &cnode, const std::vector<device::DeviceAddress *> &input_device_tensors,
                         const std::vector<device::DeviceAddress *> &output_device_tensors,
                         const DeviceContext *device_context) {
  auto debugger = Debugger::GetInstance();
  if (!debugger) {
    return;
  }
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool dump_enabled = dump_json_parser.DumpEnabledForIter();
  MS_LOG(DEBUG) << "dump_enabled: " << dump_enabled;

  TensorInfoCommForDump tensor_info_comm = GetTensorInfoCommFromCnode(cnode);
  auto stream_id = tensor_info_comm.stream_id;

  std::vector<TensorInfoForDump> tensor_info_list;
  PrepareInputDataViaCallback(cnode, input_device_tensors, &tensor_info_list);
  PrepareOutputDataViaCallback(cnode, output_device_tensors, &tensor_info_list);

  LaunchDumpCallback(tensor_info_list, device_context, stream_id, tensor_info_comm);
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
