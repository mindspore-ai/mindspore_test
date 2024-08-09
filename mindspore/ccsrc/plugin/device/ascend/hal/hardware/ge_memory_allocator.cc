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
#include "plugin/device/ascend/hal/hardware/ge_memory_allocator.h"
#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include "include/transform/graph_ir/utils.h"
#include "include/common/utils/utils.h"
#include "include/common/debug/draw.h"
#include "include/common/debug/anf_ir_dump.h"
#include "abstract/abstract_value.h"
#include "include/backend/kernel_graph.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/device/device_address_utils.h"
#include "plugin/device/cpu/hal/device/cpu_memory_manager.h"
#include "plugin/device/ascend/hal/hccl_adapter/hccl_adapter.h"
#include "plugin/device/ascend/optimizer/ge_optimization.h"
#include "plugin/device/ascend/hal/common/ascend_utils.h"
#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_memory_manager.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_device_synchronizer.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "kernel/kernel_build_info.h"
#include "ops/array_ops.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace {
constexpr size_t kNeedRecycleOutput = 5;

void UpdateTracker(const std::string &task_name, const std::string &node_name, const std::string &graph_str,
                   size_t size, void *device_ptr, device::tracker::MemType mem_type) {
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, task_name, node_name, graph_str);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, task_name, size, device_ptr, mem_type);
}

std::multimap<std::string, ParameterPtr> FilterAllParameters(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::multimap<std::string, ParameterPtr> ret;
  std::vector<AnfNodePtr> todo = kernel_graph->input_nodes();
  (void)todo.insert(todo.end(), kernel_graph->child_graph_result().begin(), kernel_graph->child_graph_result().end());
  for (const auto &node : todo) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto parameter = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    std::string name = parameter->name();
    (void)ret.emplace(name, parameter);
  }
  return ret;
}

void SetParameterKernelInfo(const AnfNodePtr &node, const std::shared_ptr<device::KernelInfo> &kernel_info) {
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (!build_info) {
    MS_LOG(ERROR) << "Parameter doesn't have build info: " << node->DebugString()
                  << ", full name: " << node->fullname_with_scope();
    return;
  }
  std::vector<TypeId> refresh_output_types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  build_info->SetOutputsDeviceType(refresh_output_types);
}

void SetKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // If kernel build info has been set up. skip
  std::shared_ptr<device::KernelInfo> kernel_info =
    std::dynamic_pointer_cast<device::KernelInfo>(node->kernel_info_ptr());
  if (utils::isa<ParameterPtr>(node)) {
    SetParameterKernelInfo(node, kernel_info);
    return;
  }

  if (!kernel_info) {
    kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    node->set_kernel_info(kernel_info);
  }

  auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (!build_info) {
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    build_info = builder->Build();
  }

  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(node);
  std::vector<TypeId> output_infer_types;
  std::vector<std::string> output_formats;
  for (const auto &output_with_index : output_with_indexs) {
    (void)output_infer_types.emplace_back(
      common::AnfAlgo::GetOutputInferDataType(output_with_index.first, output_with_index.second));
    (void)output_formats.emplace_back(kOpFormat_DEFAULT);
  }
  build_info->SetOutputsDeviceType(output_infer_types);
  build_info->SetOutputsFormat(output_formats);
  kernel_info->set_select_kernel_build_info(build_info);
}

DeviceAddressPtr CreateOutputDeviceAddress(const KernelGraphPtr &kernel_graph, const KernelWithIndex &output_with_index,
                                           size_t need_alloc_output_cnt, GeDeviceResManager *res_manager) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto output_node = output_with_index.first;
  MS_EXCEPTION_IF_NULL(output_node);
  auto ref_map = kernel_graph->GetRefMap();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto real_index = output_node->isa<ValueNode>() ? 0 : output_with_index.second;
  TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(output_node, real_index);
  size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
  auto shapes = trans::GetRuntimePaddingShape(output_node, real_index);
  auto tensor_size =
    shapes.empty() ? type_size : std::accumulate(shapes.begin(), shapes.end(), type_size, std::multiplies<size_t>());
  // When ValueNode is a graph output, runtime does not manage this memory
  // output in ref_map, mem same is input
  // ge kernel no need alloc memory
  bool need_not_alloc = (kernel_graph->has_flag(kFlagEnableZeroCopyInGraph) && !output_node->isa<ValueNode>()) ||
                        (ref_map.find(output_with_index) != ref_map.end()) || kernel_graph->has_flag(kFlagGeKernel);
  MS_EXCEPTION_IF_NULL(res_manager);
  void *mem = need_not_alloc ? nullptr : res_manager->AllocateMemory(tensor_size);

  if (common::IsNeedProfileMemory() && !need_not_alloc) {
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: ValueNodeOutput, size:" << tensor_size
                    << ", graph: " << kernel_graph->ToString() << ", node: " << output_node->fullname_with_scope()
                    << ", device address addr: " << mem;
  }
  if (!need_not_alloc) {
    UpdateTracker("ValueNodeOutput", output_node->fullname_with_scope(), kernel_graph->ToString(), tensor_size, mem,
                  device::tracker::MemType::kConstantValue);
  }

  const auto kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {output_node, real_index}, mem, tensor_size, kOpFormat_DEFAULT, output_type_id, {}, kAscendDevice, device_id);
  auto output_device_addr = std::make_shared<AscendDeviceAddress>(kernel_tensor);
  if (ref_map.find(output_with_index) != ref_map.end()) {
    auto input_with_index = ref_map[output_with_index];
    auto input_device_address = AnfAlgo::GetMutableOutputAddr(input_with_index.first, input_with_index.second, false);
    MS_EXCEPTION_IF_NULL(input_device_address);
    MS_LOG(INFO) << "The output node " << output_node->fullname_with_scope()
                 << " is in ref_map, set the same device_address ptr as the corresponding input, input node: "
                 << input_with_index.first->fullname_with_scope();
    // Update the reference count of device address.
    output_device_addr->set_pointer_ref_count(input_device_address->pointer_ref_count());
    output_device_addr->IncreaseOriginalRefCount();
    output_device_addr->ResetRefCount();
  }
  output_device_addr->set_device_synchronizer(std::make_shared<AscendDeviceSynchronizer>());
  output_device_addr->set_is_ptr_persisted(true);
  if (IsMemoryPoolRecycle() && need_alloc_output_cnt <= kNeedRecycleOutput) {
    MS_LOG(INFO) << "Set Memory Pool Recycle, graph: " << kernel_graph->ToString()
                 << ", node: " << output_node->fullname_with_scope();
    output_device_addr->set_from_persistent_mem(true);
    output_device_addr->set_need_recycle(true);
  }
  return output_device_addr;
}

void AllocParameterMemory(const KernelGraphPtr &kernel_graph, DeviceContext *device_context,
                          std::set<KernelGraphPtr> *memo) {
  // Set Device Type to be same as Host Type, AssignStaticMemoryInput will ignore parameters without DeviceType
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(device_context);
  if (memo == nullptr) {
    MS_LOG(INFO) << "Start AllocParameterMemory, kernel graph: " << kernel_graph->ToString();
    std::set<KernelGraphPtr> memo_set;
    AllocParameterMemory(kernel_graph, device_context, &memo_set);
    MS_LOG(INFO) << "AllocParameterMemory finish.";
    return;
  } else if (memo->find(kernel_graph) != memo->end()) {
    return;
  }
  (void)memo->insert(kernel_graph);
  auto parameters = FilterAllParameters(kernel_graph);
  for (const auto &iter : parameters) {
    auto parameter = utils::cast<ParameterPtr>(iter.second);
    if (parameter == nullptr) {
      continue;
    }
    SetKernelInfo(parameter);
  }
  runtime::DeviceAddressUtils::CreateParameterDeviceAddress(device_context, kernel_graph);
  // call AssignStaticMemoryInput recursively
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto runtime_instance = device::KernelRuntimeManager::Instance().GetKernelRuntime(kAscendDevice, device_id);
  MS_EXCEPTION_IF_NULL(runtime_instance);
  runtime_instance->AssignStaticMemoryInput(*kernel_graph.get());
}

void AllocOutputMemory(const KernelGraphPtr &kernel_graph, GeDeviceResManager *res_manager) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start AllocOutputMemory, kernel graph: " << kernel_graph->ToString();

  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  auto ref_map = kernel_graph->GetRefMap();
  size_t need_alloc_output_cnt = 0;
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    if (output_node->isa<Parameter>() || output_node->isa<ValueNode>()) {
      continue;
    }
    if (ref_map.find(output_with_index) != ref_map.end()) {
      continue;
    }
    need_alloc_output_cnt++;
  }

  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    SetKernelInfo(output_node);

    // Parameter's memory is allocated earlier, and there is no need to reallocate memory if Parameter is output.
    if (AnfAlgo::OutputAddrExist(output_node, output_with_index.second, false) || output_node->isa<Parameter>()) {
      MS_LOG(INFO) << "The device_address of output node " << output_node->fullname_with_scope()
                   << " is already exist, skip.";
      continue;
    }

    auto output_device_addr =
      CreateOutputDeviceAddress(kernel_graph, output_with_index, need_alloc_output_cnt, res_manager);
    AnfAlgo::SetOutputAddr(output_device_addr, output_with_index.second, output_node.get());
    MS_LOG(INFO) << "Output node info: (name " << output_node->fullname_with_scope() << ", "
                 << output_node->DebugString() << " ), output size: " << output_device_addr->GetSize()
                 << ", device_address: " << output_device_addr;
    // When both the input and output of NopNode are used as outputs, different memory needs to be allocated for them.
  }
  MS_LOG(INFO) << "AllocOutputMemory finish.";
}

void EnableGraphInputZeroCopy(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // Zero copy is only enabled for PyNative and Subgraph sink.
  if ((!graph->has_flag(kFlagPyNativeRunInGraph) && !graph->has_flag(kFlagEnableZeroCopyInGraph)) ||
      !graph->is_graph_run_mode()) {
    return;
  }
  const auto &input_nodes = graph->input_nodes();
  for (const auto &input : input_nodes) {
    MS_EXCEPTION_IF_NULL(input);
    if (AnfAlgo::OutputAddrExist(input, 0)) {
      auto input_address = AnfAlgo::GetMutableOutputAddr(input, 0, false);
      MS_EXCEPTION_IF_NULL(input_address);
      input_address->set_is_ptr_persisted(false);
      input_address->ClearFlag(device::kDeviceAddressFlagNotUsed);
      MS_LOG(INFO) << "Enable zero copy for input " << input->DebugString();
    }
  }
}

void EnableGraphOutputZeroCopy(const KernelGraphPtr &graph) {
  MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy start";
  MS_EXCEPTION_IF_NULL(graph);
  if ((!graph->has_flag(kFlagEnableZeroCopyInGraph)) || !graph->is_graph_run_mode()) {
    MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy start return";
    return;
  }
  // Zero copy is only enabled for subgraph sink.
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output : outputs) {
    const auto &node_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    const auto &node = node_with_index.first;
    const auto &index = node_with_index.second;
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy check node:" << node->DebugString();
    if (node->isa<CNode>() && AnfAlgo::OutputAddrExist(node, index)) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(node, index, false);
      MS_EXCEPTION_IF_NULL(device_address);
      device_address->set_is_ptr_persisted(false);
      MS_LOG(DEBUG) << "Disable ptr persisted in output node:" << node->DebugString() << " index:" << index
                    << " address:" << device_address << " for graph:" << graph->ToString();
    }
  }
}

void AllocConstMemory(const transform::RunOptions &options, const KernelGraphPtr &graph, size_t memory_size,
                      GeDeviceResManager *res_manager) {
  if (memory_size == 0) {
    return;
  }
  MS_EXCEPTION_IF_NULL(res_manager);
  MS_LOG(INFO) << "Start AllocConstMemory, memory_size: " << memory_size;
  auto memory = res_manager->AllocateMemory(memory_size);
  if (memory == nullptr) {
    MS_LOG(EXCEPTION) << "Allocate memory failed, memory size:" << memory_size << ", graph: " << graph->ToString();
  }
  if (common::IsNeedProfileMemory()) {
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: ConstMemory, size: " << memory_size
                    << ", graph: " << graph->ToString() << ", device address addr: " << memory;
  }
  UpdateTracker("AllocConstMemory", "ConstMemory", graph->ToString(), memory_size, memory,
                device::tracker::MemType::kGeConst);
  auto graph_runner = transform::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  auto ret = graph_runner->SetConstMemory(options, memory, memory_size);
  if (ret != transform::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "SetConstMemory for graph " << options.name << " failed.";
  }
  MS_LOG(INFO) << "End AllocConstMemory";
}

void AllocFeatureMemory(const transform::RunOptions &options, size_t memory_size, GeDeviceResManager *res_manager) {
  if (memory_size == 0) {
    return;
  }
  MS_LOG(INFO) << "Start AllocFeatureMemory, memory_size: " << memory_size;
  MS_EXCEPTION_IF_NULL(res_manager);
  auto memory_manager = res_manager->mem_manager();
  MS_EXCEPTION_IF_NULL(memory_manager);
  memory_manager->ResetDynamicMemory();
  auto memory = memory_manager->MallocWorkSpaceMem(memory_size);
  if (memory == nullptr) {
    MS_LOG(EXCEPTION) << "AllocFeatureMemory error, memory not enough, memory size: " << memory_size;
  }
  auto graph_runner = transform::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  auto ret = graph_runner->UpdateFeatureMemory(options, memory, memory_size);
  if (ret != transform::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "UpdateFeatureMemory for graph " << options.name << " failed.";
  }
  memory_manager->ResetDynamicMemory();
  MS_LOG(INFO) << "End AllocFeatureMemory";
}
}  // namespace

void GEMemoryAllocator::AllocInputHostMemory(const KernelGraphPtr &kernel_graph, DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(device_context);
  const auto &inputs = kernel_graph->inputs();
  auto device_id = device_context->device_context_key().device_id_;
  for (const auto &input : inputs) {
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    builder->SetOutputsFormat({kOpFormat_DEFAULT});
    std::vector<TypeId> output_type = {common::AnfAlgo::GetOutputInferDataType(input, 0)};
    builder->SetOutputsDeviceType(output_type);
    AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), input.get());
  }

  for (const auto &input_node : inputs) {
    if (!input_node->isa<Parameter>()) {
      MS_LOG(DEBUG) << input_node->fullname_with_scope() << " is not parameter, continue";
      continue;
    }
    TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(input_node, 0);

    size_t tensor_size;
    if (kernel_graph->is_dynamic_shape()) {
      tensor_size = 0;
    } else {
      std::vector<size_t> shape = Convert2SizeT(common::AnfAlgo::GetOutputInferShape(input_node, 0));
      size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
      tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    }

    auto input_with_index = std::make_pair(input_node, 0);
    const auto kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      input_with_index, nullptr, tensor_size, kOpFormat_DEFAULT, output_type_id, {}, kAscendDevice, device_id);
    auto device_address_ptr = std::make_shared<GeHostAddress>(kernel_tensor);
    device_address_ptr->set_is_ptr_persisted(false);
    AnfAlgo::SetOutputAddr(device_address_ptr, 0, input_node.get());
  }
}

void GEMemoryAllocator::AllocOutputHostMemory(const KernelGraphPtr &kernel_graph, DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(device_context);
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  auto device_id = device_context->device_context_key().device_id_;
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    SetKernelInfo(output_node);

    // Parameter's memory is allocated earlier, and there is no need to reallocate memory if Parameter is output.
    if (output_node->isa<Parameter>()) {
      continue;
    }

    auto i = output_with_index.second;
    TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(output_node, i);
    const auto kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
      output_with_index, nullptr, 0, kOpFormat_DEFAULT, output_type_id, {}, kAscendDevice, device_id);
    auto output_device_addr = std::make_shared<GeHostAddress>(kernel_tensor);
    AnfAlgo::SetOutputAddr(output_device_addr, i, output_node.get());

    if (common::AnfAlgo::IsNopNode(output_node)) {
      auto [real_node, real_idx] = common::AnfAlgo::GetPrevNodeOutput(output_node, i, true);
      if (real_node != output_node || real_idx != i) {
        // set output addr size if the input node is output.
        const auto &inputs = kernel_graph->inputs();
        if (std::any_of(inputs.begin(), inputs.end(),
                        [&real_node](const AnfNodePtr &input_node) { return real_node == input_node; })) {
          auto real_node_addr = AnfAlgo::GetMutableOutputAddr(real_node, real_idx);
          output_device_addr->SetSize(real_node_addr->GetSize());
        }
        AnfAlgo::SetOutputAddr(output_device_addr, real_idx, real_node.get());
      }
    }
  }
}

void GEMemoryAllocator::AllocUnuseInput(const KernelGraphPtr &kernel_graph, const AnfNodePtr &input_node,
                                        DeviceAddress *output_addr, GeDeviceResManager *res_manager) {
  std::vector<size_t> shape = Convert2SizeT(common::AnfAlgo::GetOutputInferShape(input_node, 0));
  size_t type_size = GetTypeByte(TypeIdToType(common::AnfAlgo::GetOutputInferDataType(input_node, 0)));
  size_t memory_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>{});
  MS_EXCEPTION_IF_NULL(res_manager);
  MS_EXCEPTION_IF_NULL(output_addr);
  auto memory = res_manager->AllocateMemory(memory_size);
  output_addr->set_ptr(memory);
  output_addr->SetSize(memory_size);
  if (common::IsNeedProfileMemory()) {
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: UnusedInput, size:" << memory_size
                    << ", graph: " << kernel_graph->ToString() << ", node: " << input_node->fullname_with_scope()
                    << ", device address addr: " << memory;
  }
  UpdateTracker("UnusedInput", input_node->fullname_with_scope(), kernel_graph->ToString(), memory_size, memory,
                device::tracker::MemType::kOther);
}

void GEMemoryAllocator::ProcessGraphDeviceAddress(const KernelGraphPtr &kernel_graph, DeviceContext *device_context,
                                                  GeDeviceResManager *res_manager) {
  AllocParameterMemory(kernel_graph, device_context, nullptr);
  AllocOutputMemory(kernel_graph, res_manager);
  EnableGraphInputZeroCopy(kernel_graph);
  EnableGraphOutputZeroCopy(kernel_graph);
}

void GEMemoryAllocator::AllocGraphMemory(const transform::RunOptions &options, const KernelGraphPtr &graph,
                                         const GraphSummary &summary, size_t stream_id,
                                         GeDeviceResManager *res_manager) {
  AllocConstMemory(options, graph, summary.const_memory_size, res_manager);
  if (common::IsDisableRuntimeConfig(common::kRuntimeGeKernel)) {
    AllocFeatureMemory(options, summary.fixed_memory_size, res_manager);
  } else {
    GEMemoryManager::Instance().InitGEMemory(options, summary.workspace_memory_size, summary.fixed_memory_size,
                                             summary.const_memory_size, summary.is_refreshable, stream_id);
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
