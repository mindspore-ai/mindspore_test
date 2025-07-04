/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/cpu/hal/device/cpu_kernel_runtime.h"
#include <string>
#include <vector>
#include <memory>
#include <numeric>
#include <utility>
#include <algorithm>
#include <functional>
#include <exception>
#include "common/kernel.h"
#include "kernel/framework_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "plugin/res_manager/cpu/cpu_mem_manager/cpu_memory_manager.h"
#include "utils/ms_context.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/session_basic.h"
#include "include/backend/kernel_graph.h"
#include "frontend/operator/ops.h"
#include "plugin/device/cpu/hal/profiler/cpu_profiling.h"
#include "utils/shape_utils.h"
#include "utils/profile.h"
#include "utils/trace_base.h"
#include "debug/data_dump/cpu_e2e_dump.h"
#include "include/common/debug/env_config_parser.h"
#ifdef MEM_REUSE_DEBUG
#include "backend/common/mem_reuse/mem_reuse_checker.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#include "debug/rdr/mem_address_recorder.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#endif

namespace mindspore {
namespace device {
namespace cpu {
bool CPUKernelRuntime::Init() {
  if (initialized_) {
    return true;
  }
  mem_manager_ = std::make_shared<CPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  initialized_ = true;
  return true;
}

const size_t INIT_NODE_REF = 1;

void CPUKernelRuntime::AssignValueNodeAddress(const session::KernelGraph *kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &item_node : kernel_graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(item_node);
    if (item_node->isa<ValueNode>()) {
      auto value_node = item_node->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      auto node_value = value_node->value();
      MS_EXCEPTION_IF_NULL(node_value);
      if (!node_value->isa<tensor::Tensor>()) {
        continue;
      }
      auto tensor = node_value->cast<TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor);
      if (tensor->device_address() != nullptr) {
        AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), 0,
                               item_node);
        continue;
      }
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item_node, 0);
      if (output_type_id == kTypeUnknown) {
        output_type_id = common::AnfAlgo::GetOutputInferDataType(item_node, 0);
      }
      size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
      ShapeVector data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(), type_size, std::multiplies<size_t>());
      DeviceAddressPtr address = nullptr;
      address = CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, output_type_id);
      MS_EXCEPTION_IF_NULL(address);
      address->set_from_persistent_mem(tensor->is_parameter());
      if (tensor->data_type() == output_type_id) {
        address->SetDevicePtr(tensor->data_c());
      } else {
        address->SetDevicePtr(static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(tensor_size));
        if (!address->SyncHostToDevice(data_shape, LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                       tensor->data_c())) {
          MS_LOG(EXCEPTION) << "Value node sync host to device failed!";
        }
      }
      address->set_ref_count(INIT_NODE_REF);
      AnfAlgo::SetOutputAddr(address, 0, item_node);
    }
  }
}

void CPUKernelRuntime::AssignInputNodeAddress(const session::KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &item : kernel_graph->input_nodes()) {
    MS_EXCEPTION_IF_NULL(item);
    if (item->isa<Parameter>()) {
      auto output_num = AnfAlgo::GetOutputTensorNum(item);
      for (size_t index = 0; index < output_num; index++) {
        TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
        if (output_type_id == kTypeUnknown) {
          output_type_id = common::AnfAlgo::GetOutputInferDataType(item, index);
        }
        auto fmt_shape = AnfAlgo::GetOutputDeviceShape(item, index);
        size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
        size_t tensor_size = type_size * SizeOf(fmt_shape);
        auto format = AnfAlgo::GetOutputFormat(item, index);
        auto address = CreateDeviceAddress(nullptr, tensor_size, format, output_type_id);
        address->set_from_persistent_mem(true);
        AnfAlgo::SetOutputAddr(address, index, item);
      }
    }
  }
}

void CPUKernelRuntime::AssignKernelOutputAddress(const session::KernelGraph *kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto kernels = kernel_graph->execution_order();
  for (auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      AnfAlgo::SetOutputAddr(CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type), i, kernel);
    }
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      AnfAlgo::SetWorkspaceAddr(CreateDeviceAddress(nullptr, workspace_sizes[i], kOpFormat_DEFAULT, kNumberTypeFloat32),
                                i, kernel);
    }
  }
}

DeviceAddressPtr CPUKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                       TypeId type_id) const {
  return std::make_shared<CPUDeviceAddress>(device_ptr, device_size, format, type_id);
}

DeviceAddressPtr CPUKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                       TypeId type_id, const KernelWithIndex &node_index) const {
  return std::make_shared<CPUDeviceAddress>(device_ptr, device_size, format, type_id, node_index);
}

tensor::TensorPtr CPUKernelRuntime::CreateTensorForOutput(session::KernelGraph *kernel_graph, const CNodePtr &node,
                                                          size_t index, std::set<DeviceAddressPtr> *bound_addresses) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(bound_addresses);
  size_t output_size = AnfAlgo::GetOutputTensorNum(node);
  if (index >= output_size) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "For node " << node->DebugString() << ", index " << index
                                      << " exceed output size " << output_size;
  }
  auto address = AnfAlgo::GetMutableOutputAddr(node, index);
  MS_EXCEPTION_IF_NULL(address);
  TypeId infer_type_id = common::AnfAlgo::GetOutputInferDataType(node, index);
  TypeId device_type_id = AnfAlgo::GetOutputDeviceDataType(node, index);
  auto shape = common::AnfAlgo::GetOutputInferShape(node, index);
  ShapeVector temp_shape;
  tensor::TensorPtr tensor;
  bool is_internal_output = kernel_graph->IsInternalOutput(node, index);
  (void)temp_shape.insert(temp_shape.end(), shape.begin(), shape.end());
  if (is_internal_output) {
    tensor = kernel_graph->GetInternalOutputTensor(node, index);
    if (tensor == nullptr) {
      size_t type_size = GetTypeByte(TypeIdToType(device_type_id));
      if (type_size == 0) {
        MS_LOG(EXCEPTION) << "Invalid type_size " << type_size;
      }
      size_t tensor_size = std::accumulate(temp_shape.begin(), temp_shape.end(), type_size, std::multiplies<size_t>());
      if (tensor_size < address->GetSize()) {
        temp_shape.clear();
        (void)temp_shape.emplace_back(address->GetSize() / type_size);
      }
      tensor = std::make_shared<tensor::Tensor>(infer_type_id, temp_shape);
    }
    kernel_graph->AddInternalOutputTensor(node, index, tensor);
  } else {
    tensor = std::make_shared<tensor::Tensor>(infer_type_id, temp_shape);
  }
  tensor->set_device_address(address);
  tensor->set_sync_status(kNeedSyncDeviceToHostImmediately);
  if (bound_addresses->find(address) == bound_addresses->end()) {
    if (infer_type_id != device_type_id) {
      size_t type_size = GetTypeByte(TypeIdToType(device_type_id));
      ShapeVector data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(), type_size, std::multiplies<size_t>());
      address->SetDevicePtr(static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(tensor_size));
      address->SetSize(tensor_size);
      address->set_type_id(device_type_id);
    } else {
      tensor->set_sync_status(kNoNeedSync);
    }
    (void)bound_addresses->insert(address);
  }
  tensor->SetIsGraphOutput();
  return tensor;
}

BaseRef CPUKernelRuntime::GetOrCreateTensorForOutput(
  session::KernelGraph *kernel_graph, const session::KernelWithIndex &kernel_with_index,
  std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node,
  std::map<AnfNodePtr, tensor::TensorPtr> *input_param_tensor_map, std::set<DeviceAddressPtr> *bound_addresses) {
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  MS_EXCEPTION_IF_NULL(input_param_tensor_map);
  auto &input_node = kernel_with_index.first;
  auto index = kernel_with_index.second;
  MS_EXCEPTION_IF_NULL(input_node);

  if (input_node->isa<CNode>()) {
    auto node = input_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    if (common::AnfAlgo::GetCNodeName(input_node) == prim::kPrimMakeTuple->name()) {
      VectorRef ret;
      for (size_t i = 1; i < node->size(); i++) {
        auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(node->input(i), 0);
        auto out = GetOrCreateTensorForOutput(kernel_graph, item_with_index, tensor_to_node, input_param_tensor_map,
                                              bound_addresses);
        ret.push_back(out);
      }
      return ret;
    }
    auto tensor = CreateTensorForOutput(kernel_graph, node, index, bound_addresses);
    (*tensor_to_node)[tensor] = kernel_with_index;
    return tensor;
  } else if (input_node->isa<Parameter>()) {
    auto iter = input_param_tensor_map->find(input_node);
    if (iter != input_param_tensor_map->end()) {
      return iter->second;
    }
  } else if (input_node->isa<ValueNode>()) {
    auto value_node = input_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->value();
  }
  return BaseRef();
}

void CPUKernelRuntime::CreateOutputTensors(session::KernelGraph *kernel_graph,
                                           const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs,
                                           std::map<tensor::TensorPtr, session::KernelWithIndex> *tensor_to_node) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(tensor_to_node);
  auto &input_nodes = kernel_graph->input_nodes();
  if (input_nodes.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Input size " << inputs.size() << " is not equal to input node size " << input_nodes.size();
  }

  std::map<AnfNodePtr, tensor::TensorPtr> input_param_tensor_map;
  size_t input_idx = 0;
  for (auto &item : input_nodes) {
    MS_EXCEPTION_IF_NULL(item);
    input_param_tensor_map[item] = inputs[input_idx];
    input_idx++;
  }

  std::set<DeviceAddressPtr> bound_addresses;
  auto output_nodes = kernel_graph->outputs();
  for (const auto &item : output_nodes) {
    auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(item, 0, false);
    auto out = GetOrCreateTensorForOutput(kernel_graph, item_with_index, tensor_to_node, &input_param_tensor_map,
                                          &bound_addresses);
    outputs->push_back(std::move(out));
  }
}

void CPUKernelRuntime::BindInputTensorAddressPtr(const session::KernelGraph &kernel_graph,
                                                 const std::vector<tensor::TensorPtr> &inputs) {
  auto &input_nodes = kernel_graph.input_nodes();
  if (input_nodes.size() != inputs.size()) {
    MS_LOG(EXCEPTION) << "Input size" << inputs.size() << " is not equal to input node size " << input_nodes.size();
  }
  for (size_t input_idx = 0; input_idx < input_nodes.size(); ++input_idx) {
    auto &item = input_nodes[input_idx];
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>() || HasAbstractMonad(item)) {
      continue;
    }
    auto address = AnfAlgo::GetMutableOutputAddr(item, 0);
    auto tensor = inputs[input_idx];
    MS_EXCEPTION_IF_NULL(address);
    MS_EXCEPTION_IF_NULL(tensor);
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    MS_LOG(EXCEPTION) << "CPUKernelRuntime::BindInputTensorAddressPtr is deprecated.";
    if (GetTypeByte(TypeIdToType(tensor->data_type())) == GetTypeByte(TypeIdToType(address->type_id()))) {
      address->SetDevicePtr(tensor->data_c());
    } else {
      ShapeVector data_shape = tensor->shape();
      size_t tensor_size = std::accumulate(data_shape.begin(), data_shape.end(),
                                           GetTypeByte(TypeIdToType(address->type_id())), std::multiplies<size_t>());
      if (address->GetDevicePtr() == nullptr || address->GetSize() != tensor_size) {
        address->SetDevicePtr(static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(tensor_size));
        address->SetSize(tensor_size);
      }
      if (!address->SyncHostToDevice(data_shape, LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                     tensor->data_c())) {
        MS_LOG(EXCEPTION) << "Parameter node sync host to device failed!";
      }
    }
    auto input_param = item->cast<ParameterPtr>();
    if (input_param != nullptr && input_param->IsUsedByRealKernelInGraph(kernel_graph.graph_id())) {
      auto tensor_shape = tensor->shape();
      common::AnfAlgo::SetOutputInferTypeAndShape({common::AnfAlgo::GetOutputInferDataType(item, 0)}, {tensor_shape},
                                                  item.get());
    }
    address->set_ref_count(INIT_NODE_REF);
    if (common::AnfAlgo::IsParameterWeight(input_param)) {
      tensor->set_device_address(address);
    }
  }
}

void CPUKernelRuntime::BindOutputTensorAddressPtr(const VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  for (auto item : *outputs) {
    if (utils::isa<VectorRefPtr>(item)) {
      auto vector_ref = utils::cast<VectorRef>(item);
      BindOutputTensorAddressPtr(&vector_ref);
    } else if (utils::isa<tensor::TensorPtr>(item)) {
      auto tensor = utils::cast<tensor::TensorPtr>(item);
      MS_EXCEPTION_IF_NULL(tensor);
      auto address = tensor->device_address();
      if (address == nullptr) {
        continue;
      }
      auto address_ptr = std::dynamic_pointer_cast<device::DeviceAddress>(address);
      if (address_ptr->type_id() == tensor->data_type_c() && tensor->sync_status() == kNoNeedSync) {
        address_ptr->SetDevicePtr(tensor->data_c());
      }
      address_ptr->set_ref_count(INIT_NODE_REF);
    }
  }
}

void CPUKernelRuntime::BindInputOutput(session::KernelGraph *kernel_graph, const std::vector<tensor::TensorPtr> &inputs,
                                       VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  BindInputTensorAddressPtr(*kernel_graph, inputs);
  BindOutputTensorAddressPtr(outputs);
}

void CPUKernelRuntime::AddRuntimeAddress(KernelTensor *kernel_tensor, std::vector<kernel::KernelTensor *> *input_list) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  MS_EXCEPTION_IF_NULL(input_list);
  if (kernel_tensor->device_ptr() == nullptr) {
    auto addr = static_cast<CPUMemoryManager *>(mem_manager_.get())->StaticMemMalloc(kernel_tensor->GetSize());
    MS_EXCEPTION_IF_NULL(addr);
    kernel_tensor->set_device_ptr(addr);
  }
  input_list->push_back(kernel_tensor);
}

void CPUKernelRuntime::IncreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs) {
  auto cpu_mem_manager = static_cast<CPUMemoryManager *>(mem_manager_.get());
  MS_EXCEPTION_IF_NULL(cpu_mem_manager);
  if (cpu_mem_manager->GetDynamicMalloc()) {
    if (summary_outputs.empty()) {
      return;
    }
    for (auto &output_item : summary_outputs) {
      auto node = output_item.second.first;
      size_t index = IntToSize(output_item.second.second);
      auto address = AnfAlgo::GetMutableOutputAddr(node, index);
      MS_EXCEPTION_IF_NULL(address);
      address->set_ref_count(address->ref_count() + 1);
    }
  }
}

void CPUKernelRuntime::DecreaseSummaryRefCount(const session::NamedSummaryOutputs &summary_outputs) {
  auto cpu_mem_manager = static_cast<CPUMemoryManager *>(mem_manager_.get());
  MS_EXCEPTION_IF_NULL(cpu_mem_manager);
  if (cpu_mem_manager->GetDynamicMalloc()) {
    if (summary_outputs.empty()) {
      return;
    }
    for (auto &output_item : summary_outputs) {
      auto node = output_item.second.first;
      size_t index = IntToSize(output_item.second.second);
      auto address = AnfAlgo::GetMutableOutputAddr(node, index);
      MS_EXCEPTION_IF_NULL(address);
      address->DecreaseRefCount();
      if (address->ref_count() == 0 && address->GetDevicePtr() != nullptr) {
        cpu_mem_manager->MemFree(address->GetDevicePtr());
        address->SetDevicePtr(nullptr);
      }
    }
  }
}

void CPUKernelRuntime::GetRuntimeAddressFromNode(const AnfNodePtr &node, std::vector<kernel::KernelTensor *> *inputs,
                                                 std::vector<kernel::KernelTensor *> *outputs,
                                                 std::vector<kernel::KernelTensor *> *workspaces) {
  MS_EXCEPTION_IF_NULL(inputs);
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(workspaces);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_num; ++i) {
    auto kernel_tensor = AnfAlgo::GetPrevNodeOutputKernelTensor(node, i, true).get();
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    AddRuntimeAddress(kernel_tensor, inputs);
  }
  size_t output_num = AnfAlgo::GetOutputTensorNum(node);
  for (size_t i = 0; i < output_num; ++i) {
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, i, true).get();
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    AddRuntimeAddress(kernel_tensor, outputs);
  }
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
    auto kernel_tensor = AnfAlgo::GetWorkspaceKernelTensor(node, i).get();
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    AddRuntimeAddress(kernel_tensor, workspaces);
  }
}

void CPUKernelRuntime::RunKernel(const CNodePtr &kernel, bool iter_dump_flag, uint32_t graph_id) {
  double start_time = 0;
  if (IS_OUTPUT_ON(mindspore::kInfo)) {
    start_time = GetTime();
  }
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  // akg kernel do not support dynamic shape by now
  kernel::NativeCpuKernelMod *cpu_kernel = nullptr;
  if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) != KernelType::AKG_KERNEL) {
    cpu_kernel = dynamic_cast<kernel::NativeCpuKernelMod *>(kernel_mod);
    MS_EXCEPTION_IF_NULL(cpu_kernel);
  }
  if (common::AnfAlgo::IsDynamicShape(kernel)) {
    AnfAlgo::InferShape(kernel);
    auto inputs = AnfAlgo::GetOrCreateAllInputKernelTensors(kernel);
    auto outputs = AnfAlgo::GetOrCreateAllOutputKernelTensors(kernel);
    if (cpu_kernel != nullptr && cpu_kernel->Resize(inputs, outputs) == static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
      MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Node " << kernel->fullname_with_scope() << " Resize failed!";
    }
  }
  std::vector<kernel::KernelTensor *> kernel_inputs;
  std::vector<kernel::KernelTensor *> kernel_workspaces;
  std::vector<kernel::KernelTensor *> kernel_outputs;
  GetRuntimeAddressFromNode(kernel, &kernel_inputs, &kernel_outputs, &kernel_workspaces);
  bool ret = true;
  auto profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
  uint32_t pid = IntToUint(getpid());
  profiler_inst->OpDataProducerBegin(kernel->fullname_with_scope(), pid);
#ifdef ENABLE_DUMP_IR
  kernel::KernelLaunchInfo launch_info = {kernel_inputs, kernel_outputs, kernel_workspaces};
  std::string op_name = kernel->fullname_with_scope();
  kernel::KernelLaunchAddr mem_info;
  ConvertLaunchInfoToAddr(launch_info, &mem_info);
  (void)mindspore::RDR::UpdateMemAddress(SubModuleId::SM_KERNEL, "mem_address_list", op_name, mem_info);
#endif
  try {
    ret = kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, nullptr);
  } catch (std::exception &e) {
    MS_LOG(EXCEPTION) << e.what() << trace::DumpSourceLines(kernel);
  }
  if (iter_dump_flag) {
    CPUE2eDump::DumpCNodeData(kernel, graph_id);
  }
  profiler_inst->OpDataProducerEnd();
  if (!ret) {
#ifdef ENABLE_DUMP_IR
    mindspore::RDR::TriggerAll();
#endif
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "Launch kernel failed." << trace::DumpSourceLines(kernel);
  }
  auto cpu_mem_manager = static_cast<CPUMemoryManager *>(mem_manager_.get());
  MS_EXCEPTION_IF_NULL(cpu_mem_manager);
  if (cpu_mem_manager->GetDynamicMalloc()) {
    MS_EXCEPTION_IF_NULL(kernel);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    for (size_t i = 0; i < input_num; ++i) {
      auto address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(address);
      address->DecreaseRefCount();
      if (address->ref_count() == 0 && address->GetDevicePtr() != nullptr) {
        cpu_mem_manager->MemFree(address->GetDevicePtr());
        address->SetDevicePtr(nullptr);
      }
    }
    for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
      auto address = AnfAlgo::GetWorkspaceAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(address);
      address->DecreaseRefCount();
      if (address->ref_count() == 0 && address->GetDevicePtr() != nullptr) {
        cpu_mem_manager->MemFree(address->GetDevicePtr());
        address->SetDevicePtr(nullptr);
      }
    }
  }
  if (IS_OUTPUT_ON(mindspore::kInfo)) {
    double cost_time = GetTime() - start_time;
    MS_LOG(INFO) << "cpu kernel: " << kernel->fullname_with_scope() << "  costs " << cost_time * 1e6 << " us";
  }
}

bool CPUKernelRuntime::Run(const session::KernelGraph &kernel_graph, bool) {
  auto cpu_mem_manager = static_cast<CPUMemoryManager *>(mem_manager_.get());
  MS_EXCEPTION_IF_NULL(cpu_mem_manager);
  if (cpu_mem_manager->GetDynamicMalloc()) {
    auto kernels = kernel_graph.execution_order();
    for (const auto &kernel : kernels) {
      size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
      for (size_t i = 0; i < input_num; ++i) {
        auto address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
        MS_EXCEPTION_IF_NULL(address);
        address->set_ref_count(address->ref_count() + 1);
      }
      auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
        auto address = AnfAlgo::GetWorkspaceAddr(kernel, i);
        MS_EXCEPTION_IF_NULL(address);
        address->set_ref_count(address->ref_count() + 1);
      }
    }
  }

  auto kernels = kernel_graph.execution_order();

  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool iter_dump_flag = dump_json_parser.GetIterDumpFlag();
  uint32_t graph_id = kernel_graph.graph_id();
#ifdef ENABLE_DUMP_IR
  std::string name = "mem_address_list";
  (void)mindspore::RDR::RecordMemAddressInfo(SubModuleId::SM_KERNEL, name);
#endif
  for (const auto &kernel : kernels) {
    RunKernel(kernel, iter_dump_flag, graph_id);
  }
  if (iter_dump_flag) {
    CPUE2eDump::DumpParameters(&kernel_graph, graph_id);
    CPUE2eDump::DumpConstants(&kernel_graph, graph_id);
  }
  if (graph_id == 0) {
    dump_json_parser.UpdateDumpIter();
  }
  return true;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
