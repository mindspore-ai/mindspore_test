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

#include "backend/graph_compiler/ge_backend/ge_backend.h"

#include <algorithm>
#include <set>
#include "backend/common/optimizer/common_backend_optimization.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "ir/manager.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/config_manager.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/common/profiler.h"
#include "include/backend/device_address.h"
#include "utils/file_utils.h"
#include "debug/data_dump/data_dumper.h"
#ifndef ENABLE_SECURITY
#include "debug/hooker/hook_debugger.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/debug/data_dump/e2e_dump.h"
#endif

namespace mindspore::compile {
namespace {
constexpr size_t kNormalTensorNum = 1;
constexpr size_t kMapTensorNum = 3;
constexpr size_t kMapTensorKeyIndex = 0;
constexpr size_t kMapTensorValueIndex = 1;
constexpr size_t kMapTensorStatusIndex = 2;
constexpr size_t kGraphInfoSavePrefixLen = 5;
}  // namespace
mindspore::HashSet<const tensor::Tensor *> GEBackend::weights_need_reprepare_ = {};

std::string GEBackend::CompileGraph(const FuncGraphPtr &func_graph, const device::DeviceContext *device_context,
                                    const session::JitSetting &jit_setting) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(INFO) << "Status record: start compile graph.";
  // Generate kernel graph.
  std::vector<KernelGraphPtr> all_graphs;

  auto device_target = device_context->GetDeviceType();
  auto kg_mgr = std::make_shared<session::KernelGraphMgr>();
  KernelGraphPtr root_graph = kg_mgr->ConstructKernelGraph(func_graph, &all_graphs, device_target, jit_setting);
  MS_EXCEPTION_IF_NULL(root_graph);
  for (const auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    MS_LOG(INFO) << "Set root graph for graph: " << graph->graph_id() << " to: " << root_graph->graph_id() << ".";
    graph->set_root_graph_id(root_graph->graph_id());
    graph->set_run_mode(device::RunMode::kGraphMode);
    graph->set_is_loop_count_sink(true);
    graph->set_attrs(func_graph->attrs());
    opt::OptimizationWithoutBackend(graph);
  }
  device_context->graph_executor_->OptimizeBeforeCompileGraph(root_graph);

  auto manager = MakeManager();
  MS_EXCEPTION_IF_NULL(manager);
  for (const auto &graph : all_graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    graph->set_flag(kFlagEnableZeroCopyInGraph, true);
    manager->AddFuncGraph(graph);
    graph->set_manager(manager);
    graph->SetInputNodes();
  }
  root_graph->SetInputNodes();

  if (!device_context->graph_executor_->CompileGraph(root_graph, {})) {
    MS_LOG(EXCEPTION) << "Compile graph failed: " << root_graph->graph_id();
  }
  root_graph->CacheGraphOutputToFrontNodeWithIndex({root_graph->output()}, {func_graph->output()});

  device_context->graph_executor_->InitGraphInfo(root_graph);

  auto graph_info = GenerateGraphInfo(root_graph->graph_id());
  graph_map_[graph_info] = root_graph;
  graph_run_iter_[root_graph] = 0;
  MS_LOG(INFO) << "Status record: end compile graph.";
  return graph_info;
}

void GEBackend::SetTensorUpdateCallback(const tensor::TensorPtr &update_tensor) {
  if (update_tensor != nullptr && update_tensor->update_value_callback() == nullptr && update_tensor->is_parameter()) {
    static auto callback = [this](const tensor::Tensor *tensor) { weights_need_reprepare_.insert(tensor); };
    update_tensor->set_update_value_callback(callback);
  }
}

void GEBackend::ConstructInputsUnRefMode(const KernelGraphPtr &func_graph, const VectorRef &args,
                                         std::vector<tensor::TensorPtr> *inputs_tensor) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto inputs = func_graph->inputs();
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == args.size(), "The args size is not equal to graph inputs size.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    std::vector<tensor::TensorPtr> flatten_tensors;
    auto params = common::AnfAlgo::GetAllOutput(inputs[i]);
    for (size_t j = 0; j < params.size(); ++j) {
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(params[j], 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      // skip const input
      if (TEST_FLAG(device_tensor->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
        MS_LOG(INFO) << "The input[" << i << "] is convert to const op, skip.";
        continue;
      }
      // for unrefmode, param init in a dependent graph, and no need to put it into inputs
      if (common::AnfAlgo::IsParameterWeight(params[j]->cast<ParameterPtr>())) {
        continue;
      }
      // get host tensor
      if (flatten_tensors.empty()) {
        const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(inputs[i], *func_graph);
        AnfAlgo::FlattenInputArg(args[i], front_node, &flatten_tensors);
        if (flatten_tensors.size() != params.size()) {
          MS_LOG(EXCEPTION) << "The args[" << i << "] tensor number is not equal to inputs[" << i << "] number.";
        }
      }
      // unrefmode, just the host input tensor exclude weight
      inputs_tensor->emplace_back(flatten_tensors[j]);
    }
  }
}

void GEBackend::UpdateInputsShapeAndSize(const ParameterPtr &input_node,
                                         const mindspore::device::DeviceAddressPtr &device_tensor,
                                         const tensor::TensorPtr &input_tensor,
                                         const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(device_tensor);
  MS_EXCEPTION_IF_NULL(input_tensor);
  // update shape and size, for dynamic shape
  if (!input_node->has_dynamic_shape() && !IsDynamic(device_tensor->host_shape())) {
    return;
  }

  // update shape
  MS_LOG(DEBUG) << "Update dynamic shape for parameter:" << input_node->DebugString();
  const auto &output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_node, 0);
  MS_EXCEPTION_IF_NULL(output_kernel_tensor);
  if (input_tensor->base_shape_ptr() == nullptr || (!input_tensor->base_shape_ptr()->isa<abstract::SequenceShape>())) {
    output_kernel_tensor->SetShape(input_tensor->ToAbstract()->GetShape());
    device_context->graph_executor_->AllocInputMemory(device_tensor);
    return;
  }
  output_kernel_tensor->SetShape(input_tensor->base_shape_ptr());

  // Update size.
  auto device_format = device_tensor->format();
  static const std::set<std::string> kNormalFormat = {
    kOpFormat_DEFAULT, kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN,
  };
  if (kNormalFormat.find(device_format) != kNormalFormat.end()) {
    auto tensor_data_size = input_tensor->data().nbytes();
    MS_LOG(DEBUG) << "Set device address:" << device_tensor << " size from:" << device_tensor->GetSize()
                  << " to:" << tensor_data_size;
    device_tensor->SetSize(tensor_data_size);
  } else {
    MS_LOG(DEBUG) << "Update data node device address size";
    // Size of 5D format device_tensor is larger than tensor_data_size.
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(input_node, 0);
    if (output_type_id == kTypeUnknown) {
      output_type_id = common::AnfAlgo::GetOutputInferDataType(input_node, 0);
    }
    auto device_shape =
      trans::TransShapeToDevice(input_tensor->shape(), device_tensor->format(), input_node, 0, output_type_id);
    size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
    auto device_address_size = type_size * SizeOf(device_shape);
    MS_LOG(INFO) << "Size of device_address is updated from " << device_tensor->GetSize() << " to "
                 << device_address_size;
    device_tensor->SetSize(device_address_size);
  }

  device_context->graph_executor_->AllocInputMemory(device_tensor);
}

void GEBackend::ConstructInputsRefMode(const KernelGraphPtr &func_graph, const VectorRef &args,
                                       std::vector<tensor::TensorPtr> *inputs_tensor,
                                       const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto inputs = func_graph->inputs();
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == args.size(), "The args size is not equal to graph inputs size.");
  for (size_t i = 0; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    std::vector<tensor::TensorPtr> flatten_tensors;
    auto params = common::AnfAlgo::GetAllOutput(inputs[i]);
    for (size_t j = 0; j < params.size(); ++j) {
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(params[j], 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      // skip const input
      if (TEST_FLAG(device_tensor->flag(), device::kDeviceAddressFlagIgnoreDevicePtr)) {
        MS_LOG(INFO) << "The input[" << i << "] is convert to const op, skip.";
        continue;
      }
      // for refmode, weight copy to device just once
      auto parameter = params[j]->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(parameter);
      if (is_weight_init_[parameter] && weights_need_reprepare_.empty()) {
        continue;
      }
      // get host tensor
      if (flatten_tensors.empty()) {
        const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(inputs[i], *func_graph);
        AnfAlgo::FlattenInputArg(args[i], front_node, &flatten_tensors);
        MS_EXCEPTION_IF_CHECK_FAIL(flatten_tensors.size() == params.size(),
                                   "The flatten_tensors size is not equal to params size.");
      }

      bool is_need_sync = true;
      auto host_tensor_address =
        std::dynamic_pointer_cast<mindspore::device::DeviceAddress>(flatten_tensors[j]->device_address());

      UpdateInputsShapeAndSize(parameter, device_tensor, flatten_tensors[j], device_context);

      // in different backend object, but has init, skip
      if (common::AnfAlgo::IsParameterWeight(parameter)) {
        is_weight_init_[parameter] = true;
        // for weight value update in python
        SetTensorUpdateCallback(flatten_tensors[j]);

        device_tensor->set_is_ptr_persisted(true);
        if (host_tensor_address == device_tensor) {
          continue;
        }

        if (host_tensor_address == nullptr) {
          // host is nullptr -> set & copy_to_device
          host_tensor_address = device_tensor;
          flatten_tensors[j]->set_device_address(host_tensor_address);
          is_need_sync = true;
        } else if (host_tensor_address->GetDeviceType() != device_tensor->GetDeviceType()) {
          // device_type not same -> sync_to_host & copy_to_device
          flatten_tensors[j]->data_sync();
          host_tensor_address = device_tensor;
          flatten_tensors[j]->set_device_address(device_tensor);
          is_need_sync = true;
        } else {
          // other not same condition -> device_copy
          if (!Copy(device_tensor.get(), host_tensor_address.get())) {
            MS_LOG(EXCEPTION) << "Sync data error.";
          }
          host_tensor_address = device_tensor;
          flatten_tensors[j]->set_device_address(device_tensor);
          is_need_sync = false;
        }
      } else {
        if (host_tensor_address == device_tensor) {
          continue;
        }

        if (host_tensor_address != nullptr) {
          if (host_tensor_address->GetPtr() == device_tensor->GetPtr()) {
            continue;
          } else if (host_tensor_address->GetPtr() == nullptr) {
            flatten_tensors[j]->set_device_address(nullptr);
            host_tensor_address = nullptr;
            is_need_sync = true;
          } else if (host_tensor_address->GetDeviceType() != device_tensor->GetDeviceType()) {
            // device type not same: tensor sync to host & copy to device_tensor
            flatten_tensors[j]->data_sync();
            is_need_sync = true;
          } else {
            runtime::DeviceAddressUtils::ConvertContiguousTensorSync(flatten_tensors[j]);
            host_tensor_address =
              std::dynamic_pointer_cast<mindspore::device::DeviceAddress>(flatten_tensors[j]->device_address());
            // other not same: device copy
            if (!Copy(device_tensor.get(), host_tensor_address.get())) {
              MS_LOG(EXCEPTION) << "Sync data error.";
            }
            is_need_sync = false;
          }
        } else {
          is_need_sync = true;
        }
      }
      if (is_need_sync) {
        SyncTensorData(flatten_tensors[j], device_tensor, params[j]);
      }
    }
  }
  // clear every step
  weights_need_reprepare_.clear();
}

void GEBackend::ConstructInputs(const KernelGraphPtr &func_graph, const VectorRef &args,
                                std::vector<tensor::TensorPtr> *inputs_tensor,
                                const device::DeviceContext *device_context) {
  if (IsEnableRefMode()) {
    ConstructInputsRefMode(func_graph, args, inputs_tensor, device_context);
  } else {
    ConstructInputsUnRefMode(func_graph, args, inputs_tensor);
  }
}

bool GEBackend::Copy(const mindspore::device::DeviceAddress *dst_device_tensor,
                     const mindspore::device::DeviceAddress *src_device_tensor) {
  MS_EXCEPTION_IF_NULL(dst_device_tensor);
  MS_EXCEPTION_IF_NULL(src_device_tensor);
  if (src_device_tensor->GetSize() != dst_device_tensor->GetSize()) {
    MS_LOG(INFO) << "Copy size is not equal, input size:" << src_device_tensor->GetSize()
                 << ", output size:" << dst_device_tensor->GetSize();
  }

  // Exist the size alignment in some device, so get the min device size.
  size_t copy_size = std::min(src_device_tensor->GetSize(), dst_device_tensor->GetSize());

  if (dst_device_tensor->GetDeviceType() == src_device_tensor->GetDeviceType()) {
    return dst_device_tensor->SyncDeviceToDevice(src_device_tensor);
  } else if (src_device_tensor->GetDeviceType() == device::DeviceType::kCPU) {
    // CPU device tensor copy to other device tensor.
    return dst_device_tensor->SyncHostToDevice(copy_size, src_device_tensor->GetPtr());
  } else if (dst_device_tensor->GetDeviceType() == device::DeviceType::kCPU) {
    // Other device tensor copy to CPU device tensor.
    return src_device_tensor->SyncDeviceToHost(copy_size, dst_device_tensor->GetMutablePtr());
  } else {
    MS_LOG(ERROR) << "Invalid device type, src device type: " << src_device_tensor->GetDeviceType()
                  << ", dst device type: " << dst_device_tensor->GetDeviceType();
    return false;
  }
}

void GEBackend::SyncTensorData(const tensor::TensorPtr &host_tensor,
                               const std::shared_ptr<device::DeviceAddress> &device_tensor, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(host_tensor);
  MS_EXCEPTION_IF_NULL(device_tensor);
  MS_EXCEPTION_IF_NULL(node);
  // memory has been allocate early in AllocGEInputOutputMemory
  MS_EXCEPTION_IF_NULL(device_tensor->GetPtr());
  // sync host tensor to device
  auto get_tensor_by_index = [&host_tensor](size_t index) {
    if (!host_tensor->isa<tensor::MapTensor>()) {
      return host_tensor;
    }
    const auto &map_tensor = host_tensor->cast<tensor::MapTensorPtr>();
    MS_EXCEPTION_IF_NULL(map_tensor);
    switch (index) {
      case kMapTensorKeyIndex:
        return map_tensor->key_tensor();
      case kMapTensorValueIndex:
        return map_tensor->value_tensor();
      case kMapTensorStatusIndex:
        return map_tensor->status_tensor();
      default:
        MS_LOG(EXCEPTION) << "Invalid index:" << index << " for map tensor:" << host_tensor->ToString();
    }
  };
  ShapeVector host_shape = {};
  // GetRuntimePaddingShape doesn't support the value tuple node.
  if (!node->isa<ValueNode>()) {
    host_shape = trans::GetRuntimePaddingShape(node, 0);
  }
  auto get_tensor_num = (host_tensor->isa<tensor::MapTensor>() ? kMapTensorNum : kNormalTensorNum);
  for (size_t i = 0; i < get_tensor_num; ++i) {
    const auto &real_host_tensor = get_tensor_by_index(i);
    MS_EXCEPTION_IF_NULL(real_host_tensor);
    // Copy data from host tensor to device.
    auto host_tensor_size = LongToSize(real_host_tensor->data().nbytes());
    auto host_tensor_type = real_host_tensor->data_type();
    if (node->isa<ValueNode>()) {
      host_shape = real_host_tensor->shape();
    }
    if (!device_tensor->SyncHostToDevice(host_shape, host_tensor_size, host_tensor_type,
                                         real_host_tensor->device_info().host_format_, real_host_tensor->data_ptr())) {
      MS_LOG(EXCEPTION) << "SyncHostToDevice failed, node name: " + node->fullname_with_scope() +
                             ", host tensor size: " + std::to_string(host_tensor_size) +
                             ", host tensor type: " + std::to_string(static_cast<int>(host_tensor_type)) +
                             ", device tensor size: " + std::to_string(device_tensor->GetSize());
    }
  }
}

void GEBackend::ConstructOutputs(const KernelGraphPtr &func_graph, std::vector<tensor::TensorPtr> *outputs,
                                 const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(outputs);
  auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(func_graph->output());
  // map of output_node ptr and corresponding tensor, for same output condition
  // 1. same device_address; 2. io_index, same pointer_ref_count
  mindspore::HashMap<PointerRefCountPtr, device::DeviceAddressPtr> output_node_tensor_map;
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const auto &[output_node, idx] = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_outputs[i]);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    auto output_addr = AnfAlgo::GetMutableOutputAddr(output_node, idx, false);
    const auto &output_kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, idx);
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    MS_EXCEPTION_IF_NULL(output_addr);

    // when output_addr exist, need gen fake output
    if (common::AnfAlgo::IsNoOuputNode(output_node) && output_addr == nullptr) {
      continue;
    }

    auto out_tensor =
      std::make_shared<tensor::Tensor>(output_addr->type_id(), output_addr->kernel_tensor()->GetShapeVector());

    auto kernel_tensor = std::make_shared<kernel::KernelTensor>(
      nullptr, output_addr->GetSize(), kernel::GetFormatFromStrToEnum(output_addr->format()), output_addr->type_id(),
      output_addr->host_shape(), kAscendDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID));
    kernel_tensor->SetType(output_kernel_tensor->GetType());
    kernel_tensor->SetShape(output_kernel_tensor->GetShape());
    kernel_tensor->set_stream_id(output_addr->stream_id());
    // SetShape will calculate a default size by host shape, need to set real device size for special format.
    kernel_tensor->set_size(output_addr->GetSize());
    auto tensor_device_address =
      device_context->graph_executor_->CreateDeviceAddress(kernel_tensor, output_addr->is_ptr_persisted());
    tensor_device_address->set_new_ref_count(SIZE_MAX);
    MS_EXCEPTION_IF_NULL(tensor_device_address);

    if (output_addr->is_ptr_persisted()) {
      // device_tensor persisted or format not same -> device_copy
      if (!Copy(tensor_device_address.get(), output_addr.get())) {
        MS_LOG(EXCEPTION) << "Sync data error.";
      }
    } else if (output_node_tensor_map[output_addr->pointer_ref_count()] != nullptr) {
      // create new device_address because they may have same ptr but different shape
      auto device_address = output_node_tensor_map[output_addr->pointer_ref_count()];
      tensor_device_address->set_pointer_ref_count(device_address->pointer_ref_count());
      tensor_device_address->set_need_sync_user_data(device_address->need_sync_user_data());
    } else {
      output_node_tensor_map[output_addr->pointer_ref_count()] = tensor_device_address;
      output_addr->Swap(tensor_device_address.get());
    }

    MS_LOG(DEBUG) << "Create device tensor:" << tensor_device_address << ", size: " << kernel_tensor->size()
                  << ", type:" << tensor_device_address->type_id() << ", ptr: " << tensor_device_address->GetPtr()
                  << ", output node:" << output_node->fullname_with_scope() << " output index:" << idx
                  << ", origin output device tensor: " << output_addr;

    tensor_device_address->set_host_shape(out_tensor->shape());
    out_tensor->set_device_address(tensor_device_address);
    out_tensor->set_need_release_device_mem(true);
    outputs->emplace_back(out_tensor);
  }
}

void GEBackend::RunGraph(const std::string &graph_info, const device::DeviceContext *device_context,
                         const VectorRef &args, std::vector<tensor::TensorPtr> *outputs) {
  MS_LOG(INFO) << "Status record: start run graph: " << graph_info;
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(outputs);
  // is_run_save_graph
  std::string graph_name = graph_info;
  bool is_run_save_graph = false;
  if (graph_info.substr(0, kGraphInfoSavePrefixLen) == "save.") {
    graph_name.erase(0, kGraphInfoSavePrefixLen);
    is_run_save_graph = true;
  }

  if (graph_map_.find(graph_name) == graph_map_.end()) {
    MS_LOG(EXCEPTION) << "The graph is not found, graph: " << graph_name;
  }
  MS_EXCEPTION_IF_NULL(device_context->graph_executor_);
  auto func_graph = graph_map_[graph_name];

// for data_dump
#ifndef ENABLE_SECURITY
  bool debugger_actor_need = DumpJsonParser::GetInstance().e2e_dump_enabled();
  if (common::GetEnv("MS_HOOK_ENABLE") == "on") {
    debugger_actor_need = True;
  }
#endif
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (debugger->DebuggerBackendEnabled()) {
    debugger_actor_need = true;
  }
#endif
#ifndef ENABLE_SECURITY
  bool dump_flag = false;
  if (debugger_actor_need) {
    dump_flag = DebugOnStepBegin(func_graph);
  }
#endif

  // for profiling
  bool profile_started = ProfilerOnStepBegin(func_graph, device_context);

  if (is_run_save_graph) {
    device_context->graph_executor_->RunCheckpointGraph(func_graph);
    return;
  }

  if (IsEnableRefMode()) {
    // alloc input(static), output device memory; dynamic input will alloc later
    device_context->graph_executor_->AllocGEInputOutputMemory(func_graph);
    // alloc fixed feature memory when enable gekernel, once | const memory alloc in compilegraph
    device_context->graph_executor_->AllocGEFixMemory();
    // alloc refreshable feature memory
    device_context->graph_executor_->AllocGERefreshableFeatureMemory(func_graph);
    // const alloc in compile graph
  }

  // input, weight from host(args) to device(device_address in graph)
  std::vector<tensor::TensorPtr> inputs_tensor;
  ConstructInputs(func_graph, args, &inputs_tensor, device_context);

  // run graph
  {
    std::vector<tensor::TensorPtr> outputs_tensor;
    const std::map<string, string> compile_options;
    MS_LOG(INFO) << "Start run graph, input size: " << inputs_tensor.size();
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kGraphLaunch,
                                       graph_info);
    auto ret = device_context->graph_executor_->RunGraph(func_graph, inputs_tensor, &outputs_tensor, compile_options);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Launch graph failed, graph id: " + std::to_string(func_graph->graph_id());
    }
  }
  auto ret = device_context->device_res_manager_->SyncAllStreams();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync Stream failed";
  }

  // output ->VectorRef *outputs
  ConstructOutputs(func_graph, outputs, device_context);

// for data_dump
#ifndef ENABLE_SECURITY
  if (debugger_actor_need) {
    DebugOnStepEnd(func_graph, device_context, dump_flag);
  }
#endif

  // for profiling
  ProfilerOnStepEnd(device_context, profile_started);

  // free resource
  if (IsEnableRefMode()) {
    device_context->graph_executor_->FreeGERefreshableFeatureMemory(func_graph);
    device_context->graph_executor_->FreeInputOutputMemory(func_graph);
  }

  graph_run_iter_[func_graph]++;
  MS_LOG(INFO) << "Status record: end run graph: " << graph_info;
  return;
}

bool GEBackend::DebugOnStepBegin(const KernelGraphPtr &func_graph) {
  MS_LOG(INFO) << "Debug on step begin.";
  if (common::GetEnv("ENABLE_MS_GE_DUMP") != "1" &&
      ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE &&
      ConfigManager::GetInstance().iter_num() != 1) {
    MS_LOG(EXCEPTION) << "When using acl dump in data sink mode, sink size must be 1, but got "
                      << ConfigManager::GetInstance().iter_num() << ".";
  }
  if (func_graph->IsDatasetGraph()) {
    return false;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profiler == nullptr || !profiler->IsInitialized()) {
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto &hookDebugger = hooker::HookDebugger::GetInstance();
    if (hookDebugger.IsHookerEnabled()) {
      auto step_count_num = graph_run_iter_[func_graph];
      hookDebugger.HookOnStepBegin(device_id, func_graph, step_count_num, false);
      return true;
    }
    if (common::GetEnv("ENABLE_MS_GE_DUMP") != "1") {
      return ACLDump(device_id, func_graph);
    }
  }
  return false;
}

bool GEBackend::ACLDump(uint32_t device_id, const KernelGraphPtr &graph) {
  std::vector<std::string> all_kernel_names;
  std::vector<std::string> set_dump_names;
  auto all_kernels = graph->execution_order();
  std::for_each(all_kernels.begin(), all_kernels.end(), [&](const auto &k) {
    all_kernel_names.push_back(k->fullname_with_scope());
    auto dump_flag = common::AnfAlgo::GetDumpFlag(k);
    if (dump_flag.has_value() && dump_flag.value().compare("true") == 0) {
      (set_dump_names).push_back(k->fullname_with_scope());
    }
  });

  auto step_count_num = graph_run_iter_[graph];
  if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
    step_count_num = 0;
  }
  DumpJsonParser::GetInstance().UpdateDumpIter(step_count_num);
  MS_LOG(INFO) << "Dump iter: " << step_count_num;

  auto enable_ge_dump = common::GetEnv("ENABLE_MS_GE_DUMP");
  if (DumpJsonParser::GetInstance().async_dump_enabled() && enable_ge_dump != "1") {
    bool is_init = false;
    if ((enable_ge_dump != "1") && !(DumpJsonParser::GetInstance().DumpEnabledForIter())) {
      is_init = true;
    } else {
      std::string dump_path = DumpJsonParser::GetInstance().path();
      std::string dump_path_step = dump_path + "/" + std::to_string(step_count_num);
      auto real_path = FileUtils::CreateNotExistDirs(dump_path_step, false);
      if (!real_path.has_value()) {
        MS_LOG(WARNING) << "Fail to create acl dump dir " << real_path.value();
        return false;
      }
    }
    auto registered_dumper = datadump::DataDumperRegister::Instance().GetDumperForBackend(device::DeviceType::kAscend);
    if (registered_dumper != nullptr) {
      registered_dumper->Initialize();
      if (DumpJsonParser::GetInstance().dump_mode() ==
          static_cast<uint32_t>(mindspore::DumpJsonParser::JsonDumpMode::DUMP_KERNELS_WITH_FLAG)) {
        if (set_dump_names.empty()) {
          MS_LOG(WARNING) << "[set dump] There is no target with dump flag.";
          set_dump_names.push_back("NoSetDumpTarget");
        }
        registered_dumper->EnableDump(device_id, step_count_num, is_init, set_dump_names);
      } else {
        registered_dumper->EnableDump(device_id, step_count_num, is_init, all_kernel_names);
      }
    }
    return true;
  }
  return false;
}

void GEBackend::DebugOnStepEnd(const KernelGraphPtr &graph, const device::DeviceContext *device_context,
                               bool dump_flag) {
  if (!dump_flag) {
    return;
  }
  MS_LOG(INFO) << "Debug on step end. dump_iter: " << DumpJsonParser::GetInstance().cur_dump_iter();
  auto &hookDebugger = hooker::HookDebugger::GetInstance();
  if (hookDebugger.IsHookerEnabled()) {
    device_context->device_res_manager_->SyncAllStreams();
    hookDebugger.HookOnStepEnd();
  } else {
    auto registered_dumper = datadump::DataDumperRegister::Instance().GetDumperForBackend(device::DeviceType::kAscend);
    if (registered_dumper != nullptr) {
      device_context->device_res_manager_->SyncAllStreams();
      registered_dumper->Finalize();
    }
  }
  device_context->device_res_manager_->SyncAllStreams();
}

bool GEBackend::ProfilerOnStepBegin(const KernelGraphPtr &graph, const device::DeviceContext *device_context) {
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  if (profiler == nullptr || !profiler->IsInitialized() || !profiler->GetEnableFlag()) {
    return false;
  }
  if (graph->IsDatasetGraph()) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  if (device_context->GetDeviceType() == device::DeviceType::kAscend) {
    device_context->device_res_manager_->BindDeviceToCurrentThread(false);
    MS_LOG(INFO) << "Dot step start timestamp.";
    profiler->StepStart(graph_run_iter_[graph], device_context->device_res_manager_->GetStream());
    return true;
  }
  return false;
}

void GEBackend::ProfilerOnStepEnd(const device::DeviceContext *device_context, bool profile_started) {
  if (!profile_started) {
    return;
  }
  auto profiler = profiler::Profiler::GetInstance(kAscendDevice);
  MS_EXCEPTION_IF_NULL(device_context);
  device_context->device_res_manager_->BindDeviceToCurrentThread(false);
  device_context->device_res_manager_->SyncAllStreams();
  MS_LOG(INFO) << "Dot step end timestamp.";
  profiler->StepStop();
  device_context->device_res_manager_->SyncAllStreams();
}

FuncGraphPtr GEBackend::BuildDFGraph(const device::DeviceContext *device_context, const FuncGraphPtr &func_graph,
                                     const std::map<std::string, std::shared_ptr<tensor::Tensor>> &init_tensors) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::map<std::string, std::shared_ptr<tensor::Tensor>> real_init_tensors{};
  const auto &infer_need_update_parameter_names = GetInferParameterNames(device_context);

  bool infer = false;
  if (func_graph->has_attr("phase")) {
    std::string phase = func_graph->get_attr("phase")->ToString();
    infer = phase != "train";
  }

  if (infer && !IsEnableRefMode()) {
    for (auto iter = init_tensors.begin(); iter != init_tensors.end(); ++iter) {
      if (infer_need_update_parameter_names.find(iter->first) != infer_need_update_parameter_names.end()) {
        real_init_tensors.emplace(*iter);
      }
    }
  }
  return device_context->graph_executor_->BuildDFGraph(func_graph, real_init_tensors, true);
}

string GEBackend::ExportDFGraph(const device::DeviceContext *device_context, const std::string &file_name,
                                const FuncGraphPtr &anf_graph, bool is_save_to_file) {
  return device_context->graph_executor_->ExportDFGraph(file_name, anf_graph, is_save_to_file);
}

std::unordered_set<std::string> GEBackend::GetInferParameterNames(const device::DeviceContext *device_context) {
  return device_context->graph_executor_->GetInferParameterNames();
}
}  // namespace mindspore::compile
