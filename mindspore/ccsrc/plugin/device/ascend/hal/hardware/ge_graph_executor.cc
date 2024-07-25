/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ge_graph_executor.h"
#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <map>
#include <set>
#include <sstream>
#include "include/transform/graph_ir/types.h"
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
#include "plugin/device/ascend/hal/hardware/ge_memory_allocator.h"
#include "plugin/device/ascend/hal/hardware/ge_utils.h"
#include "plugin/device/ascend/hal/device/ascend_memory_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_device_address.h"
#include "plugin/device/ascend/hal/hardware/ge_graph_optimization.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "plugin/device/ascend/hal/device/ascend_device_synchronizer.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "ge/ge_graph_compile_summary.h"
#include "op_proto/inc/array_ops.h"
#include "mindspore/ops/op_def/nn_op_name.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/compile_cache_context.h"
#include "utils/phase.h"
using InputNameAndType = std::vector<std::pair<std::string, bool>>;
using Data = ::ge::op::Data;
using RefData = ::ge::op::RefData;

namespace mindspore {
namespace device {
namespace ascend {
namespace {
const std::set<std::string> kIgnoreGEShapeOps = {kSoftMarginLossOpName};
mindspore::HashMap<std::string, size_t> feature_memorys;
mindspore::HashMap<std::string, size_t> streams;
constexpr size_t kNeedRecycleOutput = 5;

void GetMeRetDataType(const AbstractBasePtr &cnode_data, std::vector<TypeId> *me_types) {
  MS_EXCEPTION_IF_NULL(cnode_data);

  if (cnode_data->isa<abstract::AbstractNone>()) {
    return;
  }

  if (cnode_data->isa<abstract::AbstractTensor>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    if (me_type == kObjectTypeTensorType) {
      me_type = dyn_cast<TensorType>(cnode_data->BuildType())->element()->type_id();
      (void)me_types->emplace_back(me_type);
    }
    return;
  }
  if (cnode_data->isa<abstract::AbstractScalar>()) {
    TypeId me_type = cnode_data->BuildType()->type_id();
    (void)me_types->emplace_back(me_type);
    return;
  }
  auto abstract_tuple = cnode_data->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  auto elements = abstract_tuple->elements();
  for (size_t i = 0; i < abstract_tuple->size(); ++i) {
    GetMeRetDataType(elements[i], me_types);
  }
}

transform::TensorOrderMap GetDefaultParams(const FuncGraphPtr &anf_graph,
                                           std::map<std::string, ShapeVector> *origin_shape) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      MS_EXCEPTION_IF_NULL(tensor);
      origin_shape->emplace(para->name(), tensor->shape_c());
      // need ref shape when auto parallel
      auto build_shape = para->abstract()->BuildShape();
      if (build_shape != nullptr) {
        (void)tensor->MetaTensor::set_shape(build_shape->cast<abstract::ShapePtr>()->shape());
        MS_LOG(INFO) << "ref abstract Parameter: " << para->name() << ", tensor: " << tensor->ToString();
      }
      res.emplace(para->name(), tensor);
      MS_LOG(DEBUG) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}

void RevertOriginShape(const KernelGraphPtr &anf_graph, const std::map<std::string, ShapeVector> &origin_shape) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto it = origin_shape.find(para->name());
      if (it == origin_shape.end()) {
        MS_LOG(ERROR) << "Failed to find input " << para->name() << " in input_shape " << origin_shape;
        continue;
      }
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      (void)tensor->MetaTensor::set_shape(it->second);
      MS_LOG(INFO) << "ref abstract Parameter: " << para->name() << ", tensor: " << tensor->ToString();
    }
  }
}

std::vector<transform::GeTensorPtr> GetInputTensors(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  transform::TensorOrderMap init_input_map;
  std::vector<tensor::TensorPtr> init_input;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      (void)init_input_map.emplace(para->name(), value->cast<std::shared_ptr<tensor::Tensor>>());
    }
  }
  (void)std::transform(init_input_map.begin(), init_input_map.end(), std::back_inserter(init_input),
                       [](const std::pair<std::string, tensor::TensorPtr> &item) { return item.second; });
  return transform::ConvertInputTensors(init_input, kOpFormat_NCHW);
}

void RunGEInitGraph(const FuncGraphPtr &anf_graph) {
  MS_LOG(DEBUG) << "ExecInitGraph start.";
  MS_EXCEPTION_IF_NULL(anf_graph);

  transform::RunOptions run_options;
  run_options.name = "init_subgraph." + anf_graph->ToString();

  auto graph_runner = transform::CheckAndGetGraphRunner(run_options);
  if (graph_runner == nullptr) {
    return;
  }

  std::vector<transform::GeTensorPtr> ge_tensors;
  std::vector<transform::GeTensorPtr> ge_outputs;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    transform::Status ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << run_options.name << " graph failed.";
    }

    MS_LOG(DEBUG) << "Exec " << run_options.name << " graph success.";

    if ((ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::DISTRIBUTION) &&
        (transform::GetGraphByName(BROADCAST_GRAPH_NAME) != nullptr)) {
      run_options.name = BROADCAST_GRAPH_NAME;
      ge_tensors = GetInputTensors(anf_graph);
      ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
      if (ret != transform::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec BROADCAST_GRAPH_NAME failed.";
      }
      MS_LOG(DEBUG) << "Exec broadcast graph success.";
    }
  }
}

void UpdateOutputNodeShape(const AnfNodePtr &node, size_t index, TypeId output_type, const ShapeVector &output_shape) {
  MS_EXCEPTION_IF_NULL(node);
  std::string name;
  if (node->isa<CNode>()) {
    name = common::AnfAlgo::GetCNodeName(node);
  }
  size_t total_output_num = AnfAlgo::GetOutputElementNum(node);
  if (index >= total_output_num) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Invalid output index " << index << ", node " << node->fullname_with_scope()
                                      << " has " << total_output_num << " outputs.";
  }
  std::vector<TypeId> types = {};
  std::vector<ShapeVector> shapes = {};
  for (size_t i = 0; i < total_output_num; ++i) {
    if (i == index && kIgnoreGEShapeOps.count(name) == 0) {
      types.push_back(output_type);
      shapes.push_back(output_shape);
    } else {
      types.push_back(common::AnfAlgo::GetOutputInferDataType(node, i));
      (void)shapes.emplace_back(common::AnfAlgo::GetOutputInferShape(node, i));
    }
  }
  common::AnfAlgo::SetOutputInferTypeAndShape(types, shapes, node.get());
}

void SetDynamicShapeAttr(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto nodes = TopoSort(kernel_graph->output());
  for (auto &node : nodes) {
    if (common::AnfAlgo::IsDynamicShape(node)) {
      MS_LOG(DEBUG) << "Set Dynamic Shape Attr to Node : " << node->fullname_with_scope();
      kernel_graph->SetGraphDynamicAttr(true);
      return;
    }
  }
}

bool BuildFakeGraph(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_before_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_before_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif
  (void)setenv("GE_TRAIN", IsGeTrain() ? "1" : "0", 1);
  if (!AddFakeGraph(anf_graph)) {
    MS_LOG(ERROR) << "Add fake graph failed";
    return false;
  }
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_after_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_after_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif
  return true;
}

void ClearForwardOutputAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->has_flag(kFlagPyNativeRunInGraph)) {
    return;
  }
  const auto &input_nodes = graph->input_nodes();
  for (const auto &input : input_nodes) {
    MS_EXCEPTION_IF_NULL(input);
    auto parameter = input->cast<ParameterPtr>();
    if (parameter != nullptr) {
      if (parameter->has_user_data(kForwardOutput)) {
        auto device_address = AnfAlgo::GetMutableOutputAddr(parameter, 0);
        auto new_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(device_address, device_context);
        AnfAlgo::SetOutputAddr(new_address, 0, parameter.get());
        MS_LOG(DEBUG) << "Clear old address " << device_address.get() << " and set new address " << new_address.get()
                      << " to parameter " << parameter->name();
      }
    }
  }
}

class ContextReset {
 public:
  explicit ContextReset(DeviceContext *device_context) : device_context_(device_context) {}
  ~ContextReset() {
    if (device_context_ != nullptr && device_context_->device_res_manager_ != nullptr) {
      device_context_->device_res_manager_->BindDeviceToCurrentThread(true);
    }
  }

 private:
  DeviceContext *device_context_;
};

void UpdateFMTracker(size_t feature_memory_size, const std::string &graph_name) {
  device::tracker::CALL_MEMORY_TRACKER(AllocMemBlock, 0, feature_memory_size, "Ascend",
                                       AscendMemAdapter::GetInstance()->GetActualPeakMemory(), 0, 0, 0);
  device::tracker::CALL_MEMORY_TRACKER(FreeMemBlock, 0, 0, 0);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "RunGeGraph", "RunGeGraph", graph_name);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, "RunGeGraph", feature_memory_size, 0,
                                                 device::tracker::MemType::kGeFixed);
}

bool CacheFileExists(const std::string &name) {
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto dep_files_hash = compile_cache_context.CompileCacheDepFilesHash();
  auto ge_graph_key = name;
  if (!dep_files_hash.empty()) {
    ge_graph_key = dep_files_hash + "_" + ge_graph_key;
  }
  auto ge_cache_path = Common::GetCompilerCachePath() + kGeCache;
  ge_graph_key = NormalizeString(ge_graph_key);
  auto cache_idx_file = ge_cache_path + "/" + ge_graph_key + ".idx";
  struct stat buffer;
  bool ret = stat(cache_idx_file.c_str(), &buffer) == 0;
  MS_LOG(INFO) << "Cached index file name: " << cache_idx_file << " exists: " << ret;
  return ret;
}

void SetOutputs(const std::vector<KernelWithIndex> &graph_outputs,
                const std::vector<transform::GeTensorPtr> &ge_outputs, const std::vector<TypeId> &me_types) {
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const auto &[output_node, idx] = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_outputs[i]);
    const auto &tensor = ge_outputs[i];
    auto output_addr = AnfAlgo::GetMutableOutputAddr(output_node, idx);
    ::ge::Placement dp = tensor->GetTensorDesc().GetPlacement();
    auto &&ge_data_uni = tensor->ResetData();
    auto deleter = ge_data_uni.get_deleter();
    auto ge_data = ge_data_uni.release();
    MS_EXCEPTION_IF_NULL(ge_data);
    if (dp == ::ge::kPlacementHost) {
      constexpr int64_t kTensorAlignBytes = 64;
      if (reinterpret_cast<uintptr_t>(ge_data) % kTensorAlignBytes != 0) {
        MS_LOG(EXCEPTION) << "Skip zero-copy ge tensor " << reinterpret_cast<uintptr_t>(ge_data)
                          << ", bytes not aligned with expected.";
      }
      if (me_types[i] == TypeId::kObjectTypeString) {
        MS_LOG_WITH_NODE(EXCEPTION, output_node) << "It is not supported that Output node "
                                                 << output_node->DebugString() << "'s output data type is string now.";
      }
      MS_LOG(DEBUG) << "Zero-copy ge tensor " << reinterpret_cast<uintptr_t>(ge_data) << " as aligned with "
                    << kTensorAlignBytes << " types.";
      output_addr->set_is_ptr_persisted(false);
      output_addr->set_from_mem_pool(false);
      output_addr->set_deleter(deleter);
      output_addr->set_ptr(ge_data);
      output_addr->SetSize(tensor->GetSize());
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, output_node) << "It is not supported that Output node " << output_node->DebugString()
                                               << "'s output data's placement is device now.";
    }
    auto actual_shapes = tensor->GetTensorDesc().GetShape().GetDims();
    UpdateOutputNodeShape(output_node, idx, me_types[i], actual_shapes);
  }
}

void SetOutput(GeDeviceResManager *res_manager, GeTensor *ge_output, const AnfNodePtr &output_node, size_t idx) {
  if (output_node->isa<ValueNode>()) {
    auto &&ge_data_uni = ge_output->ResetData();
    auto deleter = ge_data_uni.get_deleter();
    auto ge_data = ge_data_uni.release();
    deleter(ge_data);
    return;
  }
  auto actual_shapes = ge_output->GetTensorDesc().GetShape().GetDims();
  for (size_t i = 0; i < actual_shapes.size(); ++i) {
    if (actual_shapes[i] < 0) {
      MS_LOG(EXCEPTION) << "Output shape must be greater than 0, but got " << actual_shapes;
    }
  }
  auto output_addr = AnfAlgo::GetMutableOutputAddr(output_node, idx, false);
  output_addr->SetSize(ge_output->GetSize());
  auto &&ge_data_uni = ge_output->ResetData();
  auto deleter = ge_data_uni.get_deleter();
  auto ge_data = ge_data_uni.release();
  MS_EXCEPTION_IF_NULL(ge_data);
  output_addr->set_is_ptr_persisted(false);
  output_addr->set_from_mem_pool(false);
  output_addr->set_deleter(deleter);
  output_addr->set_ptr(ge_data);
  auto placement = ge_output->GetTensorDesc().GetPlacement();
  if (placement == ::ge::kPlacementHost) {
    MS_LOG(DEBUG) << output_node->DebugString() << "'s output data's placement is host";
    size_t size = ge_output->GetSize();
    void *mem = res_manager->AllocateMemory(size);
    if (mem == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, output_node)
        << "Allocate memory failed, memory size:" << size << ", output_node: " << output_node->ToString();
    }
    output_addr->set_from_mem_pool(true);
    output_addr->set_ptr(mem);
    auto *ascend_addr = dynamic_cast<AscendDeviceAddress *>(output_addr.get());
    MS_EXCEPTION_IF_NULL(ascend_addr);
    ascend_addr->SyncHostToDevice(size, ge_data);
  }
  // Update shape in kernel tensor.
  const auto &kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, idx);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  kernel_tensor->SetShapeVector(actual_shapes);
  MS_LOG(INFO) << "[ZeroCopy] Update output " << output_node->DebugString() << " address to "
               << output_addr->GetMutablePtr() << ", shape:" << actual_shapes
               << ", type: " << TypeIdToString(output_addr->type_id()) << ", format: " << output_addr->format();
}

void SetDynamicOutputs(const std::vector<KernelWithIndex> &graph_outputs, std::vector<GeTensor> *ge_outputs,
                       GeDeviceResManager *res_manager) {
  MS_EXCEPTION_IF_NULL(res_manager);
  size_t ge_outputs_index = 0;
  size_t ge_outputs_size = ge_outputs->size();
  for (size_t i = 0; i < graph_outputs.size(); ++i) {
    const auto &[output_node, idx] = common::AnfAlgo::FetchRealNodeSkipMonadControl(graph_outputs[i]);
    MS_EXCEPTION_IF_NULL(output_node);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    if (common::AnfAlgo::IsNoOuputNode(output_node)) {
      continue;
    }
    if (ge_outputs_index >= ge_outputs_size) {
      MS_LOG(EXCEPTION) << "GE data access is out of bounds, which the current index value is " << ge_outputs_index
                        << ", the total number of GE output is " << ge_outputs_size << ".";
    }
    SetOutput(res_manager, &((*ge_outputs)[ge_outputs_index++]), output_node, idx);
  }
}

void SetDynamicOutputsForKernel(const std::vector<GeTensor> &ge_outputs,
                                const std::vector<KernelTensor *> &kernel_outputs, const AnfNodeWeakPtr &node,
                                GeDeviceResManager *res_manager) {
  size_t index_ge = 0;
  for (size_t index_kernel = 0; index_kernel < kernel_outputs.size(); ++index_kernel) {
    auto kernel_output = kernel_outputs[index_kernel];
    if (kernel_output == nullptr) {
      continue;
    }
    std::vector<TypeId> monad_type_id = {TypeId::kObjectTypeMonad, TypeId::kObjectTypeUMonad,
                                         TypeId::kObjectTypeIOMonad};
    if (std::any_of(monad_type_id.begin(), monad_type_id.end(),
                    [&kernel_output](const TypeId type_id) { return type_id == kernel_output->dtype_id(); })) {
      continue;
    }

    if (index_ge >= ge_outputs.size()) {
      MS_LOG(EXCEPTION) << "index " << index_ge << " is larger than ge_outputs size: " << ge_outputs.size();
    }
    auto ge_output = ge_outputs[index_ge];
    SetOutput(res_manager, &ge_output, node.lock(), index_kernel);
    ++index_ge;
  }
}
}  // namespace

void GeGraphExecutor::SetFlagIgnoreDevicePtr(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &param_list = graph->parameters();
  for (const auto &param_node : param_list) {
    MS_EXCEPTION_IF_NULL(param_node);
    if (param_node->has_user_data(transform::kNoNeedAllocDeviceAddress)) {
      auto output_addr = AnfAlgo::GetMutableOutputAddr(param_node, 0, false);
      MS_EXCEPTION_IF_NULL(output_addr);
      output_addr->UpdateFlag(device::kDeviceAddressFlagIgnoreDevicePtr);
      MS_LOG(INFO) << "Node " << param_node->fullname_with_scope()
                   << " does not need device memory, so set kDeviceAddressFlagIgnoreDevicePtr.";
    }
  }
}

void GeGraphExecutor::InitGraphInfo(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  SetFlagIgnoreDevicePtr(graph);
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kg);
  BuildInputDataGeTensor(kg);
  BuildOutputDataGeTensor(kg);
}

void GeGraphExecutor::BuildInputDataGeTensor(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start BuildInputDataGeTensor, kernel graph: " << kernel_graph->ToString();
  std::vector<GeTensor> ge_inputs;
  std::vector<DeviceAddress *> device_addrs;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> need_update_input;
  std::vector<AnfNodeWeakPtr> ge_input_nodes;
  auto ge_input_list = kernel_graph->user_data<transform::GEInputList>();
  if (ge_input_list) {
    ge_input_nodes = ge_input_list->ge_inputs;
  }
  for (const auto &node_wptr : ge_input_nodes) {
    auto node = node_wptr.lock();
    if (!node) {
      MS_LOG(ERROR) << "Get node lock failed, kerne graph: " << kernel_graph->ToString();
      continue;
    }
    auto name = node->fullname_with_scope();
    MS_LOG(INFO) << "Build input ge tensor: " << name << ", kernel graph: " << kernel_graph->graph_id();
    auto output_addr = AnfAlgo::GetMutableOutputAddr(node, 0, false);
    (void)device_addrs.emplace_back(output_addr.get());
    auto shapes = trans::GetRuntimePaddingShape(node, 0);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
    auto ge_tensor_desc = transform::TransformUtil::GetGeTensorDesc(shapes, host_type, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(ge_tensor_desc);
    ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
    GeTensor ge_tensor(*ge_tensor_desc);
    if (output_addr->GetMutablePtr() != nullptr) {
      MS_LOG(INFO) << "Node: " << name << " Has addr, size: " << output_addr->GetSize();
      if (ge_tensor.SetData(reinterpret_cast<uint8_t *>(output_addr->GetMutablePtr()), output_addr->GetSize(),
                            [](void *) {}) != ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "SetData failed, ge input data " << ge_inputs.size() << " name: " << name
                          << " size: " << output_addr->GetSize();
      }
      MS_LOG(INFO) << "ge input data " << ge_inputs.size() << " name: " << name << " size: " << output_addr->GetSize();
    }
    // The device address of input tensor may change every step.
    // Always keep the input node address consistent with the input tensor address.
    (void)need_update_input.emplace_back(node, ge_inputs.size());
    (void)ge_inputs.emplace_back(std::move(ge_tensor));
  }
  input_datas_[kernel_graph.get()] = {ge_inputs, device_addrs, need_update_input};
  MS_LOG(INFO) << "BuildInputDataGeTensor finish.";
}

void GeGraphExecutor::BuildOutputDataGeTensor(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start BuildOutputDataGeTensor, kernel graph: " << kernel_graph->ToString();
  std::vector<GeTensor> ge_outputs;
  std::vector<DeviceAddress *> device_addrs;
  std::vector<std::pair<AnfNodeWeakPtr, size_t>> graph_outputs;
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    auto index = output_with_index.second;
    MS_EXCEPTION_IF_NULL(output_node);
    if (HasAbstractMonad(output_node)) {
      continue;
    }
    if (common::AnfAlgo::IsNoOuputNode(output_node)) {
      continue;
    }
    auto real_index = output_node->isa<ValueNode>() ? 0 : index;
    auto device_addr = AnfAlgo::GetMutableOutputAddr(output_node, real_index, false);
    (void)device_addrs.emplace_back(device_addr.get());
    auto shapes = trans::GetRuntimePaddingShape(output_node, real_index);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(output_node, real_index);
    auto ge_tensor_desc = transform::TransformUtil::GetGeTensorDesc(shapes, host_type, kOpFormat_DEFAULT);
    MS_EXCEPTION_IF_NULL(ge_tensor_desc);
    ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
    GeTensor ge_tensor(*ge_tensor_desc);
    (void)ge_outputs.emplace_back(std::move(ge_tensor));
    (void)graph_outputs.emplace_back(output_node, index);
  }
  MS_EXCEPTION_IF_CHECK_FAIL(
    ge_outputs.size() == graph_outputs.size(),
    "The size of ge_outputs and graph_outputs check error, kernel graph: " + kernel_graph->ToString());
  output_datas_[kernel_graph.get()] = {ge_outputs, device_addrs, graph_outputs};
  MS_LOG(INFO) << "BuildOutputDataGeTensor finish.";
}

GeDeviceResManager *GeGraphExecutor::ResManager() const {
  MS_EXCEPTION_IF_NULL(device_context_);
  auto res_manager = dynamic_cast<GeDeviceResManager *>(device_context_->device_res_manager_.get());
  MS_EXCEPTION_IF_NULL(res_manager);
  return res_manager;
}

void GeGraphExecutor::PreprocessBeforeRun(const KernelGraphPtr &graph) {
  auto ret = CompileGraph(graph, {});
  if (!ret) {
    MS_LOG(EXCEPTION) << "Compile graph fail, graph id: " << graph->graph_id();
  }
  InitGEFixMemory(graph, 0);
}

bool GeGraphExecutor::BuildGraph(const KernelGraphPtr &graph, const transform::TensorOrderMap &tensor_order_map) {
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto use_compile_cache = compile_cache_context.UseCompileCache();
  auto name = GetGraphName(graph);
  bool has_cache = CacheFileExists(name);
  if (use_compile_cache && has_cache) {
    MS_LOG(INFO) << "Use ge compile cache, and skip specific optimization and ge_adapter execution";
    if (!BuildFakeGraph(graph)) {
      return false;
    }
  } else {
    (void)BuildDFGraph(graph, tensor_order_map, false);
  }
  return true;
}

bool GeGraphExecutor::CompileGraph(const KernelGraphPtr &graph,
                                   const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "ge graph executor compile graph " << graph->ToString();
  auto &compile_cache_context = CompileCacheContext::GetInstance();
  auto use_compile_cache = compile_cache_context.UseCompileCache();
  std::map<std::string, ShapeVector> origin_shape;
  const auto &tensor_order_map = GetDefaultParams(graph, &origin_shape);
  auto name = GetGraphName(graph);
  bool has_cache = CacheFileExists(name);
  if (use_compile_cache && has_cache) {
    MS_LOG(INFO) << "Use ge compile cache, and skip specific optimization and ge_adapter execution";
    if (!BuildFakeGraph(graph)) {
      return false;
    }
  } else {
    (void)BuildGraph(graph, tensor_order_map);
  }
  SetDynamicShapeAttr(graph);
  transform::RunOptions run_options;
  run_options.name = GetGraphName(graph);
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }
  // create loop var
  RunInitGraph(run_options.name);
  if (graph->is_dynamic_shape()) {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    auto ret = graph_runner->CompileGraph(run_options);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Compile graph " << run_options.name << " failed.";
    }
  } else {
    ::ge::CompiledGraphSummaryPtr ge_graph_summary = nullptr;
    {
      // Release GIL before calling into (potentially long-running) C++ code
      GilReleaseWithCheck gil_release;
      auto ret = graph_runner->CompileGraph(run_options, &ge_graph_summary);
      if (ret != transform::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Compile graph " << run_options.name << " failed.";
      }
    }
    GraphSummary summary(ge_graph_summary);
    MS_LOG(INFO) << "Graph " << run_options.name << " summary: " << summary.ToString();
    ge_message_manager_.SetSummary(run_options.name, summary);
    ge_message_manager_.SetFeatureMemory(run_options.name, summary.fixed_memory_size);
    ge_message_manager_.SetStream(run_options.name, summary.stream_num);
    AddRefCorrespondPairs(graph, summary.io_indexes);
    // if not static, set graph dynamic
    if (!summary.is_static) {
      graph->SetGraphDynamicAttr(true);
    }
  }
  GEMemoryAllocator::ProcessGraphDeviceAddress(graph, device_context_, ResManager());

  graph->set_run_mode(RunMode::kGraphMode);
  graph->set_memory_managed_by_ge(true);
  if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
    graph->set_is_loop_count_sink(true);
  }
  RevertOriginShape(graph, origin_shape);
  return true;
}

std::vector<std::pair<uint32_t, uint32_t>> GeGraphExecutor::GetGraphRefIndexes(const KernelGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &key = GetGraphName(graph);
  if (io_indexes_.count(key) != 0) {
    return io_indexes_.at(key);
  }
  return {};
}

void GeGraphExecutor::SetGraphWorkspaceMemory(const KernelGraphPtr &graph, void *device_ptr, size_t size) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_runner = transform::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  transform::RunOptions run_options;
  run_options.name = GetGraphName(graph);
  auto ret = graph_runner->UpdateRefreshableMemory(run_options, device_ptr, size);
  if (ret != transform::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "UpdateRefreshableMemory for graph " << run_options.name << " failed.";
  }
}

size_t GeGraphExecutor::GetGraphWorkSpaceMemory(const std::string &graph_name) const {
  if (!ge_message_manager_.SummaryExist(graph_name)) {
    MS_LOG(INFO) << "The summary of graph: " << graph_name << " is not exist.";
    return 0;
  }
  auto summary = ge_message_manager_.GetSummary(graph_name);
  return summary.workspace_memory_size;
}

void GeGraphExecutor::AllocGEFixMemory() const {
  MS_LOG(INFO) << "Start AllocGEFixMemory";
  auto res_manager = ResManager();
  auto alloc_func = [&res_manager](size_t size) { return res_manager->AllocateStaticMemory(size); };
  auto update_func = [](bool is_refreshable, const transform::RunOptions &options, const void *const memory,
                        size_t size) {
    MS_LOG(INFO) << "Update GE fixed memory, graph name: " << options.name << ", is_refreshable: " << is_refreshable
                 << ", size: " << size << ", memory: " << memory;
    if (common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat)) {
      std::cout << "[MS_RUNTIME_PROF]"
                << "Update GE fixed memory, graph name: " << options.name << ", is_refreshable: " << is_refreshable
                << ", size: " << size << ", memory: " << memory << std::endl;
    }
    auto graph_runner = transform::GetGraphRunner();
    MS_EXCEPTION_IF_NULL(graph_runner);
    if (is_refreshable) {
      return graph_runner->SetFixedMemory(options, memory, size);
    }
    return graph_runner->UpdateFeatureMemory(options, memory, size);
  };
  GEMemoryManager::Instance().AllocGEMemory(alloc_func, update_func);
}

void GeGraphExecutor::InitGEFixMemory(const KernelGraphPtr &graph, size_t stream_id) const {
  transform::RunOptions run_options;
  run_options.name = GetGraphName(graph);
  if (graph->is_dynamic_shape()) {
    return;
  }
  auto summary = ge_message_manager_.GetSummary(run_options.name);
  GEMemoryAllocator::AllocGraphMemory(run_options, graph, summary, stream_id, ResManager());
}

bool GeGraphExecutor::CompileGraphForKernel(const KernelGraphPtr &graph) { return CompileGraph(graph, {}); }

void GeGraphExecutor::AddRefCorrespondPairs(const KernelGraphPtr &graph,
                                            const std::vector<std::pair<uint32_t, uint32_t>> &io_indexes) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Start convert io_indexes to ref_map, kernel graph: " << graph->ToString();

  std::map<session::AnfWithOutIndex, session::AnfWithOutIndex> ref_out_in_map = {};
  auto graph_inputs_all = graph->parameters();
  std::vector<AnfNodePtr> graph_inputs = {};
  std::map<uint32_t, uint32_t> input_index_ge_to_kg;
  uint32_t ge_index = 0;
  uint32_t kg_index = 0;
  for (auto &node : graph_inputs_all) {
    MS_EXCEPTION_IF_NULL(node);
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (HasAbstractMonad(node)) {
      MS_LOG(INFO) << "Input node: " << node->DebugString() << " is a monad parameter, skip.";
      ++kg_index;
      continue;
    }

    if (abs->isa<abstract::AbstractSequence>()) {
      MS_LOG(INFO) << "Input node: " << node->DebugString() << " is a tuple/list parameter, skip.";
      continue;
    }

    graph_inputs.emplace_back(node);
    input_index_ge_to_kg[ge_index] = kg_index;
    ++kg_index;
    ++ge_index;
  }

  std::vector<common::KernelWithIndex> graph_outputs_all = {};
  common::AnfAlgo::GetRealInputs(graph->get_return(), &graph_outputs_all);
  std::vector<common::KernelWithIndex> graph_outputs = {};
  std::map<uint32_t, uint32_t> output_index_ge_to_kg;

  ge_index = 0;
  kg_index = 0;
  for (auto &node_with_index : graph_outputs_all) {
    if (common::AnfAlgo::IsNoOuputNode(node_with_index.first) || HasAbstractMonad(node_with_index.first)) {
      MS_LOG(INFO) << "Output node: " << node_with_index.first->fullname_with_scope()
                   << " is a no output node or monad node, skip.";
      ++kg_index;
      continue;
    }

    graph_outputs.emplace_back(node_with_index);
    output_index_ge_to_kg[ge_index] = kg_index;
    ++kg_index;
    ++ge_index;
  }

  std::vector<std::pair<uint32_t, uint32_t>> kg_io_indexes;
  for (auto in_out_index : io_indexes) {
    if (in_out_index.first >= graph_inputs.size() || in_out_index.second >= graph_outputs.size()) {
      MS_LOG(EXCEPTION) << "The io_indexes out of range, input index: " << in_out_index.first
                        << ", output index: " << in_out_index.second << ", graph input size: " << graph_inputs.size()
                        << ", graph output size: " << graph_outputs.size();
    }

    session::AnfWithOutIndex origin_node = std::make_pair(graph_inputs[in_out_index.first], 0);
    session::AnfWithOutIndex final_node = graph_outputs[in_out_index.second];
    if (origin_node.first == final_node.first) {
      kg_io_indexes.emplace_back(
        std::make_pair(input_index_ge_to_kg[in_out_index.first], output_index_ge_to_kg[in_out_index.second]));
      MS_LOG(INFO) << "The origin node is same as final node, node: " << origin_node.first->fullname_with_scope();
      continue;
    }
    if (ref_out_in_map.count(final_node) != 0) {
      MS_LOG(INFO) << "The node is already in ref_out_in_map, node: " << final_node.first->fullname_with_scope()
                   << ", index: " << final_node.second;
      continue;
    }
    // if input node is not abstract ref, set ref may cause memory reuse error
    auto abs = origin_node.first->abstract();
    if (!abs->isa<abstract::AbstractRefTensor>()) {
      MS_LOG(INFO) << "The node is not abstract tensor: " << final_node.first->fullname_with_scope()
                   << ", index: " << final_node.second;
      continue;
    }

    ref_out_in_map.emplace(final_node, origin_node);
    kg_io_indexes.emplace_back(
      std::make_pair(input_index_ge_to_kg[in_out_index.first], output_index_ge_to_kg[in_out_index.second]));
    MS_LOG(INFO) << "Convert io_index [" << in_out_index.first << ", " << in_out_index.second
                 << "] to ref_out_in_map, final_node: " << final_node.first->fullname_with_scope()
                 << ", index:" << final_node.second << ", origin_node: " << origin_node.first->fullname_with_scope()
                 << ", index: " << origin_node.second;
  }

  io_indexes_[GetGraphName(graph)] = kg_io_indexes;
  graph->set_ref_out_in_map(ref_out_in_map);
}

bool GeGraphExecutor::CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) {
  MS_EXCEPTION_IF_NULL(graph);

  auto graph_name = GetGraphName(graph);
  uint64_t start_time = profiler::GetClockSyscnt();

  // cppcheck-suppress unreadVariable
  ContextReset reset_context(device_context_);
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kg);

  if (IsEnableRefMode()) {
    auto ret = CompileGraph(kg, compile_options);

    (void)profiler::CollectHostInfo("Ascend", "CompileGraph", "GeCompileGraph_" + graph_name, start_time,
                                    profiler::GetClockSyscnt(), 1);
    InitGEFixMemory(kg, 0);
    return ret;
  } else {
    // delete SetCPUMemManager when delete env MS_DISABLE_REF_MODE
    ResManager()->SetCPUMemManager();
    std::map<std::string, ShapeVector> origin_shape;
    const auto &tensor_order_map = GetDefaultParams(graph, &origin_shape);
    auto &compile_cache_context = CompileCacheContext::GetInstance();
    auto use_compile_cache = compile_cache_context.UseCompileCache();
    std::set<KernelGraphPtr> memo;
    GEGraphOptimization::GetInstance().OptimizeGEGraph(kg, &memo);
    if (use_compile_cache) {
      MS_LOG(INFO) << "Use ge compile cache, and skip specific optimization and ge_adapter execution";

      if (!BuildFakeGraph(kg)) {
        (void)profiler::CollectHostInfo("Ascend", "CompileGraph", "GeCompileGraph_" + graph_name, start_time,
                                        profiler::GetClockSyscnt(), 1);
        return false;
      }
    } else {
      (void)BuildGraph(kg, tensor_order_map);
    }
    SetDynamicShapeAttr(kg);
    GEMemoryAllocator::AllocInputHostMemory(kg, device_context_);
    GEMemoryAllocator::AllocOutputHostMemory(kg, device_context_);
    kg->set_run_mode(RunMode::kGraphMode);
    if (ConfigManager::GetInstance().dataset_mode() == DatasetMode::DS_SINK_MODE) {
      kg->set_is_loop_count_sink(true);
    }
    // copy init weight to device
    RunGEInitGraph(kg);
    RevertOriginShape(kg, origin_shape);
    (void)profiler::CollectHostInfo("Ascend", "CompileGraph", "GeCompileGraph_" + graph_name, start_time,
                                    profiler::GetClockSyscnt(), 1);
    return true;
  }
}

size_t GeGraphExecutor::GetGraphFeatureMemory(const FuncGraphPtr &graph) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  MS_LOG(WARNING) << "Need Profile Memory, graph: " << graph_name
                  << ", stream: " << ge_message_manager_.GetStream(graph_name);
  if (disable_ge_kernel_) {
    auto max_static_memory_size = ResManager()->GetMaxUsedMemorySize();
    auto feature_memory_size = ge_message_manager_.GetFeatureMemory(graph_name);
    auto total_memory_size = max_static_memory_size + feature_memory_size;
    AscendMemAdapter::GetInstance()->UpdateActualPeakMemory(total_memory_size);
    UpdateFMTracker(feature_memory_size, graph_name);
    return feature_memory_size;
  }
  return 0;
}
int64_t GeGraphExecutor::CurGraphSinkSize(std::string graph_name) {
  int64_t sink_size = -1;
  auto result = graph_sink_size_.find(graph_name);
  if (result != graph_sink_size_.end()) {
    sink_size = result->second;
  } else {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    if (ConfigManager::GetInstance().dataset_mode() == DS_SINK_MODE &&
        ms_context->get_param<bool>(MS_CTX_ENABLE_LOOP_SINK)) {
      sink_size = ConfigManager::GetInstance().iter_num();
    }
    MS_LOG(INFO) << "Graph [" << graph_name << "] sink size is " << sink_size;
    graph_sink_size_.insert(std::pair(graph_name, sink_size));
  }
  return sink_size;
}

bool GeGraphExecutor::RunGraphRefModeInnner(const FuncGraphPtr &graph, const std::vector<GeTensor> &ge_inputs,
                                            std::vector<GeTensor> *ge_outputs, void *stream) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  RunInitGraph(graph_name);
  MS_LOG(INFO) << "GE run graph start in ref mode, graph: " << graph_name << ".";
  (void)ResManager()->BindDeviceToCurrentThread(false);

  // call ge rungraph
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  transform::RunOptions run_options;
  run_options.name = graph_name;
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  bool is_dynamic_shape = kg->is_dynamic_shape();
  if (IsMemoryPoolRecycle() && !is_dynamic_shape) {
    auto max_static_memory_size = ResManager()->GetMaxUsedMemorySize();
    auto feature_memory_size = ge_message_manager_.GetFeatureMemory(graph_name);
    if (feature_memory_size != 0) {
      size_t total_memory_size = max_static_memory_size + feature_memory_size;
      size_t max_hbm_memory_size = static_cast<size_t>(AscendMemAdapter::GetInstance()->GetMsUsedHbmSize());
      AscendMemAdapter::GetInstance()->UpdateActualPeakMemory(total_memory_size);
      UpdateFMTracker(feature_memory_size, graph_name);
      if (common::IsNeedMemoryStatistic()) {
        MS_LOG(WARNING) << "Now Memory Status, graph: " << graph_name
                        << ", max_static_memory_size: " << max_static_memory_size
                        << ", feature_memory_size: " << feature_memory_size
                        << ", max_hbm_memory_size: " << max_hbm_memory_size;
      }
      if (total_memory_size > max_hbm_memory_size) {
        MS_LOG(EXCEPTION) << "Memory pool not enough, graph: " << graph_name
                          << ", max_static_memory_size: " << max_static_memory_size
                          << ", feature_memory_size: " << feature_memory_size
                          << ", max_hbm_memory_size: " << max_hbm_memory_size;
      }
    }
  }

  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    if (IsNeedNotifyTTP(graph)) {
      MS_LOG(INFO) << "Found optimizer sub graph and send event to mindio";
      auto sync_ret = ResManager()->SyncStream();
      if (!sync_ret) {
        MS_LOG(EXCEPTION) << "Sync stream failed";
      } else {
        mindio::MindIOAdapter::GetInstance()->NotifyStartUpdatingOs();
      }
    }
    transform::Status ret =
      transform::RunGraphWithStreamAsync(graph_runner, run_options, stream, ge_inputs, ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec graph failed";
    }
  }
  return true;
}

bool GeGraphExecutor::RunGraphRefMode(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  MS_LOG(INFO) << "Run graph begin, inputs size is: " << inputs.size() << ", " << graph_name;
  KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  std::vector<GeTensor> ge_inputs = GenerateInputGeTensor(kg);
  std::vector<GeTensor> ge_outputs = GenerateOutputGeTensor(kg);
  auto ret = RunGraphRefModeInnner(graph, ge_inputs, &ge_outputs, ResManager()->GetStream());
  if (!ret) {
    return ret;
  }
  if (kg->is_dynamic_shape()) {
    auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
    SetDynamicOutputs(graph_outputs, &ge_outputs, ResManager());
    auto sync_ret = ResManager()->SyncStream();
    if (!sync_ret) {
      MS_LOG(EXCEPTION) << "Sync stream failed";
    }
  }
  ClearForwardOutputAddress(kg, device_context_);
  return true;
}

bool GeGraphExecutor::RunGraphRefModeForKernel(const KernelGraphPtr &graph, const AnfNodeWeakPtr &node,
                                               const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs, void *stream) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(stream);
  auto graph_name = GetGraphName(graph);
  MS_LOG(INFO) << "Run graph begin, inputs size is: " << inputs.size() << ", " << graph_name;
  bool is_dynamic_shape = graph->is_dynamic_shape();
  std::vector<GeTensor> ge_inputs = CreateInputGeTensorList(inputs, graph);
  std::vector<GeTensor> ge_outputs = CreateOutputGeTensorList(outputs, graph);
  auto ret = RunGraphRefModeInnner(graph, ge_inputs, &ge_outputs, stream);
  if (!ret) {
    return ret;
  }
  if (is_dynamic_shape) {
    MS_LOG(INFO) << "Update outputs for graph: " << graph_name;
    SetDynamicOutputsForKernel(ge_outputs, outputs, node, ResManager());
  }
  return true;
}

std::vector<GeTensor> GeGraphExecutor::CreateInputGeTensorList(const std::vector<KernelTensor *> &tensors,
                                                               const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<GeTensor> ge_inputs;
  auto iter = input_datas_.find(graph.get());
  if (iter == input_datas_.end()) {
    return ge_inputs;
  }
  bool is_dynamic_shape = graph->is_dynamic_shape();
  const auto &input_datas = iter->second.ge_inputs;
  ge_inputs = input_datas;
  size_t index = 0;
  for (const auto &tensor : tensors) {
    // input monad is nullptr
    if (tensor == nullptr) {
      continue;
    }
    // remove monad
    std::vector<TypeId> monad_type_id = {TypeId::kObjectTypeMonad, TypeId::kObjectTypeUMonad,
                                         TypeId::kObjectTypeIOMonad};
    if (std::any_of(monad_type_id.begin(), monad_type_id.end(),
                    [&tensor](const TypeId type_id) { return type_id == tensor->dtype_id(); })) {
      continue;
    }

    if (index >= ge_inputs.size()) {
      MS_LOG(EXCEPTION) << "The index " << index << " >= ge_inputs size: " << ge_inputs.size();
    }

    if (is_dynamic_shape) {
      auto ge_tensor_desc =
        transform::TransformUtil::GetGeTensorDesc(tensor->GetShapeVector(), tensor->dtype_id(), kOpFormat_DEFAULT);
      MS_EXCEPTION_IF_NULL(ge_tensor_desc);
      ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
      (void)ge_inputs[index].SetTensorDesc(*ge_tensor_desc);
    }

    if (tensor->device_ptr() == nullptr) {
      // alloc static memory for unused inputs
      // error in ge when set nullptr into ge tensor
      GEMemoryAllocator::AllocUnuseInput(graph, tensor, ResManager());
    }

    if (tensor->device_ptr() != ge_inputs[index].GetData() || tensor->size() != ge_inputs[index].GetSize()) {
      if (ge_inputs[index].SetData(reinterpret_cast<uint8_t *>(tensor->device_ptr()), tensor->size(), [](void *) {}) !=
          ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "Set ge tensor addr failed! addr size is " << tensor->size();
      }
    }

    MS_LOG(DEBUG) << "Update GeTensor, shape: " << tensor->GetShapeVector() << ", tensor_size: " << tensor->size()
                  << ", dtype: " << tensor->dtype_id();
    ++index;
  }
  return ge_inputs;
}

std::vector<GeTensor> GeGraphExecutor::CreateOutputGeTensorList(const std::vector<KernelTensor *> &tensors,
                                                                const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<GeTensor> ge_outputs;
  auto iter = output_datas_.find(graph.get());
  if (iter == output_datas_.end()) {
    return ge_outputs;
  }
  bool is_dynamic_shape = graph->is_dynamic_shape();
  const auto &output_datas = iter->second.ge_outputs;
  ge_outputs = output_datas;

  size_t index = 0;
  for (const auto &tensor : tensors) {
    // input monad is nullptr
    if (tensor == nullptr) {
      continue;
    }
    // remove monad
    std::vector<TypeId> monad_type_id = {TypeId::kObjectTypeMonad, TypeId::kObjectTypeUMonad,
                                         TypeId::kObjectTypeIOMonad};
    if (std::any_of(monad_type_id.begin(), monad_type_id.end(),
                    [&tensor](const TypeId type_id) { return type_id == tensor->dtype_id(); })) {
      continue;
    }
    if (index >= ge_outputs.size()) {
      MS_LOG(EXCEPTION) << "The index " << index << " >= ge_outputs size: " << ge_outputs.size();
    }

    if (is_dynamic_shape) {
      ge_outputs[index].SetData(nullptr, 0U, [](void *) {});
      continue;
    }

    if (tensor->device_ptr() != ge_outputs[index].GetData() || tensor->size() != ge_outputs[index].GetSize()) {
      if (ge_outputs[index].SetData(reinterpret_cast<uint8_t *>(tensor->device_ptr()), tensor->size(), [](void *) {}) !=
          ::ge::GRAPH_SUCCESS) {
        MS_LOG(EXCEPTION) << "Set ge tensor addr failed! addr size is " << tensor->size();
      }
    }
    MS_LOG(DEBUG) << "Update GeTensor, shape: " << tensor->GetShapeVector() << ", tensor_size: " << tensor->size()
                  << ", dtype: " << tensor->dtype_id();
    ++index;
  }
  return ge_outputs;
}

void GeGraphExecutor::DoAsyncCkpt(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto env = common::GetEnv("MS_ENABLE_CKPT_D2H_ASYNC");
  if (env == "1" && ms_context->get_param<bool>(MS_CTX_NEED_CKPT) && kg != nullptr) {
    auto cur_step = ms_context->get_param<int>(MS_CTX_CUR_STEP_NUM);
    auto save_steps = ms_context->get_param<int>(MS_CTX_SAVE_CKPT_STEPS);
    auto last_triggered_step = ms_context->get_param<int>(MS_CTX_LAST_TRIGGERED_STEP);
    MS_LOG(DEBUG) << "cur_step:" << cur_step << ", save_steps: " << save_steps
                  << ", last_triggered_step:" << last_triggered_step;
    if (cur_step >= (last_triggered_step + save_steps)) {
      if (SkipOrResetCopyAction()) {
        MS_LOG(INFO) << "Enable async d2h copy";
        SavePrevStepWeight(kg->GetRootWeights(), ResManager()->GetCopyDataStream());
      }
      if (kg->has_attr(kIsRefGraph) && GetValue<bool>(kg->get_attr(kIsRefGraph)) && SkipOrResetSyncAction()) {
        MS_LOG(INFO) << "Ref graph sync once action";
        SyncCopyStream(ResManager()->GetCopyDataStream());
      }
    }
  }
}

bool GeGraphExecutor::IsNeedNotifyTTP(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  if (mindio::MindIOAdapter::GetInstance()->IsEnable() && kg != nullptr && kg->has_attr(kIsRefGraph) &&
      GetValue<bool>(kg->get_attr(kIsRefGraph))) {
    return true;
  }
  return false;
}

bool GeGraphExecutor::RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                               std::vector<tensor::Tensor> *outputs,
                               const std::map<string, string> & /* compile_options */) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_name = GetGraphName(graph);
  uint64_t start_time = profiler::GetClockSyscnt();
  DoAsyncCkpt(graph);
  if (IsEnableRefMode()) {
    if (!RunGraphRefMode(graph, inputs)) {
      (void)profiler::CollectHostInfo("Ascend", "CompileGraph", "GeRunGraph_" + graph_name, start_time,
                                      profiler::GetClockSyscnt(), 1);
      return false;
    }
  } else {
    MS_LOG(INFO) << "GE run graph start, graph: " << graph_name << ".";
    (void)ResManager()->BindDeviceToCurrentThread(false);
    // copy input from device to host
    const auto &cur_inputs = graph->get_inputs();
    std::vector<tensor::TensorPtr> input_tensors;
    for (const auto &input : cur_inputs) {
      MS_EXCEPTION_IF_NULL(input);
      auto output_addr = AnfAlgo::GetMutableOutputAddr(input, 0);
      auto shapes = trans::GetRuntimePaddingShape(input, 0);
      auto host_type = common::AnfAlgo::GetOutputInferDataType(input, 0);
      auto tensor = std::make_shared<tensor::Tensor>(host_type, shapes);
      MS_EXCEPTION_IF_NULL(tensor);
      tensor->set_device_address(output_addr, false);
      tensor->data_sync();
      (void)input_tensors.emplace_back(std::move(tensor));
    }
    auto ge_inputs = transform::ConvertInputTensors(input_tensors, kOpFormat_NCHW);

    // call ge rungraph
    KernelGraphPtr kg = std::dynamic_pointer_cast<session::KernelGraph>(graph);
    if (kg != nullptr) {
      graph_name = kg->GetFuncGraph()->ToString();
    }
    transform::RunOptions run_options;
    run_options.name = graph_name;
    auto graph_runner = transform::GetGraphRunner();
    if (graph_runner == nullptr) {
      MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
    }

    AnfNodePtr output = graph->get_return()->input(1);
    MS_EXCEPTION_IF_NULL(output);
    std::vector<TypeId> me_types;
    auto output_c = output->cast<CNodePtr>()->abstract();
    // get output node data types
    GetMeRetDataType(output_c, &me_types);
    std::vector<transform::GeTensorPtr> ge_outputs;
    {
      // Release GIL before calling into (potentially long-running) C++ code
      GilReleaseWithCheck gil_release;
      MS_LOG(DEBUG) << "Run graph begin, inputs size is: " << inputs.size();
      transform::Status ret = transform::RunGraphAsync(graph_runner, run_options, ge_inputs, &ge_outputs);
      MS_LOG(DEBUG) << "Run graph finish, outputs size is: " << ge_outputs.size();
      if (ret == transform::Status::NOT_FOUND) {
        MS_LOG(WARNING) << "The Graph[" << graph_name << "] is not found, skip run it.";
        (void)profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, start_time,
                                        profiler::GetClockSyscnt(), 1);
        return true;
      } else if (ret != transform::Status::SUCCESS) {
        MS_LOG(EXCEPTION) << "Exec graph failed";
      }
    }
    auto no_output = common::AnfAlgo::IsNoOuputNode(output);
    if (!no_output) {
      if (me_types.size() != ge_outputs.size()) {
        MS_LOG(EXCEPTION) << "Invalid output size, me_type's size " << me_types.size() << " tensor size "
                          << ge_outputs.size();
      }
      // copy output from host to device
      auto graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
      if (graph_outputs.size() != ge_outputs.size()) {
        MS_LOG(EXCEPTION) << "Invalid output size, graph's size " << graph_outputs.size() << " tensor size "
                          << ge_outputs.size();
      }
      SetOutputs(graph_outputs, ge_outputs, me_types);
    }
  }
  if (graph->has_flag(transform::kGraphFlagHasGetNext)) {
    MS_LOG(DEBUG) << "Reset ConfigManager, graph: " << graph_name;
    ConfigManager::GetInstance().ResetConfig();
    ConfigManager::GetInstance().ResetIterNum();
  }
  (void)profiler::CollectHostInfo("Ascend", "RunGraph", "GeRunGraph_" + graph_name, start_time,
                                  profiler::GetClockSyscnt(), 1);
  MS_LOG(INFO) << "GE run graph end.";
  return true;
}

FuncGraphPtr GeGraphExecutor::BuildDFGraph(const FuncGraphPtr &anf_graph,
                                           const transform::TensorOrderMap &init_inputs_map, bool export_air) {
  MS_EXCEPTION_IF_NULL(anf_graph);
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_before_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_before_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif

  if (!AddDFGraph(anf_graph, init_inputs_map, export_air)) {
    MS_LOG(ERROR) << "GenConvertor failed";
    return nullptr;
  }

#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    if (context->CanDump(kFully)) {
      draw::Draw("anf_graph_after_build_df_graph.dot", anf_graph);  // for debug
    }
    DumpIR("anf_graph_after_build_df_graph.ir", anf_graph, true, kWholeStack);
  }
#endif

  if (export_air) {
    // export air can't use session->AddGraph, it will cause atc error.
    return anf_graph;
  }

  return anf_graph;
}

static inline std::string GetNodeInfo(const AnfNodeWeakPtr &node) {
  auto input_node = node.lock();
  MS_EXCEPTION_IF_NULL(input_node);
  return input_node->DebugString();
}

std::vector<GeTensor> GeGraphExecutor::GenerateInputGeTensor(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_inputs;
  auto iter = input_datas_.find(kernel_graph.get());
  if (iter == input_datas_.end()) {
    return ge_inputs;
  }
  bool is_dynamic_shape = kernel_graph->is_dynamic_shape();
  const auto &input_datas = iter->second.ge_inputs;
  ge_inputs = input_datas;
  for (size_t i = 0; i < iter->second.device_addrs.size(); ++i) {
    auto output_addr = iter->second.device_addrs[i];
    MS_EXCEPTION_IF_NULL(output_addr);
    if (is_dynamic_shape) {
      auto ge_tensor_desc = transform::TransformUtil::GetGeTensorDesc(output_addr->kernel_tensor()->GetShapeVector(),
                                                                      output_addr->type_id(), output_addr->format());
      MS_EXCEPTION_IF_NULL(ge_tensor_desc);
      ge_tensor_desc->SetPlacement(::ge::kPlacementDevice);
      (void)ge_inputs[i].SetTensorDesc(*ge_tensor_desc);
    }
    auto node_output_addr = output_addr->GetMutablePtr();
    if (node_output_addr == nullptr) {
      auto input_node = iter->second.need_update_input[i].first.lock();
      MS_EXCEPTION_IF_NULL(input_node);
      // alloc static memory for unused inputs
      // error in ge when set nullptr into ge tensor
      GEMemoryAllocator::AllocUnuseInput(kernel_graph, input_node, output_addr, ResManager());
    }
    MS_LOG(INFO) << "[ZeroCopy] For Graph " << kernel_graph->ToString() << ", update input "
                 << GetNodeInfo(iter->second.need_update_input[i].first) << " address to "
                 << output_addr->GetMutablePtr() << ", shape:" << output_addr->kernel_tensor()->GetShapeVector()
                 << ", type: " << TypeIdToString(output_addr->type_id()) << ", format: " << output_addr->format()
                 << ", memory size: " << output_addr->GetSize();
    if (node_output_addr != ge_inputs[i].GetData() || output_addr->GetSize() != ge_inputs[i].GetSize()) {
      (void)ge_inputs[i].SetData(static_cast<uint8_t *>(node_output_addr), output_addr->GetSize(), [](void *) {});
    }
  }
  return ge_inputs;
}

std::vector<GeTensor> GeGraphExecutor::GenerateOutputGeTensor(const KernelGraphPtr &kernel_graph) const {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<GeTensor> ge_outputs;
  auto iter = output_datas_.find(kernel_graph.get());
  if (iter == output_datas_.end()) {
    return ge_outputs;
  }
  const auto &output_datas = iter->second.ge_outputs;
  ge_outputs = output_datas;

  bool is_dynamic_shape = kernel_graph->is_dynamic_shape();
  for (size_t idx = 0; idx < iter->second.device_addrs.size(); ++idx) {
    if (is_dynamic_shape) {
      ge_outputs[idx].SetData(nullptr, 0U, [](void *) {});
      continue;
    }
    auto output_node = iter->second.graph_outputs[idx].first.lock();
    auto index = iter->second.graph_outputs[idx].second;
    MS_EXCEPTION_IF_NULL(output_node);
    MS_EXCEPTION_IF_CHECK_FAIL(
      idx < ge_outputs.size(),
      "GenerateOutputGeTensor idx is greater equal than ge_outputs size, idx: " + std::to_string(idx) +
        ", ge outputs size: " + std::to_string(ge_outputs.size()) + ", kernel graph: " + kernel_graph->ToString());
    auto output_device_addr = iter->second.device_addrs[idx];
    auto node_output_device_addr = output_device_addr->GetMutablePtr();
    MS_LOG(INFO) << "Output addr " << node_output_device_addr;
    if (node_output_device_addr == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, output_node)
        << "Output " << output_node->fullname_with_scope() << ", index: " << index
        << " address is nullptr, kernel graph: " << kernel_graph->ToString()
        << ", addr memory size: " << output_device_addr->GetSize()
        << "\n Maybe memory is not enough, memory statistics:" << AscendMemAdapter::GetInstance()->DevMemStatistics();
    }
    MS_LOG(INFO) << "[ZeroCopy] For Graph " << kernel_graph->ToString() << ", update output "
                 << output_node->DebugString() << " out_idx " << index << " address to "
                 << output_device_addr->GetMutablePtr()
                 << ", shape:" << output_device_addr->kernel_tensor()->GetShapeVector()
                 << ", type: " << TypeIdToString(output_device_addr->type_id())
                 << ", format: " << output_device_addr->format() << ", memory size: " << output_device_addr->GetSize();
    if (node_output_device_addr != ge_outputs[idx].GetData() ||
        output_device_addr->GetSize() != ge_outputs[idx].GetSize()) {
      (void)ge_outputs[idx].SetData(reinterpret_cast<uint8_t *>(node_output_device_addr), output_device_addr->GetSize(),
                                    [](void *) {});
    }
  }
  return ge_outputs;
}

void GeGraphExecutor::RunInitGraph(const std::string &graph_name) {
  transform::RunOptions run_options;
  run_options.name = "init_subgraph." + graph_name;
  if (transform::GetGraphByName(run_options.name) == nullptr) {
    MS_LOG(INFO) << "Can not find " << run_options.name << " sub graph, don't need data init subgraph in INFER mode.";
    return;
  }
  auto graph_runner = transform::GetGraphRunner();
  if (graph_runner == nullptr) {
    MS_LOG(EXCEPTION) << "Can not found GraphRunner.";
  }

  auto cur_sink_size = CurGraphSinkSize(graph_name);
  if (pre_sink_size_ == cur_sink_size) {
    return;
  }
  pre_sink_size_ = cur_sink_size;
  MS_LOG(INFO) << "Start run init graph: " << run_options.name << ", sink size:" << cur_sink_size;
  std::vector<transform::GeTensorPtr> ge_outputs;
  std::vector<transform::GeTensorPtr> ge_tensors;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    GilReleaseWithCheck gil_release;
    transform::Status ret = transform::RunGraph(graph_runner, run_options, ge_tensors, &ge_outputs);
    if (ret != transform::Status::SUCCESS) {
      MS_LOG(EXCEPTION) << "Exec " << run_options.name << " graph failed.";
    }
    MS_LOG(INFO) << "Exec " << run_options.name << " graph success.";
  }
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
