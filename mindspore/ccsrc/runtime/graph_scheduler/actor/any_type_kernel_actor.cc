/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/any_type_kernel_actor.h"
#include <set>
#include <functional>
#include "include/common/debug/anf_ir_dump.h"
#include "plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "include/common/fallback.h"
#include "include/common/utils/stub_tensor.h"
#include "include/backend/py_execute_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace runtime {
namespace {
using AddressPtr = kernel::AddressPtr;
using PyExecuteOutputUserData = kernel::PyExecuteOutputUserData;
}  // namespace

std::mutex AnyTypeKernelActor::instance_lock_;

AnyTypeKernelActor::AnyTypeKernelActor(const std::string &name, const KernelGraphPtr &graph,
                                       const DeviceContext *device_context, const AID &memory_manager_aid,
                                       const AID *debug_aid, const AID *recorder_aid, KernelTransformType type)
    : SuperKernelActor(name, graph, "", device_context, memory_manager_aid, debug_aid, recorder_aid, type) {
  enable_kbk_sub_graph_execute_ = true;
}

void AnyTypeKernelActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter == input_op_datas_.end()) {
    return;
  }
  for (auto &input_data : data_iter->second) {
    MS_EXCEPTION_IF_NULL(input_data);
    MS_EXCEPTION_IF_NULL(input_data->data_);
    size_t index = IntToSize(input_data->index_);
    if (index >= input_device_tensors_.size()) {
      std::string error_info = "Invalid graph input index:" + std::to_string(index) +
                               " total:" + std::to_string(input_device_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[index] = input_data->data_;
  }

  for (auto &device_tensor_store_key : extern_device_tensor_store_keys_) {
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    "Invalid device context for any type actor:" + GetAID().Name());
    }
    auto device_tensor = DeviceTensorStore::GetInstance()
                           .Fetch(device_tensor_store_key.second.get(), device_contexts_[0]->GetDeviceType())
                           .get();
    if (device_tensor == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, device_tensor_store_key.second)
        << "Failed get device tensor for node:" << device_tensor_store_key.second->DebugString()
        << " index:" << device_tensor_store_key.first << " device type:" << device_contexts_[0]->GetDeviceType();
      continue;
    }
    if (device_tensor_store_key.first >= input_device_tensors_.size()) {
      std::string error_info =
        "Invalid graph input device tensor store index:" + std::to_string(device_tensor_store_key.first) +
        " total:" + std::to_string(input_device_tensors_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[device_tensor_store_key.first] = device_tensor;
  }
}

bool AnyTypeKernelActor::CheckGraphOutputRunningCondition(const OpContext<DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_LOG(DEBUG) << "graph output data num:" << graph_output_data_num_[current_data_type_]
                << " control num:" << graph_output_control_num_[current_data_type_];
  if (graph_output_data_num_[current_data_type_] != 0) {
    const auto &data_iter = graph_output_op_data_.find(context->sequential_num_);
    if (data_iter == graph_output_op_data_.end()) {
      return false;
    }
    if (data_iter->second.size() < graph_output_data_num_[current_data_type_]) {
      return false;
    } else if (data_iter->second.size() > graph_output_data_num_[current_data_type_]) {
      MS_LOG(ERROR) << "Invalid graph output data num:" << data_iter->second.size()
                    << " need:" << graph_output_data_num_[current_data_type_] << " for actor:" << GetAID()
                    << ", sequential num:" << context->sequential_num_;
      return false;
    }
  }

  if (graph_output_control_num_[current_data_type_] != 0) {
    const auto &control_iter = graph_output_op_control_.find(context->sequential_num_);
    if (control_iter == graph_output_op_control_.end()) {
      return false;
    }
    if (control_iter->second.size() < graph_output_control_num_[current_data_type_]) {
      return false;
    } else if (control_iter->second.size() > graph_output_control_num_[current_data_type_]) {
      MS_LOG(ERROR) << "Invalid input control num:" << control_iter->second.size()
                    << " need:" << graph_output_control_num_[current_data_type_] << " for actor:" << GetAID()
                    << ", sequential num:" << context->sequential_num_;
      return false;
    }
  }
  return true;
}
namespace {
GraphSegmentPtr BuildSegmentByGraph(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> nodes;
  std::vector<AnfNodePtr> all_nodes = TopoSort(graph->get_return());
  for (const auto &node : all_nodes) {
    if (node == nullptr || (!node->isa<CNode>()) || common::AnfAlgo::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    MS_LOG(DEBUG) << "build new segment node:" << node->DebugString();
    nodes.emplace_back(node);
  }
  return std::make_shared<GraphSegment>(nodes, false);
}

std::string GenerateIDForGraph(const std::vector<DeviceTensor *> &device_tensors, const std::vector<size_t> &indexes) {
  std::string id;
  auto get_shape_and_type_string = [&id](const ShapeVector &shape_vector, TypeId type_id) {
    id += "shape_";
    (void)std::for_each(shape_vector.begin(), shape_vector.end(), [&id](int64_t shape) {
      id += std::to_string(shape);
      id += "_";
    });
    id = id + "type_" + std::to_string(type_id) + "_";
  };
  for (const auto &index : indexes) {
    if (index >= device_tensors.size()) {
      MS_LOG(EXCEPTION) << "Invalid parameter index:" << index << " for device tensor num:" << device_tensors.size();
    }
    id = id + "index_" + std::to_string(index) + "_";
    const auto &device_tensor = device_tensors[index];
    if (device_tensor == nullptr || device_tensor->kernel_tensor() == nullptr) {
      MS_LOG(EXCEPTION) << "Empty device tensor index:" << index;
    }
    if (device_tensor->user_data() == nullptr) {
      get_shape_and_type_string(device_tensor->host_shape(), device_tensor->type_id());
      continue;
    }

    const auto &user_data_obj =
      device_tensor->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
    if (user_data_obj == nullptr) {
      MS_LOG(ERROR) << "Failed to get user data from input index:" << index
                    << " device tensor:" << device_tensor->PrintInfo();
      return "FAILED";
    }
    const auto &obj = user_data_obj->obj;
    py::gil_scoped_acquire gil_acquire;
    const auto &abstract = pyexecute::GenerateAbstractFromPyObject(obj);
    MS_EXCEPTION_IF_NULL(abstract);
    if (abstract->isa<abstract::AbstractSequence>()) {
      auto sequence_abs = abstract->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(sequence_abs);
      id = id + "Tuple_" + std::to_string(sequence_abs->size()) + "_";
    } else if (abstract->isa<abstract::AbstractScalar>()) {
      id = id + "Scalar_";
    } else if (abstract->isa<abstract::AbstractTensor>()) {
      id = id + "Tensor_";
    }
    get_shape_and_type_string(device_tensor->host_shape(), device_tensor->type_id());
  }
  return id;
}

void InferParameterAbstractForModelGraph(const KernelGraphPtr &graph, const std::vector<DeviceTensor *> &device_tensors,
                                         const std::vector<size_t> &indexes) {
  MS_EXCEPTION_IF_NULL(graph);
  for (size_t index : indexes) {
    if (index >= device_tensors.size() || index >= graph->input_nodes().size()) {
      MS_LOG(EXCEPTION) << "Invalid index:" << index << " for input device tensor size:" << device_tensors.size()
                        << " for graph:" << graph->ToString();
    }
    const auto &device_tensor = device_tensors[index];
    MS_EXCEPTION_IF_NULL(device_tensor);
    MS_EXCEPTION_IF_NULL(device_tensor->kernel_tensor());
    auto input_node = graph->input_nodes()[index];
    MS_EXCEPTION_IF_NULL(input_node);
    abstract::AbstractBasePtr abstract;
    if (device_tensor->user_data() != nullptr &&
        device_tensor->user_data()->has(kernel::PyExecuteOutputUserData::key)) {
      MS_LOG(DEBUG) << "User data:" << device_tensor->user_data() << " in device address:" << device_tensor
                    << " for input:" << input_node->DebugString();
      const auto &user_data_obj =
        device_tensor->user_data()->get<kernel::PyExecuteOutputUserData>(kernel::PyExecuteOutputUserData::key);
      MS_EXCEPTION_IF_NULL(user_data_obj);
      const auto &obj = user_data_obj->obj;
      py::gil_scoped_acquire gil_acquire;
      abstract = pyexecute::GenerateAbstractFromPyObject(obj);
    } else {
      TypePtr type = device_tensor->kernel_tensor()->GetType();
      BaseShapePtr shape = device_tensor->kernel_tensor()->GetShape();
      MS_EXCEPTION_IF_NULL(type);
      MS_EXCEPTION_IF_NULL(shape);
      if (type->isa<Tuple>() && shape->isa<abstract::TupleShape>()) {
        const auto &tuple_type = type->cast<TuplePtr>();
        const auto &tuple_shape = shape->cast<abstract::TupleShapePtr>();
        MS_EXCEPTION_IF_NULL(tuple_shape);
        MS_EXCEPTION_IF_NULL(tuple_type);
        if (tuple_type->dynamic_len() && tuple_type->dynamic_element_type() != nullptr && tuple_shape->size() > 0) {
          TypePtrList typle_list(tuple_shape->size(), tuple_type->dynamic_element_type());
          type = std::make_shared<Tuple>(typle_list);
          MS_LOG(DEBUG) << "Replace type from:" << device_tensor->kernel_tensor()->GetType()
                        << " to:" << type->ToString();
        }
      }
      abstract = abstract::MakeAbstract(shape, type);
    }
    MS_EXCEPTION_IF_NULL(abstract);
    MS_LOG(DEBUG) << "Infer parameter by abstract:" << abstract->ToString();
    if (!abstract->isa<abstract::AbstractSequence>()) {
      MS_LOG(DEBUG) << "Set abstract:" << abstract->ToString() << " for input node:" << input_node->DebugString()
                    << " device tensor:" << device_tensor << " type id:" << device_tensor->type_id();
      input_node->set_abstract(abstract);
      continue;
    }
    MS_LOG(DEBUG) << "Sequence abstract:" << abstract->ToString();
    auto new_abstract = abstract->Clone();
    MS_EXCEPTION_IF_NULL(new_abstract);
    auto seq_abstract = new_abstract->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(seq_abstract);
    seq_abstract->set_dynamic_len(true);
    // Dynamic len element is used to check if the sequence is dynamic len.
    if (!seq_abstract->elements().empty() && seq_abstract->elements()[0] != nullptr) {
      seq_abstract->set_dynamic_len_element_abs(seq_abstract->elements()[0]->Clone());
    }
    MS_LOG(DEBUG) << "Set abstract:" << seq_abstract->ToString() << " for input node:" << input_node->DebugString()
                  << device_tensor << " type id:" << device_tensor->type_id();
    input_node->set_abstract(seq_abstract);
  }
}

TypeId GetElementType(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  TypePtr type = nullptr;
  if (abstract->isa<abstract::AbstractScalar>()) {
    type = abstract->BuildType();
  } else if (abstract->isa<abstract::AbstractTensor>()) {
    const auto &tensor_abs = abstract->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor_abs);
    MS_EXCEPTION_IF_NULL(tensor_abs->element());
    type = tensor_abs->element()->BuildType();
  } else if (abstract->isa<abstract::AbstractSequence>()) {
    const auto &sequence_abs = abstract->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(sequence_abs);
    if (sequence_abs->dynamic_len() || sequence_abs->elements().empty() || sequence_abs->elements()[0] == nullptr) {
      MS_LOG(INFO) << "Invalid abstract:" << abstract->ToString();
      return TypeId::kNumberTypeInt64;
    }
    return GetElementType(sequence_abs->elements()[0]);
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract:" << abstract->ToString();
  }
  MS_EXCEPTION_IF_NULL(type);
  return type->type_id();
}
}  // namespace

void AnyTypeKernelActor::UpdataDynamicShapeParameterForGraphInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (graph_input_backend_parameters_.find(current_data_type_) == graph_input_backend_parameters_.end()) {
    return;
  }
  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    if (input_device_tensors_[i] != nullptr && input_device_tensors_[i]->user_data() != nullptr) {
      MS_EXCEPTION_IF_NULL(input_device_tensors_[i]->kernel_tensor());
      const auto &user_data_obj = input_device_tensors_[i]->user_data()->get<kernel::PyExecuteOutputUserData>(
        kernel::PyExecuteOutputUserData::key);
      MS_EXCEPTION_IF_NULL(user_data_obj);
      const auto &obj = user_data_obj->obj;
      auto abstract = pyexecute::GenerateAbstractFromPyObject(obj);
      MS_EXCEPTION_IF_NULL(abstract);
      MS_EXCEPTION_IF_NULL(abstract->BuildType());
      MS_EXCEPTION_IF_NULL(abstract->BuildShape());
      MS_LOG(DEBUG) << "actor:" << GetAID() << " set shape by abstract:" << abstract->ToString()
                    << " shape:" << abstract->BuildShape()->ToString() << " type:" << abstract->BuildType()->ToString()
                    << " for device address:" << input_device_tensors_[i];
      input_device_tensors_[i]->kernel_tensor()->SetType(abstract->BuildType());
      input_device_tensors_[i]->kernel_tensor()->SetShape(abstract->BuildShape());
      MS_LOG(DEBUG) << "Infer abstract:" << abstract->ToString();
    }
  }
}

namespace {
void ClearAttrForGraph(const KernelGraphPtr &graph, const std::string &attr_name) {
  MS_EXCEPTION_IF_NULL(graph);
  for (const auto &node_pair : graph->front_backend_anf_map()) {
    MS_EXCEPTION_IF_NULL(node_pair.second);
    if (!node_pair.second->isa<CNode>()) {
      continue;
    }
    MS_LOG(DEBUG) << "Check for node:" << node_pair.second->DebugString() << " attr name:" << attr_name;
    const auto &cnode = node_pair.second->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (common::AnfAlgo::HasNodeAttr(attr_name, cnode)) {
      MS_LOG(DEBUG) << "Erase flag for node:" << node_pair.second->DebugString() << " attr name:" << attr_name;
      common::AnfAlgo::EraseNodeAttr(attr_name, cnode);
    }
  }
}

void PrepareValueNode(const AnfNodePtr &value_node, DeviceTensor *device_tensor) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(device_tensor);
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_tensor->device_name(), device_tensor->device_id()});
  MS_EXCEPTION_IF_NULL(device_context);
  if (device_tensor->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(device_tensor)) {
      MS_LOG(EXCEPTION) << "Failed to allocate memory for device tensor store:" << device_tensor;
    }
    MS_LOG(DEBUG) << "Device address:" << device_tensor << " allocate ptr:" << device_tensor->GetPtr()
                  << " for value node:" << value_node->DebugString();
  } else {
    MS_LOG(DEBUG) << "Device address:" << device_tensor << " already has ptr:" << device_tensor->GetPtr()
                  << " for value node:" << value_node->DebugString();
  }

  const auto &kernel_tensor = device_tensor->kernel_tensor();
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  if (!device_tensor->SyncHostToDevice(kernel_tensor->GetShapeVector(), kernel_tensor->size(),
                                       kernel_tensor->dtype_id(), kernel_tensor->GetValuePtr())) {
    MS_LOG_WITH_NODE(EXCEPTION, value_node) << "Failed to sync data for value node:" << value_node->DebugString();
  }

  MS_LOG(DEBUG) << "Device address:" << device_tensor << " ptr:" << device_tensor->GetPtr()
                << " for value node:" << value_node->DebugString();
}

void PersisitValueNode(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(DEBUG) << "Value node size:" << graph->graph_value_nodes().size() << " for graph:" << graph->ToString();
  for (auto &value_node : graph->graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    if (!AnfAlgo::OutputAddrExist(value_node, 0)) {
      MS_LOG(INFO) << "The device address is not exist: " << value_node->ToString()
                   << " for graph:" << graph->ToString();
      continue;
    }
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(value_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(value_node, *graph);
    MS_EXCEPTION_IF_NULL(front_node);
    device_tensor->SetNodeIndex(value_node, 0);
    DeviceTensorStore::GetInstance().Insert(const_cast<AnfNode *>(front_node.get()), device_tensor);
    PrepareValueNode(value_node, device_tensor.get());
    MS_LOG(DEBUG) << "Add device tensor store:" << device_tensor << " node:" << front_node->DebugString()
                  << " graph:" << graph->ToString();

    // If the device tensor store of this device type is not exist, then create the new device tensor of this type.
    if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), device_context->GetDeviceType()) == nullptr) {
      MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->fullname_with_scope()
                   << ", type:" << device_context->GetDeviceType() << " dtype:" << device_tensor->type_id()
                   << " current device address:" << device_tensor << " in value node:" << value_node->DebugString();

      const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
        {value_node, 0}, nullptr, device_tensor->GetSize(), device_tensor->format(), device_tensor->type_id(),
        device_tensor->host_shape(), device_context->device_context_key().device_name_,
        device_context->device_context_key().device_id_);
      kernel_tensor->set_stream_id(device_tensor->stream_id());
      auto other_type_device_tensor = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
      MS_EXCEPTION_IF_NULL(other_type_device_tensor);
      other_type_device_tensor->SetNodeIndex(value_node, 0);
      other_type_device_tensor->set_from_persistent_mem(true);
      MS_LOG(DEBUG) << "Create device tensor:" << other_type_device_tensor
                    << " type:" << other_type_device_tensor->type_id();
      DeviceTensorStore::GetInstance().Insert(const_cast<AnfNode *>(front_node.get()), other_type_device_tensor);
      MS_LOG(DEBUG) << "Add device tensor store:" << other_type_device_tensor << " node:" << front_node->DebugString()
                    << " graph:" << graph->ToString();
      PrepareValueNode(value_node, device_tensor.get());
    }
  }
  for (const auto &kernel : graph->execution_order()) {
    MS_EXCEPTION_IF_NULL(kernel);
    const auto &real_device_context = device::FetchRealDeviceContext(kernel, device_context);
    if (real_device_context == device_context) {
      continue;
    }
    MS_LOG(DEBUG) << "Handle device tensor store for backoff kernel:" << kernel->fullname_with_scope()
                  << " in graph:" << graph->ToString();
    MS_EXCEPTION_IF_NULL(real_device_context);
    for (size_t i = 0; i < common::AnfAlgo::GetInputNum(kernel); ++i) {
      auto input_node = common::AnfAlgo::GetInputNode(kernel, i);
      if (input_node == nullptr || !input_node->isa<ValueNode>()) {
        continue;
      }
      const auto &front_node = AnfAlgo::FetchFrontNodeByBackendNode(input_node, *graph);
      MS_EXCEPTION_IF_NULL(front_node);
      if (DeviceTensorStore::GetInstance().Fetch(front_node.get(), real_device_context->GetDeviceType()) != nullptr ||
          !AnfAlgo::OutputAddrExist(input_node, 0, false)) {
        MS_LOG(DEBUG) << "Failed to get device tensor in value node:" << input_node->DebugString()
                      << " has device address:" << AnfAlgo::OutputAddrExist(input_node, 0, false)
                      << " real device type:" << real_device_context->GetDeviceType() << " fetch device address:"
                      << DeviceTensorStore::GetInstance().Fetch(front_node.get(), real_device_context->GetDeviceType());
        continue;
      }
      auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
      MS_EXCEPTION_IF_NULL(device_tensor);
      MS_LOG(INFO) << "Fetch no device tensor store by:" << front_node->fullname_with_scope()
                   << ", type:" << real_device_context->GetDeviceType()
                   << " node device tensor:" << device_tensor->PrintInfo();
      const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
        {input_node, 0}, nullptr, device_tensor->GetSize(), device_tensor->format(), device_tensor->type_id(),
        device_tensor->host_shape(), real_device_context->device_context_key().device_name_,
        real_device_context->device_context_key().device_id_);
      kernel_tensor->set_stream_id(device_tensor->stream_id());
      auto other_type_device_tensor = real_device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
      MS_EXCEPTION_IF_NULL(other_type_device_tensor);
      other_type_device_tensor->SetNodeIndex(input_node, 0);
      other_type_device_tensor->set_from_persistent_mem(true);
      MS_LOG(DEBUG) << "Create device tensor:" << other_type_device_tensor
                    << " type:" << other_type_device_tensor->type_id()
                    << " device type:" << real_device_context->device_context_key().ToString();
      DeviceTensorStore::GetInstance().Insert(const_cast<AnfNode *>(front_node.get()), other_type_device_tensor);
      MS_LOG(DEBUG) << "Add device tensor store:" << other_type_device_tensor << " node:" << front_node->DebugString()
                    << " graph:" << graph->ToString();
      PrepareValueNode(input_node, device_tensor.get());
    }
  }
}
}  // namespace

KernelGraphPtr AnyTypeKernelActor::CompileRealKernelGraph(OpContext<DeviceTensor> *const context) {
  try {
    std::lock_guard<std::mutex> lock(instance_lock_);
    InferParameterAbstractForModelGraph(model_graph_, input_device_tensors_, any_type_parameter_indexes_);
    ClearAttrForGraph(model_graph_, kAttrInputIsDynamicShape);
    ClearAttrForGraph(model_graph_, kAttrOutputIsDynamicShape);
    model_graph_->InferType();
    const auto &return_node = model_graph_->get_return();
    MS_EXCEPTION_IF_NULL(return_node);
    if (!return_node->isa<CNode>() || return_node->cast<CNodePtr>()->size() <= 1) {
      MS_LOG_WITH_NODE(EXCEPTION, return_node)
        << "Invalid return node:" << return_node->DebugString() << " for graph:" << model_graph_->ToString();
    }
    if (device_contexts().empty() || device_contexts()[0] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid device context for actor:" << GetAID();
    }
    AnfNodePtrList inputs{};
    AnfNodePtrList outputs{return_node->cast<CNodePtr>()->input(1)};
    auto io_nodes = std::make_pair(inputs, outputs);
    device_contexts()[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    auto new_graph =
      compile_func_(BuildSegmentByGraph(model_graph_), io_nodes, device_contexts()[0], device::RunMode::kKernelMode);
    MS_EXCEPTION_IF_NULL(new_graph);
    MS_LOG(DEBUG) << "Compile graph:" << new_graph->ToString() << " for:" << model_graph_->ToString();
    for (const auto &node_pair : new_graph->front_backend_anf_map()) {
      MS_EXCEPTION_IF_NULL(node_pair.first);
      if (!node_pair.first->isa<CNode>()) {
        continue;
      }
      MS_LOG(DEBUG) << "Check for node:" << node_pair.first->DebugString();
      const auto &cnode = node_pair.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->HasAttr(kAttrReplaceRealKernelInBackend)) {
        MS_LOG(DEBUG) << "Erase flag for node:" << node_pair.first->DebugString();
        cnode->EraseAttr(kAttrReplaceRealKernelInBackend);
      }
    }
    PersisitValueNode(new_graph, device_contexts()[0]);
    return new_graph;
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    MS_LOG(ERROR) << "Failed to compile graph:" << model_graph_->ToString() << " in actor:" << GetAID()
                  << " error info:" << e.what();
    return nullptr;
  }
}

void AnyTypeKernelActor::PrepareRunContext(OpContext<DeviceTensor> *const context) {
  const auto &data_type = GenerateIDForGraph(input_device_tensors_, any_type_parameter_indexes_);
  if (data_type == "FAILED") {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  GetAID().Name() + " failed to generate id.");
  }
  if (data_type == current_data_type_) {
    return;
  }
  current_data_type_ = data_type;
  if (real_graphs_.find(current_data_type_) == real_graphs_.end()) {
    const auto &new_graph = CompileRealKernelGraph(context);
    if (new_graph == nullptr) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    GetAID().Name() + " compile graph failed.");
    }
    if (new_graph->input_nodes().size() != model_graph_->input_nodes().size()) {
      MS_LOG(EXCEPTION) << "Invalid input node num:" << new_graph->input_nodes().size()
                        << " in graph:" << new_graph->ToString() << " for model graph:" << model_graph_->ToString()
                        << " input num:" << model_graph_->input_nodes().size() << " for actor:" << GetAID();
    }
    real_graphs_[current_data_type_] = new_graph;
  }
  MS_EXCEPTION_IF_NULL(real_graphs_[current_data_type_]);
  graph_ = real_graphs_[current_data_type_];
  MS_LOG(DEBUG) << "Set graph:" << graph_->ToString() << " for actor:" << GetAID();
  ClearElements(context);
  MS_LOG(DEBUG) << "Clear element end and start build and link for actor:" << GetAID();
  BuildAndLinkKernelActors();
  return;
}

void AnyTypeKernelActor::ClearElements(OpContext<DeviceTensor> *const context) {
  memory_alloc_list_.clear();
  kernel_actors_insert_event_.clear();
  param_node_to_input_idx_.clear();
  cnode_to_kernel_actor_.clear();
  kernel_actors_.clear();
  kernel_input_to_graph_input_indices_.clear();
  kernel_input_to_actor_output_indices_.clear();
  MS_EXCEPTION_IF_NULL(model_graph_);
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &model_output = common::AnfAlgo::GetAllOutputWithOutMonadAndParameter(model_graph_->output());
  const auto &real_output = common::AnfAlgo::GetAllOutputWithOutMonadAndParameter(graph_->output());
  if (model_output.size() != real_output.size()) {
    MS_LOG(EXCEPTION) << "Invalid model output size:" << model_output.size()
                      << " and real output size:" << real_output.size() << " for actor:" << GetAID();
  }
  std::map<KernelWithIndex, KernelWithIndex> model_to_real_outputs;
  for (size_t i = 0; i < model_output.size(); ++i) {
    MS_EXCEPTION_IF_NULL(model_output[i].first);
    MS_EXCEPTION_IF_NULL(real_output[i].first);
    model_to_real_outputs[model_output[i]] = real_output[i];
  }
  if (output_data_nodes_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "Invalid output data node size:" << output_data_nodes_.size()
                      << " and arrow size:" << output_data_arrows_.size() << " for actor:" << GetAID();
  }
  MS_LOG(DEBUG) << "node size:" << output_data_nodes_.size() << " for actor:" << GetAID();
  for (size_t i = 0; i < output_data_nodes_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_data_nodes_[i]);
    MS_EXCEPTION_IF_NULL(output_data_arrows_[i]);
    const auto &iter = model_to_real_outputs.find({output_data_nodes_[i], output_data_arrows_[i]->from_output_index_});
    if (iter == model_to_real_outputs.end()) {
      MS_LOG(EXCEPTION) << "Failed to get real output by model output:" << output_data_nodes_[i]->DebugString()
                        << " model graph:" << model_graph_->ToString() << " real graph:" << graph_->ToString();
    }
    output_data_nodes_[i] = iter->second.first;
    output_data_arrows_[i]->from_output_index_ = SizeToInt(iter->second.second);
    MS_LOG(DEBUG) << "Update output node:" << iter->second.first->fullname_with_scope()
                  << " index:" << iter->second.second << " for actor:" << GetAID();
  }
  MS_LOG(DEBUG) << "Reorder Input device tensor";
  if (graph_->input_nodes().size() != model_graph_->input_nodes().size()) {
    MS_LOG(EXCEPTION) << "Invalid model input size:" << model_graph_->input_nodes().size()
                      << " and real input size:" << graph_->input_nodes().size() << " for actor:" << GetAID();
  }
  std::vector<DeviceTensor *> new_input_device_tensors;
  for (size_t i = 0; i < graph_->input_nodes().size(); ++i) {
    const auto &backend_node = graph_->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(backend_node);
    const auto &front_node = graph_->GetFrontAnfByBackendAnf(backend_node);
    if (front_node == nullptr) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
        GraphExecutionStrategy::kPipeline, (*context),
        "Failed to get front node by backend node:" + backend_node->DebugString() +
          " in real graph:" + graph_->ToString());
    }
    MS_LOG(DEBUG) << "output data front node:" << front_node->DebugString();
    const auto &front_parameters = model_graph_->input_nodes();
    const auto &iter = find(front_parameters.begin(), front_parameters.end(), front_node);
    if (iter == front_parameters.end()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(
        GraphExecutionStrategy::kPipeline, (*context),
        "Failed to find index by backend parameter:" + backend_node->DebugString() +
          " front node:" + front_node->DebugString());
    }
    size_t position = LongToSize(iter - front_parameters.begin());
    if (position >= input_device_tensors_.size()) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    "Invalid input position:" + std::to_string(position) +
                                                      " total size:" + std::to_string(input_device_tensors_.size()) +
                                                      " for actor:" + GetAID().Name());
    }
    new_input_device_tensors.emplace_back(input_device_tensors_[position]);
    MS_LOG(DEBUG) << "Add model index:" << position << " real input index:" << i
                  << " device tensor:" << input_device_tensors_[position] << " for actor:" << GetAID();
  }
  input_device_tensors_.swap(new_input_device_tensors);
  MS_LOG(DEBUG) << "Clear element end";
}

void AnyTypeKernelActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(model_graph_);
  actor_state_ = AnyTypeKernelActorState::kAnyTypeKernelActorSendInput;
  MS_LOG(DEBUG) << "Any type kernel actor:" << GetAID() << " run.";
  FetchInputDeviceTensor(context);
  if (!WaitRuntimePipelineFinish(context)) {
    MS_LOG(INFO) << "Run failed and early stop.";
    return;
  }
  PrepareRunContext(context);
  try {
    if (!LaunchAllKernels(context)) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    "Launch kernels by execution order failed for graph: " +
                                                      graph_->ToString() + " error info:" + context->error_info_);
    }
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    MS_LOG(WARNING) << "Catch error:" << e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), e.what());
  }
  EraseInput(context);
  PostRun(context);
}

void AnyTypeKernelActor::RunForGraphInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph());
  actor_state_ = AnyTypeKernelActorState::kAnyTypeKernelActorSendInput;
  MS_LOG(DEBUG) << "Any type kernel actor:" << GetAID() << " run for graph input.";
  FetchInputDeviceTensor(context);
  current_data_type_ = GenerateIDForGraph(input_device_tensors_, any_type_parameter_indexes_);
  MS_LOG(DEBUG) << "Current data type:" << current_data_type_ << " for actor:" << GetAID();
  std::vector<AbstractActorPtr> actors;
  if (real_graphs_.find(current_data_type_) == real_graphs_.end()) {
    try {
      std::lock_guard<std::mutex> lock(instance_lock_);
      InferParameterAbstractForModelGraph(graph(), input_device_tensors_, any_type_parameter_indexes_);
      ClearAttrForGraph(graph(), kAttrInputIsDynamicShape);
      ClearAttrForGraph(graph(), kAttrOutputIsDynamicShape);
      graph()->InferType();
      const auto &return_node = graph()->get_return();
      MS_EXCEPTION_IF_NULL(return_node);
      if (!return_node->isa<CNode>() || return_node->cast<CNodePtr>()->size() <= 1) {
        MS_LOG_WITH_NODE(EXCEPTION, return_node)
          << "Invalid return node:" << return_node->DebugString() << " for graph:" << graph()->ToString();
      }
      if (device_contexts().empty() || device_contexts()[0] == nullptr) {
        MS_LOG(EXCEPTION) << "Invalid device context for actor:" << GetAID();
      }
      AnfNodePtrList inputs{};
      AnfNodePtrList outputs{return_node->cast<CNodePtr>()->input(1)};
      auto io_nodes = std::make_pair(inputs, outputs);
      device_contexts()[0]->device_res_manager_->BindDeviceToCurrentThread(false);
      auto new_graph =
        compile_func_(BuildSegmentByGraph(graph()), io_nodes, device_contexts()[0], device::RunMode::kKernelMode);
      MS_EXCEPTION_IF_NULL(new_graph);
      MS_LOG(INFO) << "Add new kernel graph:" << new_graph->ToString() << " for graph:" << graph()->ToString();
      real_graphs_[current_data_type_] = new_graph;
      actors = transform_func_(graph(), new_graph, device_contexts()[0]);
      actors_[current_data_type_] = actors;
      schedule_func_(actors);

      for (const auto &node_pair : new_graph->front_backend_anf_map()) {
        MS_EXCEPTION_IF_NULL(node_pair.first);
        if (!node_pair.first->isa<CNode>()) {
          continue;
        }
        MS_LOG(DEBUG) << "Check for node:" << node_pair.first->DebugString();
        const auto &cnode = node_pair.first->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        if (cnode->HasAttr(kAttrReplaceRealKernelInBackend)) {
          MS_LOG(DEBUG) << "Erase flag for node:" << node_pair.first->DebugString();
          cnode->EraseAttr(kAttrReplaceRealKernelInBackend);
        }
      }
    } catch (const std::exception &e) {
      MsException::Instance().SetException();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), e.what());
    }
  }
  UpdataDynamicShapeParameterForGraphInput(context);
  EraseInput(context);
  if (memory_alloc_list_.size() > 0) {
    MS_LOG(EXCEPTION) << "Any type kernel actor:" << GetAID() << "cannot send memory alloc message.";
  } else {
    OnMemoryAllocFinish(context);
  }
}

size_t FetchInputIndexByBackendParameter(const AnfNodePtr &backend_node, const KernelGraphPtr &front_graph,
                                         const KernelGraphPtr &backend_graph) {
  MS_EXCEPTION_IF_NULL(backend_node);
  MS_EXCEPTION_IF_NULL(front_graph);
  MS_EXCEPTION_IF_NULL(backend_graph);
  const auto &front_node = backend_graph->GetFrontAnfByBackendAnf(backend_node);
  MS_EXCEPTION_IF_NULL(front_node);
  const auto &front_parameters = front_graph->input_nodes();
  const auto &iter = find(front_parameters.begin(), front_parameters.end(), front_node);
  if (iter == front_parameters.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, front_node)
      << "Invalid front parameter:" << front_node->DebugString() << " for graph:" << front_graph->ToString();
  }
  return iter - front_parameters.begin();
}
void AnyTypeKernelActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(graph());
  if (real_graphs_.find(current_data_type_) == real_graphs_.end()) {
    MS_LOG(EXCEPTION) << "Invalid index:" << current_data_type_ << " for any type kernel actor:" << GetAID();
  }
  const auto &real_graph = real_graphs_[current_data_type_];
  MS_EXCEPTION_IF_NULL(real_graph);
  if (real_graph->input_nodes().size() != graph()->input_nodes().size()) {
    MS_LOG(EXCEPTION) << "Invalid input node num:" << real_graph->input_nodes().size()
                      << " in graph:" << real_graph->ToString() << " for model graph:" << graph()->ToString()
                      << " input num:" << graph()->input_nodes().size() << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < node_device_tensors_.size(); ++i) {
    const auto &input_node = real_graph->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (HasAbstractMonad(input_node)) {
      continue;
    }
    size_t from_index = FetchInputIndexByBackendParameter(input_node, graph(), real_graph);
    if (!AnfAlgo::OutputAddrExist(input_node, 0, false)) {
      MS_LOG_WITH_NODE(EXCEPTION, input_node)
        << "Input node:" << input_node->DebugString() << " has no device address for actor:" << GetAID();
    }
    auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
    MS_EXCEPTION_IF_NULL(device_address);
    if (from_index >= node_device_tensors_.size() || from_index >= input_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid from index:" << from_index
                        << " node device tensor size:" << node_device_tensors_.size()
                        << " input device tensor size:" << input_device_tensors_.size() << " for actor:" << GetAID();
    }
    node_device_tensors_[from_index] = device_address;
    if (input_device_tensors_[from_index] == nullptr) {
      MS_LOG_WITH_NODE(EXCEPTION, input_node)
        << "actor:" << GetAID() << " real graph:" << real_graph->ToString()
        << " input node:" << input_node->DebugString() << " index : " << i << " is nullptr ";
    }
    node_device_tensors_[from_index]->SetNodeIndex(input_device_tensors_[from_index]->node_index().first.lock(),
                                                   input_device_tensors_[from_index]->node_index().second);
    MS_LOG(DEBUG) << "Actor:" << GetAID() << " input " << from_index << ":"
                  << " device address:" << device_address
                  << " original ref count:" << device_address->original_ref_count()
                  << " ref count:" << device_address->ref_count()
                  << " dynamic ref count:" << device_address->dynamic_ref_count()
                  << " real shape:" << node_device_tensors_[from_index]->kernel_tensor()->GetShape()->ToString()
                  << " model shape:" << input_device_tensors_[from_index]->kernel_tensor()->GetShape()->ToString();
  }
  if (node_device_tensors_.size() != input_device_tensors_.size()) {
    MS_LOG(EXCEPTION) << "Invalid device tensor num:" << input_device_tensors_.size() << " and "
                      << node_device_tensors_.size() << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < node_device_tensors_.size(); ++i) {
    if (node_device_tensors_[i] != nullptr && input_device_tensors_[i] != nullptr) {
      MS_EXCEPTION_IF_NULL(input_device_tensors_[i]->kernel_tensor());
      MS_EXCEPTION_IF_NULL(node_device_tensors_[i]->kernel_tensor());
      MS_LOG(DEBUG) << "set shape:"
                    << (input_device_tensors_[i]->kernel_tensor()->GetShape() == nullptr
                          ? "null"
                          : input_device_tensors_[i]->kernel_tensor()->GetShape()->ToString())
                    << " type:"
                    << (input_device_tensors_[i]->kernel_tensor()->GetType() == nullptr
                          ? "null"
                          : input_device_tensors_[i]->kernel_tensor()->GetType()->ToString())
                    << " from device address:" << input_device_tensors_[i]
                    << " to device address:" << node_device_tensors_[i];
      node_device_tensors_[i]->kernel_tensor()->SetType(input_device_tensors_[i]->kernel_tensor()->GetType());
      node_device_tensors_[i]->kernel_tensor()->SetShape(input_device_tensors_[i]->kernel_tensor()->GetShape());
      node_device_tensors_[i]->kernel_tensor()->SetValue(input_device_tensors_[i]->kernel_tensor()->GetValueTrack());
      MS_LOG(DEBUG) << "set shape:" << input_device_tensors_[i]->kernel_tensor()->GetShape()->ToString()
                    << " from device address:" << input_device_tensors_[i]
                    << " to device address:" << node_device_tensors_[i];
    }
  }
  CopyInputData(context, real_graphs_[current_data_type_]);
  if (!memory_free_lists_.empty()) {
    for (size_t i = 0; i < node_device_tensors_.size(); ++i) {
      if (node_device_tensors_[i] != nullptr) {
        memory_free_lists_.back().emplace_back(node_device_tensors_[i].get());
      }
    }
  }
  SendOutput(context);
}

void AnyTypeKernelActor::EraseGraphOutput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if ((graph_output_data_num_[current_data_type_] != 0) && (!graph_output_op_data_.empty())) {
    auto ret = graph_output_op_data_.erase(context->sequential_num_);
    if (ret == 0) {
      MS_LOG(WARNING) << "Erase graph output data failed: " << GetAID().Name()
                      << ", sequential_num: " << context->sequential_num_;
      return;
    }
  }

  if ((graph_output_control_num_[current_data_type_] != 0) && (!graph_output_op_control_.empty())) {
    auto ret = graph_output_op_control_.erase(context->sequential_num_);
    if (ret == 0) {
      MS_LOG(WARNING) << "Erase graph output controls failed: " << GetAID().Name()
                      << ", sequential_num: " << context->sequential_num_;
      return;
    }
  }
}

void AnyTypeKernelActor::RunForGraphOutput(OpContext<DeviceTensor> *const context) {
  MS_LOG(DEBUG) << "actor:" << GetAID() << " run for graph output start";
  actor_state_ = AnyTypeKernelActorState::kAnyTypeKernelActorSendOutput;
  FetchGraphOutput(context);
  EraseGraphOutput(context);
  SendMemoryFreeReq(context);
  AbstractActor::SendOutput(context);
}

void AnyTypeKernelActor::Init() {
  MS_EXCEPTION_IF_NULL(graph());
  MS_LOG(DEBUG) << "actor:" << GetAID() << " init";
  SuperKernelActor::Init();
  memory_alloc_list_.clear();
  for (size_t i = 0; i < graph()->input_nodes().size(); ++i) {
    const auto &input = graph()->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(input);
    const auto &abs = input->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractAny>()) {
      any_type_parameter_indexes_.emplace_back(i);
      MS_LOG(DEBUG) << "Add any type parameter index:" << i << " by parameter:" << input->DebugString()
                    << " for actor:" << GetAID();
    }
  }
  model_graph_ = graph();
  extern_device_tensor_store_keys_.swap(device_tensor_store_keys_);
  for (const auto &node_with_index : common::AnfAlgo::GetAllOutputWithOutMonadAndParameter(graph()->output())) {
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    if (!AnfAlgo::OutputAddrExist(node_with_index.first, node_with_index.second)) {
      MS_LOG_WITH_NODE(EXCEPTION, node_with_index.first)
        << "Failed to get output address from node:" << node_with_index.first->DebugString()
        << " index:" << node_with_index.second << " for actor:" << GetAID();
    }
    graph_ouput_device_tensors_.emplace_back(
      AnfAlgo::GetMutableOutputAddr(node_with_index.first, node_with_index.second, false).get());
  }
  fallback_device_tensors_.resize(graph_ouput_device_tensors_.size());
}

namespace {
void FreeMemory(DeviceTensor *device_tensor) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_tensor->device_name(), device_tensor->device_id()});
  if (device_context == nullptr || device_context->device_res_manager_ == nullptr) {
    return;
  }
  MS_LOG(DEBUG) << "Device tensor:" << device_tensor << " release memory:" << device_tensor->GetMutablePtr();
  device_context->device_res_manager_->FreeMemory(device_tensor->GetMutablePtr());
  device_tensor->set_ptr(nullptr);
}
}  // namespace

void AnyTypeKernelActor::CheckParams(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph());
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  "Invalid device context for any type actor:" + GetAID().Name());
  }
}

void AnyTypeKernelActor::FetchGraphOutput(OpContext<DeviceTensor> *const context) {
  CheckParams(context);
  const auto &data_iter = graph_output_op_data_.find(context->sequential_num_);
  if (data_iter != graph_output_op_data_.end()) {
    std::set<DeviceTensor *> clear_device_tensors;
    for (auto &graph_output_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(graph_output_data);
      MS_EXCEPTION_IF_NULL(graph_output_data->data_);
      size_t index = IntToSize(graph_output_data->index_);
      if (index < graph()->input_nodes().size()) {
        MS_LOG(WARNING) << "Invalid graph output index:" << index << " input num:" << input_datas_num_
                        << " for actor:" << GetAID();
        continue;
      }
      index -= graph()->input_nodes().size();
      if (index >= graph_ouput_device_tensors_.size() ||
          graph_ouput_device_tensors_.size() != fallback_device_tensors_.size()) {
        std::string error_info = "Invalid graph output index:" + std::to_string(index) +
                                 " total:" + std::to_string(graph_ouput_device_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_LOG(DEBUG) << "Fetch graph output index:" << index << " set ptr:" << graph_output_data->data_->GetMutablePtr()
                    << " size:" << graph_output_data->data_->GetSize()
                    << " from device address:" << graph_output_data->data_
                    << " to:" << graph_ouput_device_tensors_[index] << " for actor:" << GetAID();
      MS_EXCEPTION_IF_NULL(graph_ouput_device_tensors_[index]);
      if (graph_ouput_device_tensors_[index]->GetDeviceType() != graph_output_data->data_->GetDeviceType()) {
        MS_LOG(INFO) << "Different device type for actor:" << GetAID()
                     << " front device address:" << graph_ouput_device_tensors_[index]
                     << " device type:" << graph_ouput_device_tensors_[index]->GetDeviceType()
                     << " backend device address:" << graph_output_data->data_
                     << " device type:" << graph_output_data->data_->GetDeviceType();
        if (fallback_device_tensors_[index] != nullptr) {
          if (fallback_device_tensors_[index]->GetDeviceType() != graph_output_data->data_->GetDeviceType()) {
            MS_LOG(ERROR) << "Invalid device type for actor:" << GetAID()
                          << " fallback device address:" << fallback_device_tensors_[index]
                          << " device type:" << fallback_device_tensors_[index]->GetDeviceType()
                          << " backend device address:" << graph_output_data->data_
                          << " device type:" << graph_output_data->data_->GetDeviceType();
            SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), GetAID().Name() + " invalid device type.");
          }
        } else {
          auto tmp_device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
            {graph_output_data->data_->device_name(), graph_output_data->data_->device_id()});
          MS_EXCEPTION_IF_NULL(tmp_device_context);

          const auto &graph_output_kernel_tensor = graph_output_data->data_->kernel_tensor();
          MS_EXCEPTION_IF_NULL(graph_output_kernel_tensor);
          const auto &fallback_kernel_tensor = graph_output_kernel_tensor->CloneKernelTensor();
          MS_EXCEPTION_IF_NULL(fallback_kernel_tensor);
          fallback_kernel_tensor->set_device_ptr(nullptr);
          fallback_device_tensors_[index] =
            tmp_device_context->device_res_manager_->CreateDeviceAddress(fallback_kernel_tensor);
          MS_EXCEPTION_IF_NULL(fallback_device_tensors_[index]);
          MS_LOG(DEBUG) << "Create device address:" << fallback_device_tensors_[index] << " for actor:" << GetAID()
                        << " index:" << index << " device type:" << fallback_device_tensors_[index]->GetDeviceType()
                        << " size:" << fallback_device_tensors_[index]->GetSize();
          fallback_device_tensors_[index]->set_ref_count(graph_ouput_device_tensors_[index]->ref_count());
          fallback_device_tensors_[index]->set_original_ref_count(
            graph_ouput_device_tensors_[index]->original_ref_count());
          fallback_device_tensors_[index]->set_dynamic_ref_count(
            graph_ouput_device_tensors_[index]->dynamic_ref_count());
        }
        graph_ouput_device_tensors_[index] = fallback_device_tensors_[index].get();
      }
      if (graph_ouput_device_tensors_[index]->GetPtr() != nullptr) {
        // As the from memory pool flag of any type kernel graph is false, the memory cannot be released automatically,
        // and the memory needs to be released before overwriting.
        FreeMemory(graph_ouput_device_tensors_[index]);
      }
      graph_ouput_device_tensors_[index]->set_ptr(graph_output_data->data_->GetMutablePtr());
      graph_ouput_device_tensors_[index]->set_need_sync_user_data(graph_output_data->data_->need_sync_user_data());
      clear_device_tensors.emplace(graph_output_data->data_);
      graph_ouput_device_tensors_[index]->SetSize(graph_output_data->data_->GetSize());

      // Update Shape.
      const auto &graph_output_device_kernel_tensor = graph_ouput_device_tensors_[index]->kernel_tensor();
      const auto &graph_output_data_kernel_tensor = graph_output_data->data_->kernel_tensor();
      MS_EXCEPTION_IF_NULL(graph_output_device_kernel_tensor);
      MS_EXCEPTION_IF_NULL(graph_output_data_kernel_tensor);
      MS_LOG(DEBUG) << "actor:" << GetAID() << " set shape from device address:" << graph_output_data->data_
                    << " to:" << graph_ouput_device_tensors_[index]
                    << " for shape:" << graph_output_data_kernel_tensor->GetShape()->ToString();
      graph_output_device_kernel_tensor->SetType(graph_output_data_kernel_tensor->GetType()->Clone());
      graph_output_device_kernel_tensor->SetShape(graph_output_data_kernel_tensor->GetShape()->Clone());

      auto node_with_index = graph_output_data->data_->node_index();
      graph_ouput_device_tensors_[index]->SetNodeIndex(node_with_index.first.lock(), node_with_index.second);
      MS_LOG(DEBUG) << "Actor:" << GetAID() << "src device address:" << graph_output_data->data_
                    << " shape:" << graph_output_data->data_->host_shape()
                    << " type:" << graph_output_data->data_->type_id()
                    << "dst device address:" << graph_ouput_device_tensors_[index]
                    << " shape:" << graph_ouput_device_tensors_[index]->host_shape()
                    << " type:" << graph_ouput_device_tensors_[index]->type_id();
      graph_ouput_device_tensors_[index]->set_type_id(graph_output_data->data_->type_id());
      graph_ouput_device_tensors_[index]->set_host_shape(graph_output_data->data_->host_shape());
      graph_ouput_device_tensors_[index]->set_user_data(graph_output_data->data_->user_data());
    }
    for_each(clear_device_tensors.begin(), clear_device_tensors.end(),
             [](DeviceTensor *device_tensor) { device_tensor->set_ptr(nullptr); });
  }
}

void AnyTypeKernelActor::UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                                          const AnfNodePtr &output_node, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_data);
  MS_EXCEPTION_IF_NULL(data_arrow);
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(context);
  if (!output_node->isa<CNode>()) {
    return;
  }
  if (!AnfAlgo::OutputAddrExist(output_node, data_arrow->from_output_index_, false)) {
    std::stringstream error_info;
    error_info << "Failed to get output device address for:" << output_node->DebugString()
               << " index:" << data_arrow->from_output_index_ << " for actor:" << GetAID();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info.str());
  }
  output_data->data_ = AnfAlgo::GetMutableOutputAddr(output_node, data_arrow->from_output_index_, false).get();
  MS_LOG(DEBUG) << "Set output address:" << output_data->data_
                << " to output data, output index:" << data_arrow->from_output_index_ << " node:" << output_node
                << " in actor:" << GetAID();
}
}  // namespace runtime
}  // namespace mindspore
