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
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "ir/dtype.h"
#include "ir/tensor.h"
#include "ir/core_ops_name.h"
#include "ir/tensor_new.h"
#include "ir/device_address_maker.h"
#include "ir/device_sync.h"
#include "ir/dtype/type_id.h"
#include "utils/shape_utils.h"
#include "mindapi/base/format.h"
#include "utils/anf_utils.h"
#include "utils/llm_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/ccsrc/include/common/utils/convert_utils.h"
#include "runtime/pipeline/pipeline.h"
#include "debug/profiler/profiler.h"

#include "backend/ms_infer_backend/graph_adapter.h"
#include "backend/ms_infer_backend/host_value_store.h"
#include "backend/ms_infer_backend/utils.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {
namespace {
TypePtr GetSequenceElementType(const ValueSequencePtr &value_seq) {
  MS_EXCEPTION_IF_NULL(value_seq);

  const auto &element_values = value_seq->value();
  if (element_values.empty()) {
    MS_LOG(INFO) << "The sequence is empty: " << value_seq->ToString();
    return nullptr;
  }

  const auto &first_element = element_values[0];
  if (!first_element->isa<Scalar>()) {
    MS_LOG(EXCEPTION) << "Only sequence of scalar is valid, but got: " << value_seq->ToString();
  }
  return first_element->type();
}

void SetDATensorTypeAndShape(da::tensor::DATensor *tensor, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(value);

  auto shape = ShapeVector();
  TypePtr dtype = nullptr;
  if (utils::isa<Scalar>(value)) {
    dtype = value->type();
  } else if (utils::isa<Monad>(value)) {
    tensor->type = da::tensor::Type_Monad;
    return;
  } else if (utils::isa<ValueSequence>(value)) {
    auto value_seq = utils::cast<ValueSequencePtr>(value);
    dtype = GetSequenceElementType(value_seq);
    (void)shape.emplace_back(value_seq->value().size());
  } else if (utils::isa<tensor::Tensor>(value)) {
    auto tensor_ptr = utils::cast<tensor::TensorPtr>(value);
    dtype = tensor_ptr->Dtype();
    shape = tensor_ptr->shape();
  } else if (utils::isa<None>(value)) {
    tensor->type = da::tensor::Type_None;
    return;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unsupported type " << value->ToString();
  }

  tensor->type = ConvertDataType(dtype);
  SetTensorShape(tensor, shape);
}
}  // namespace

void GraphAdapter::SetNodeOutputType(da::tensor::DATensor *tensor, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(tensor);

  const TypePtr &type = node->Type();
  MS_EXCEPTION_IF_NULL(type);
  const BaseShapePtr &shape = node->Shape();
  MS_EXCEPTION_IF_NULL(shape);

  if (type->isa<TensorType>()) {
    tensor->type = ConvertDataType(dyn_cast<TensorType>(type)->element());
    SetTensorShape(tensor, shape->GetShapeVector());
  } else if (type->isa<Tuple>()) {
    graph_executor_.CastToTensorList(tensor, AnfUtils::GetOutputTensorNum(node));
    MS_EXCEPTION_IF_CHECK_FAIL(tensor->type == da::tensor::Type_Tensor, "The type of DATensor is not Type_Tensor");
    SetTupleType(tensor, type->cast<TuplePtr>());
    SetTupleShape(tensor, shape->cast<TupleShapePtr>());
  } else if (type->isa<MonadType>()) {
    tensor->type = da::tensor::Type_Monad;
  } else {
    tensor->type = ConvertDataType(type->type_id());
  }
}

da::tensor::DATensor *GraphAdapter::ConvertValueNode(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = GetValueNode(value_node);
  MS_EXCEPTION_IF_NULL(value);
  MS_LOG(INFO) << "Convert value to DATensor: " << value.get() << ", " << value->ToString();

  auto da_tensor = graph_executor_.AddTensor();
  PrepareData(da_tensor, value);

  return da_tensor;
}

void GraphAdapter::PrepareData(da::tensor::DATensor *da_value, const ValuePtr &value) {
  // Set tensor type and shape
  SetDATensorTypeAndShape(da_value, value);

  if (da_value->type == da::tensor::Type_Monad || da_value->type == da::tensor::Type_None) {
    return;
  }

  // malloc for all parameters and valuenodes and copy them to device
  if (value->isa<tensor::Tensor>()) {
    da_value->tensorType = da::tensor::TensorType::DEVICE_TENSOR;
    da_value->data = PrepareTensorDataToDevice(value->cast<tensor::TensorPtr>());
  } else if (value->isa<ValueSequence>() || value->isa<Scalar>() || value->isa<StringImm>()) {
    da_value->tensorType = da::tensor::TensorType::HOST_TENSOR;
    HostValueStore::GetInstance().InsertValueForDATensor(da_value, value);
    auto kernel_tensor_value = ConvertValueToKernelTensorValue(value);
    MS_EXCEPTION_IF_NULL(kernel_tensor_value);
    (void)converted_values_.emplace(kernel_tensor_value);
    MS_LOG(INFO) << "Create ktvalue for DATensor: " << da_value << ", data_ptr: " << kernel_tensor_value->GetDataPtr();
    da_value->data = const_cast<void *>(kernel_tensor_value->GetDataPtr());
  } else {
    MS_LOG(EXCEPTION) << "Unsupported value: " << value->ToString();
  }
}

void *GraphAdapter::PrepareTensorDataToDevice(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context_);

  MS_LOG(INFO) << "start prepare tensor data to device, tensor: " << tensor->ToString();

  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->GetSize() == 0) {
    MS_LOG(DEBUG) << "input tensor size is 0, skip prepare, tensor: " << tensor->ToString();
    return nullptr;
  }

  if (device_address->GetDeviceType() != device_context_->GetDeviceType()) {
    auto &device_context_key = device_context_->device_context_key();
    MS_LOG(INFO) << "need sync data to device, device name: " << device_context_key.device_name_;
    // create new device address for tensor
    auto new_device_address = device_context_->device_res_manager_->CreateDeviceAddress(
      nullptr, tensor->Size(), tensor->shape(), Format::DEFAULT_FORMAT, tensor->data_type(),
      device_context_key.device_name_, device_context_key.device_id_, kDefaultStreamIndex);
    // malloc device memory
    device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
    if (!device_context_->device_res_manager_->AllocateMemory(new_device_address.get(), kDefaultStreamIndex)) {
      MS_LOG(EXCEPTION) << "Allocate memory failed, device address: " << new_device_address->ToString()
                        << ", tensor: " << tensor->ToString();
    }
    // async H2D copy
    if (!AsyncCopy(new_device_address, device_address, kDefaultStreamIndex)) {
      MS_LOG(EXCEPTION) << "Failed async copy H2D for tensor: " << tensor->ToString()
                        << ", dst: " << new_device_address->ToString() << ", src: " << device_address->ToString();
    }
    // set the new device address to tensor
    tensor->set_device_address(new_device_address);
    return new_device_address->GetMutablePtr();
  }

  if (device_address->GetMutablePtr() == nullptr) {
    MS_LOG(EXCEPTION) << "Invalid device ptr, tensor: " << tensor->ToString()
                      << ", device address: " << device_address->ToString();
  }

  return device_address->GetMutablePtr();
}

da::tensor::DATensor *GraphAdapter::GetNodeDATensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);

  if (node->isa<ValueNode>()) {
    auto iter = const_map_.find(node);
    if (iter == const_map_.end()) {
      const_map_[node] = ConvertValueNode(node->cast<ValueNodePtr>());
    }
    return const_map_[node];
  }

  if (node->isa<CNode>()) {
    auto iter = apply_map_.find(node);
    if (iter == apply_map_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Can not find node '" << node << "' in apply_map_";
    }
    return iter->second;
  }

  if (node->isa<Parameter>()) {
    auto iter = parameter_map_.find(node);
    if (iter == parameter_map_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Can not find node '" << node << "' in parameter_map_";
    }
    return iter->second;
  }

  MS_LOG(INTERNAL_EXCEPTION) << "Unknown node type. node is '" << node << "'";
}

void GraphAdapter::ConvertGraph() {
  MS_LOG(INFO) << "Convert graph: " << func_graph_->ToString();

  WaitTaskFinish();

  SetupFrontendParameterMapping();

  // parameters DATensor should be created before BeginGraph, added as parameters after BeginGraph
  ConvertParameters();

  graph_executor_.BeginGraph(func_graph_->ToString());
  InsertParameters();
  ConvertCNodes();
  graph_executor_.EndGraph();
  graph_executor_.BuildKernels();
  graph_executor_.RecordTensorRefCount();
  graph_executor_.SetFreeFunc([this](void *data) {
    MS_EXCEPTION_IF_NULL(device_context_);
    MS_EXCEPTION_IF_NULL(data);
    device_context_->device_res_manager_->FreeMemory(data);
  });

  graph_executor_.DumpGraph();
}

void GraphAdapter::RunGraph(const VectorRef &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  MS_LOG(INFO) << "Run graph: " << func_graph_->ToString()
               << ", is_dynamic_shape_: " << func_graph_->is_dynamic_shape();

  if (AnfAlgo::IsGraphOutputValueNodeOrParameter(func_graph_->output(), inputs, outputs)) {
    return;
  }

  WaitTaskFinish();

  ConvertInputs(inputs);
  MS_LOG(INFO) << "Begin run DAGraph, is_dynamic_shape: " << is_dynamic_shape_;
  graph_executor_.RunGraph(is_dynamic_shape_);
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  ConvertOutputs(outputs);

  auto &llm_manger = LLMManager::GetInstance();
  llm_manger.reset_graph_inputs();
  PROFILER_END(start_time, runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kOutputProcess,
               func_graph_->ToString(), false);
}

void GraphAdapter::WaitTaskFinish() const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                     runtime::kDefaultOpName);
  runtime::Pipeline::Get().WaitAll();
}

void GraphAdapter::SetupFrontendParameterMapping() {
  const auto &backend_params = func_graph_->parameters();
  for (size_t j = 0; j < backend_params.size(); j++) {
    const auto &backend_param = backend_params[j];
    MS_EXCEPTION_IF_NULL(backend_param);

    auto frontend_param_with_index = func_graph_->GetElementInTupleBackendFrontIndexMap(backend_param);
    if (frontend_param_with_index.first == nullptr) {
      frontend_param_with_index = func_graph_->GetFrontNodeByInternalParameter(backend_param);
      if (frontend_param_with_index.first == nullptr) {
        frontend_param_with_index = {func_graph_->GetFrontAnfByBackendAnf(backend_param), 0};
      }
    }
    MS_EXCEPTION_IF_NULL(frontend_param_with_index.first);

    (void)frontend_params_to_backend_params_[frontend_param_with_index.first].emplace_back(
      std::make_pair(frontend_param_with_index.second, backend_param));
  }
}

void GraphAdapter::ConvertInputs(const VectorRef &inputs) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kInputProcess,
                                     func_graph_->ToString());
  const auto &frontend_params = func_graph_->GetFuncGraph()->parameters();
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == frontend_params.size(),
                             "The inputs size is not equal to graph frontend params size.");
  MS_LOG(DEBUG) << "Graph inputs size: " << inputs.size();
  // front node index to input tensors
  std::map<size_t, std::vector<tensor::TensorPtr>> infer_input_tensors;

  if (is_first_step_) {
    PrepareAllInputs(inputs, frontend_params, &infer_input_tensors);
    is_first_step_ = false;
  } else {
    PrepareNonWeightInputs(inputs, frontend_params, &infer_input_tensors);
  }

  RecordInputTensorShapes(infer_input_tensors);
}

void GraphAdapter::PrepareAllInputs(const VectorRef &inputs, const AnfNodePtrList &frontend_params,
                                    std::map<size_t, std::vector<tensor::TensorPtr>> *infer_input_tensors) {
  MS_EXCEPTION_IF_NULL(infer_input_tensors);
  for (size_t i = 0; i < inputs.size(); ++i) {
    // flatten input tensors
    std::vector<tensor::TensorPtr> flatten_input_tensors;
    AnfAlgo::FlattenInputArg(inputs[i], frontend_params[i], &flatten_input_tensors);

    // find backend params
    auto frontend_param = frontend_params[i];
    auto iter1 = frontend_params_to_backend_params_.find(frontend_param);
    if (iter1 == frontend_params_to_backend_params_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Can not find the frontend parameters: " << frontend_param->fullname_with_scope();
    }
    auto backend_params = iter1->second;
    MS_LOG(INFO) << "frontend parameters: " << frontend_param->fullname_with_scope();

    for (auto &frontend_index_to_backend_param : backend_params) {
      // get input_tensor
      auto frontend_param_index = frontend_index_to_backend_param.first;
      size_t input_tensor_index = 0;
      const auto &frontend_param_abs = frontend_param->abstract();
      MS_EXCEPTION_IF_NULL(frontend_param_abs);
      if (frontend_param_abs->isa<abstract::AbstractSequence>() &&
          !common::AnfAlgo::IsDynamicSequence(frontend_param)) {
        input_tensor_index = frontend_param_index;
      }
      if (input_tensor_index >= flatten_input_tensors.size()) {
        MS_LOG(INTERNAL_EXCEPTION) << "Input tensor index out of args range, index: " << input_tensor_index
                                   << ", tensors size: " << flatten_input_tensors.size()
                                   << ", parameter: " << frontend_param->fullname_with_scope();
      }

      auto backend_param = frontend_index_to_backend_param.second;
      auto input_tensor = flatten_input_tensors[input_tensor_index];
      if (!common::AnfAlgo::IsParameterWeight(backend_param->cast<ParameterPtr>())) {
        (void)(*infer_input_tensors)[i].emplace_back(input_tensor);
        (void)front_node_index_to_backend_nodes_with_index_[i].emplace_back(
          std::make_pair(backend_param, input_tensor_index));
        auto &llm_manger = LLMManager::GetInstance();
        llm_manger.add_graph_input(backend_param->fullname_with_scope(),
                                   std::make_shared<tensor::Tensor>(*input_tensor));
        MS_LOG(DEBUG) << "Record input tensor: " << input_tensor->ToString()
                      << "for parameter: " << backend_param->fullname_with_scope();
      }

      // get da_param from parameter_map_
      auto iter2 = parameter_map_.find(backend_param);
      if (iter2 == parameter_map_.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << "Can not find parameter '" << backend_param->ToString() << "' in parameter_map_";
      }
      auto da_param = iter2->second;
      PrepareData(da_param, input_tensor);
    }
  }
}

void GraphAdapter::PrepareNonWeightInputs(const VectorRef &inputs, const AnfNodePtrList &frontend_params,
                                          std::map<size_t, std::vector<tensor::TensorPtr>> *infer_input_tensors) {
  MS_EXCEPTION_IF_NULL(infer_input_tensors);
  for (auto &front_node_to_backend_nodes : front_node_index_to_backend_nodes_with_index_) {
    auto front_index = front_node_to_backend_nodes.first;
    auto &backend_nodes_with_index = front_node_to_backend_nodes.second;
    if (front_index < 0 || front_index >= inputs.size()) {
      MS_LOG(EXCEPTION) << "Invalid front node index: " << front_index;
    }
    // flatten input tensors
    std::vector<tensor::TensorPtr> flatten_input_tensors;
    AnfAlgo::FlattenInputArg(inputs[front_index], frontend_params[front_index], &flatten_input_tensors);
    for (auto &backend_node_with_index : backend_nodes_with_index) {
      auto backend_param = backend_node_with_index.first;
      auto input_tensor_index = backend_node_with_index.second;
      if (input_tensor_index < 0 || input_tensor_index >= flatten_input_tensors.size()) {
        MS_LOG(EXCEPTION) << "Invalid input_tensor_index: " << input_tensor_index;
        return;
      }
      (void)(*infer_input_tensors)[front_index].emplace_back(flatten_input_tensors[input_tensor_index]);
      auto &llm_manger = LLMManager::GetInstance();
      llm_manger.add_graph_input(backend_param->fullname_with_scope(),
                                 std::make_shared<tensor::Tensor>(*flatten_input_tensors[input_tensor_index]));
      MS_LOG(DEBUG) << "Record input tensor: " << flatten_input_tensors[input_tensor_index]->ToString()
                    << "for parameter: " << backend_param->fullname_with_scope();
      // get da_param from parameter_map_
      auto iter = parameter_map_.find(backend_param);
      if (iter == parameter_map_.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << "Can not find parameter '" << backend_param->ToString() << "' in parameter_map_";
      }
      auto da_param = iter->second;
      PrepareData(da_param, flatten_input_tensors[input_tensor_index]);
    }
  }
}

void GraphAdapter::RecordInputTensorShapes(const std::map<size_t, std::vector<tensor::TensorPtr>> &input_tensors) {
  MS_EXCEPTION_IF_CHECK_FAIL(input_tensors.size() == infer_input_tensors_shape_.size(),
                             "args size is not equal to ordinary_input_tensors_shape_ size");
  is_dynamic_shape_ = false;
  size_t index = kIndex0;
  for (auto &frontend_index_to_input_tensors : input_tensors) {
    auto infer_input_tensors = frontend_index_to_input_tensors.second;
    if (infer_input_tensors.size() != 1) {
      MS_LOG(DEBUG) << "Skip record list tensor input";
      ++index;
      continue;
    }
    auto input_tensor = infer_input_tensors[kIndex0];
    if (input_tensor == nullptr) {
      MS_LOG(DEBUG) << "Input tensor is nullptr";
      ++index;
      continue;
    }
    if (!is_dynamic_shape_) {
      if (infer_input_tensors_shape_[index] != input_tensor->shape() || input_tensor->shape().empty()) {
        is_dynamic_shape_ = true;
      }
    }
    infer_input_tensors_shape_[index++] = input_tensor->shape();
  }
}

void GraphAdapter::ConvertOutputs(VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  MS_EXCEPTION_IF_NULL(device_context_);
  MS_LOG(INFO) << "start convert outputs";

  MS_LOG(INFO) << "start get DA output node, ms node: " << func_graph_->get_return()->fullname_with_scope();
  auto *output_da_node = GetNodeDATensor(func_graph_->get_return());
  std::vector<da::tensor::DATensor *> output_da_tensors;
  if (output_da_node->type == da::tensor::Type_Tensor) {
    MS_LOG(INFO) << "get multiple outputs";
    auto **tensor_list = reinterpret_cast<da::tensor::DATensor **>(output_da_node->data);
    MS_EXCEPTION_IF_NULL(tensor_list);
    for (size_t i = 0; i < output_da_node->shape[0]; ++i) {
      (void)output_da_tensors.emplace_back(tensor_list[i]);
    }
  } else {
    MS_LOG(INFO) << "get single output";
    (void)output_da_tensors.emplace_back(output_da_node);
  }

  for (auto &da_tensor : output_da_tensors) {
    MS_LOG(INFO) << "start convert output tensor";
    ShapeVector shape;
    for (size_t i = 0; i < da_tensor->dim; ++i) {
      (void)shape.emplace_back(da_tensor->shape[i]);
    }
    auto dtype = ConvertDataType(da_tensor->type);
    MS_LOG(INFO) << "start construct output tensor, shape: " << shape << ", dtye: " << dtype;
    MS_EXCEPTION_IF_NULL(da_tensor->data);
    auto device_address = device_context_->device_res_manager_->CreateDeviceAddress(
      da_tensor->data, SizeOf(shape) * abstract::TypeIdSize(dtype), shape, Format::DEFAULT_FORMAT, dtype,
      device_context_->device_context_key().device_name_, device_context_->device_context_key().device_id_,
      kDefaultStreamIndex);
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->set_from_mem_pool(true);
    auto output = std::make_shared<tensor::Tensor>(dtype, shape, device_address);
    MS_EXCEPTION_IF_NULL(output);
    MS_LOG(INFO) << "converted output tensor: " << output->ToString();
    (void)outputs->emplace_back(output);
  }

  MS_LOG(INFO) << "end convert outputs";
}

void GraphAdapter::ConvertParameters() {
  auto root_graph = func_graph_->GetFuncGraph();
  MS_EXCEPTION_IF_NULL(root_graph);
  infer_input_tensors_shape_.resize(root_graph->get_inputs().size());
  for (auto &param : func_graph_->parameters()) {
    const ParameterPtr param_ptr = dyn_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_ptr);

    auto da_param = graph_executor_.AddTensor();
    SetNodeOutputType(da_param, param);
    parameter_map_[param] = da_param;
  }
}

void GraphAdapter::InsertParameters() {
  for (auto &item : parameter_map_) {
    graph_executor_.AddParameter(item.second);
  }
}

void GraphAdapter::ConvertCNodes() {
  auto nodes = TopoSort(func_graph_->get_return(), SuccIncoming, AlwaysInclude);
  for (const AnfNodePtr &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<CNode>()) {
      ConvertCNode(node->cast<CNodePtr>());
    }
  }
}

void GraphAdapter::ConvertCNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);

  auto &inputs = node->inputs();
  if (inputs.size() < 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "Inputs of CNode is empty" << node->ToString();
  }

  // Get primitive
  AnfNodePtr op = inputs[0];
  if (!IsValueNode<Primitive>(op)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Operator must be a primitive" << node->ToString();
  }
  auto prim = GetValueNode<PrimitivePtr>(op);
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Convert CNode: " << node << ", primitive: " << prim->ToString();

  // Add output DATensor
  auto da_op = ConvertPrimitiveOp(prim);
  std::vector<da::tensor::DATensor *> da_inputs;
  for (size_t i = 1; i < inputs.size(); ++i) {  // skip the first input which is the primitive
    (void)da_inputs.emplace_back(GetNodeDATensor(inputs[i]));
  }
  auto da_cnode = graph_executor_.AddTensor(da_op, da_inputs);
  HostValueStore::GetInstance().InsertPrimForDATensor(da_cnode, prim);
  SetNodeOutputType(da_cnode, node);
  apply_map_[node] = da_cnode;
}

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
