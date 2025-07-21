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
#include "utils/shape_utils.h"
#include "utils/anf_utils.h"
#include "utils/llm_manager.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

#include "backend/ms_infer_backend/graph_adapter.h"
#include "backend/ms_infer_backend/host_value_store.h"
#include "backend/ms_infer_backend/device_tensor_store.h"
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
    graph_executor_.AddTensorList(tensor, AnfUtils::GetOutputTensorNum(node));
    MS_EXCEPTION_IF_CHECK_FAIL(tensor->type == da::tensor::Type_Tensor, "The type of DATensor is not Type_Tensor");
    auto tuple_type = type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    SetTupleType(tensor, tuple_type);
    auto tuple_shape = shape->cast<TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    SetTupleShape(tensor, tuple_shape);
  } else if (type->isa<MonadType>()) {
    tensor->type = da::tensor::Type_Monad;
  } else {
    tensor->type = ConvertDataType(type->type_id());
  }
}

void GraphAdapter::ConvertValueNode(const ValueNodePtr &value_node) {
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = GetValueNode(value_node);
  MS_EXCEPTION_IF_NULL(value);

  if (HostValueStore::GetInstance().HasValue(value)) {
    auto da_tensor = HostValueStore::GetInstance().Get(value);
    MS_EXCEPTION_IF_NULL(da_tensor);
    MS_LOG(INFO) << "Get DATensor: " << da_tensor << " for value: " << value.get() << ", " << value->ToString()
                 << " from HostValueStore";
    const_map_[value_node] = da_tensor;
    graph_executor_.AddTensor(da_tensor);
    (void)converted_values_.emplace(value);
    return;
  }

  if (DeviceTensorStore::GetInstance().HasValue(value)) {
    auto da_tensor = DeviceTensorStore::GetInstance().Get(value);
    MS_EXCEPTION_IF_NULL(da_tensor);
    MS_LOG(INFO) << "Get DATensor: " << da_tensor << " for value: " << value.get() << ", " << value->ToString()
                 << " from DeviceTensorStore";
    const_map_[value_node] = da_tensor;
    graph_executor_.AddTensor(da_tensor);
    (void)converted_values_.emplace(value);
    return;
  }

  auto da_tensor = graph_executor_.AddTensor();
  // Set tensor type and shape
  SetDATensorTypeAndShape(da_tensor, value);
  const_map_[value_node] = da_tensor;
  MS_LOG(INFO) << "Convert value to DATensor: " << da_tensor << ", value: " << value.get() << ", " << value->ToString();

  if (da_tensor->type == da::tensor::Type_Monad || da_tensor->type == da::tensor::Type_None) {
    return;
  }

  // malloc for all parameters and valuenodes and copy them to device
  da_tensor->data = PrepareData(da_tensor, value);
  // save the value in converted_values_ to keep data from being released
  (void)converted_values_.emplace(value);
}

void *GraphAdapter::PrepareData(da::tensor::DATensor *da_value, const ValuePtr &value) {
  if (value->isa<tensor::Tensor>()) {
    da_value->tensorType = da::tensor::TensorType::DEVICE_TENSOR;
    DeviceTensorStore::GetInstance().Insert(value, da_value);
    return PrepareTensorDataToDevice(value->cast<tensor::TensorPtr>());
  } else if (value->isa<ValueSequence>() || value->isa<Scalar>() || value->isa<StringImm>()) {
    da_value->tensorType = da::tensor::TensorType::HOST_TENSOR;
    HostValueStore::GetInstance().Insert(da_value, value);
    auto kernel_tensor_value = ConvertValueToKernelTensorValue(value);
    MS_EXCEPTION_IF_NULL(kernel_tensor_value);
    (void)converted_values_.emplace(kernel_tensor_value);
    MS_LOG(INFO) << "Create ktvalue for DATensor: " << da_value << ", data_ptr: " << kernel_tensor_value->GetDataPtr();
    return const_cast<void *>(kernel_tensor_value->GetDataPtr());
  } else {
    MS_LOG(EXCEPTION) << "Unsupported value: " << value->ToString();
  }
}

void *GraphAdapter::PrepareTensorDataToDevice(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(device_context_);

  MS_LOG(INFO) << "start prepare tensor value: " << tensor->ToString();

  // tensor already prepared
  auto device_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  if (device_address != nullptr && device_address->IsPtrValid()) {
    MS_LOG(WARNING) << "tensor already has device address: " << tensor->ToString();
    return nullptr;
  }

  // malloc device memory for tensor
  device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
  auto device_ptr = device_context_->device_res_manager_->AllocateMemory(tensor->Size(), kDefaultStreamIndex);
  MS_EXCEPTION_IF_NULL(device_ptr);

  // copy tensor data from host to device
  if (!device_context_->device_res_manager_->Copy(device_ptr, tensor->data_c(), static_cast<uint64_t>(tensor->Size()),
                                                  device::CopyType::kH2D, kDefaultStreamIndex)) {
    MS_LOG(EXCEPTION) << "Copy tensor data failed for tensor: " << tensor->ToString();
  }

  MS_LOG(INFO) << "end prepare tensor value: " << tensor->ToString();

  return device_ptr;
}

da::tensor::DATensor *GraphAdapter::GetNodeDATensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);

  if (node->isa<ValueNode>()) {
    auto iter = const_map_.find(node);
    if (iter == const_map_.end()) {
      auto value_node = node->cast<ValueNodePtr>();
      ConvertValueNode(value_node);
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

  SetupFrontendParameterMapping();

  // parameters DATensor should be created before BeginGraph, added as parameters after BeginGraph
  ConvertParameters();

  graph_executor_.BeginGraph(func_graph_->ToString());
  InsertParameters();
  ConvertCNodes();
  graph_executor_.EndGraph();

  graph_executor_.DumpGraph();
}

void GraphAdapter::RunGraph(const VectorRef &inputs, VectorRef *outputs) {
  MS_LOG(INFO) << "Run graph: " << func_graph_->ToString();

  if (AnfAlgo::IsGraphOutputValueNodeOrParameter(func_graph_->output(), inputs, outputs)) {
    return;
  }

  ConvertInputs(inputs);
  graph_executor_.SetFreeFunc([this](void *data) {
    MS_EXCEPTION_IF_NULL(device_context_);
    MS_EXCEPTION_IF_NULL(data);
    device_context_->device_res_manager_->FreeMemory(data);
  });
  graph_executor_.RunGraph();
  ConvertOutputs(outputs);

  auto &llm_manger = LLMManager::GetInstance();
  llm_manger.reset_graph_inputs();
}

void GraphAdapter::SetupFrontendParameterMapping() {
  const auto &backend_params = func_graph_->input_nodes();
  for (size_t j = 0; j < backend_params.size(); j++) {
    const auto &backend_param = backend_params[j];
    MS_EXCEPTION_IF_NULL(backend_param);

    auto frontend_param_with_index = func_graph_->GetElementInTupleBackendFrontIndexMap(backend_param);
    if (frontend_param_with_index.first == nullptr) {
      frontend_param_with_index = {AnfAlgo::FetchFrontNodeByBackendNode(backend_param, *func_graph_), 0};
    }
    MS_EXCEPTION_IF_NULL(frontend_param_with_index.first);

    (void)frontend_params_to_backend_params_[frontend_param_with_index.first].emplace_back(
      std::make_pair(frontend_param_with_index.second, backend_param));
  }
}

void GraphAdapter::ConvertInputs(const VectorRef &inputs) {
  const auto &frontend_params = func_graph_->GetFuncGraph()->parameters();
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == frontend_params.size(),
                             "The inputs size is not equal to graph frontend params size.");

  auto &llm_manger = LLMManager::GetInstance();
  llm_manger.Clear();

  for (size_t i = 0; i < inputs.size(); ++i) {
    // flatten input tensors
    std::vector<tensor::TensorPtr> flatten_input_tensors;
    AnfAlgo::FlattenInputArg(inputs[i], frontend_params[i], &flatten_input_tensors);

    // find backend params
    auto frontend_param = frontend_params[i];
    auto iter = frontend_params_to_backend_params_.find(frontend_param);
    if (iter == frontend_params_to_backend_params_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Can not find the frontend parameters: " << frontend_param->fullname_with_scope();
      continue;
    }
    auto backend_params = iter->second;

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
      auto input_tensor = flatten_input_tensors[input_tensor_index];

      // get da_param from parameter_map_
      auto backend_param = frontend_index_to_backend_param.second;
      auto iter = parameter_map_.find(backend_param);
      if (iter == parameter_map_.end()) {
        MS_LOG(INTERNAL_EXCEPTION) << "Can not find parameter '" << backend_param->ToString() << "' in parameter_map_";
      }
      auto da_param = iter->second;

      auto is_weight = common::AnfAlgo::IsParameterWeight(backend_param->cast<ParameterPtr>());
      if (!is_weight) {
        llm_manger.add_graph_input(backend_param->fullname_with_scope(), input_tensor->data_ptr());
        MS_LOG(INFO) << "Record input tensor: " << input_tensor->ToString()
                     << "for parameter: " << backend_param->fullname_with_scope();
      }

      if (da_param->tensorType == da::tensor::TensorType::DEVICE_TENSOR && !is_weight) {
        // free input device memory before new inputs come in
        MS_EXCEPTION_IF_NULL(da_param->data);
        device_context_->device_res_manager_->FreeMemory(da_param->data);
        da_param->data = nullptr;
      }

      if (is_weight && DeviceTensorStore::GetInstance().HasValue(input_tensor)) {
        da_param->data = DeviceTensorStore::GetInstance().Get(input_tensor)->data;
        continue;
      }

      da_param->data = PrepareData(da_param, input_tensor);
    }
  }
}

void GraphAdapter::ConvertOutputs(VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
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

    size_t tensor_size = GetTypeByte(TypeIdToType(dtype)) * SizeOf(shape);
    char *buffer = new char[tensor_size];
    // copy tensor data from device to host
    device_context_->device_res_manager_->BindDeviceToCurrentThread(false);
    if (!device_context_->device_res_manager_->Copy(buffer, da_tensor->data, static_cast<uint64_t>(tensor_size),
                                                    device::CopyType::kD2H, kDefaultStreamIndex)) {
      MS_LOG(EXCEPTION) << "Copy da_tensor data failed, tensor_size: " << tensor_size;
    }
    auto output = std::make_shared<tensor::Tensor>(dtype, shape, buffer, dtype);
    MS_EXCEPTION_IF_NULL(output);
    MS_LOG(INFO) << "converted output tensor: " << output->ToString();
    delete[] buffer;
    (void)outputs->emplace_back(output);
  }

  MS_LOG(INFO) << "end convert outputs";
}

void GraphAdapter::ConvertParameters() {
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
  SetNodeOutputType(da_cnode, node);
  apply_map_[node] = da_cnode;
}

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
