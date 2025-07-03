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
#include "ir/core_ops_name.h"
#include "utils/shape_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "plugin/device/ascend/hal/hardware/ascend_device_res_manager.h"

#include "backend/ms_infer_backend/graph_adapter.h"
#include "backend/ms_infer_backend/host_value_store.h"
#include "backend/ms_infer_backend/utils.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

static TypePtr GetSequenceElementType(const ValueSequencePtr &value_seq) {
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

void GraphAdapter::SetValue(da::tensor::DATensor *da_value, const BaseRef &val) {
  MS_EXCEPTION_IF_NULL(val);
  MS_EXCEPTION_IF_NULL(da_value);

  if (!utils::isa<ValuePtr>(val)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Not a value: " << val.ToString();
  }
  ValuePtr value = utils::cast<ValuePtr>(val);
  MS_EXCEPTION_IF_NULL(value);
  MS_LOG(DEBUG) << "Set value to DATensor: " << value->ToString();

  // Set tensor type
  auto shape = ShapeVector();
  TypePtr dtype = nullptr;
  if (utils::isa<Scalar>(value)) {
    dtype = value->type();
  } else if (utils::isa<Monad>(value)) {
    da_value->type = da::tensor::Type_Monad;
    return;
  } else if (utils::isa<ValueSequence>(value)) {
    auto value_seq = utils::cast<ValueSequencePtr>(value);
    dtype = GetSequenceElementType(value_seq);
    (void)shape.emplace_back(value_seq->value().size());
  } else if (utils::isa<tensor::Tensor>(value)) {
    auto tensor_ptr = utils::cast<tensor::TensorPtr>(value);
    dtype = tensor_ptr->Dtype();
    shape = tensor_ptr->shape();
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unsupported type " << val.ToString();
  }
  da_value->type = ConvertDataType(dtype);
  SetTensorShape(da_value, shape);

  // malloc for all parameters and valuenodes and copy them to device
  da_value->data = PrepareData(da_value, value);
  // save the value in converted_values_ to keep data from being released
  (void)converted_values_.emplace(value);
}

void *GraphAdapter::PrepareData(da::tensor::DATensor *da_value, const ValuePtr &value) {
  if (value->isa<tensor::Tensor>()) {
    da_value->tensorType = da::tensor::TensorType::DEVICE_TENSOR;
    return PrepareTensorDataToDevice(value->cast<tensor::TensorPtr>());
  } else if (value->isa<ValueSequence>() || value->isa<Scalar>() || value->isa<StringImm>()) {
    da_value->tensorType = da::tensor::TensorType::HOST_TENSOR;
    HostValueStore::GetInstance().Insert(da_value, value);
    return value.get();
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

static void SetNodeOutputType(da::tensor::DATensor *tensor, const AnfNodePtr &node) {
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
    tensor->type = da::tensor::Type_Tuple;
  } else if (type->isa<MonadType>()) {
    tensor->type = da::tensor::Type_Monad;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unsupported type: " << type->type_name();
  }
}

da::tensor::DATensor *GraphAdapter::GetNodeDATensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);

  if (node->isa<ValueNode>()) {
    auto iter = const_map_.find(node);
    if (iter == const_map_.end()) {
      da::tensor::DATensor *da_value = graph_executor_.AddTensor();
      SetValue(da_value, GetValueNode(node));
      const_map_[node] = da_value;
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
  graph_executor_.RunGraph();
  ConvertOutputs(outputs);
}

void GraphAdapter::ConvertInputs(const VectorRef &inputs) {
  const auto &params = func_graph_->parameters();
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == params.size(), "The inputs size is not equal to graph params size.");

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto iter = parameter_map_.find(params[i]);
    if (iter == parameter_map_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Can not find parameter '" << params[i]->ToString() << "' in parameter_map_";
    }
    auto da_param = iter->second;
    if (da_param->tensorType == da::tensor::TensorType::DEVICE_TENSOR) {
      // free input device memory before new inputs come in
      MS_EXCEPTION_IF_NULL(da_param->data);
      device_context_->device_res_manager_->FreeMemory(da_param->data);
      da_param->data = nullptr;
    }
    SetValue(da_param, inputs[i]);
  }
}

void GraphAdapter::ConvertOutputs(VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  MS_LOG(INFO) << "start convert outputs";

  std::vector<da::tensor::DATensor *> output_da_tensors;
  MS_LOG(INFO) << "start AppendNodeOutputs, node: " << func_graph_->get_return()->fullname_with_scope();
  graph_executor_.AppendNodeOutputs(output_da_tensors, GetNodeDATensor(func_graph_->get_return()));

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
