/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "ir/core_ops_name.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_n.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_name_s.h"

#include "dalang/dair/ops/ops_name.h"

#include "backend/ms_infer_backend/graph_adapter.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

static std::map<std::string, da::ops::Op> primitive_op_map = {{ops::kNameAdd, da::ops::Op_add},
                                                            {ops::kNameSub, da::ops::Op_sub},
                                                            {ops::kNameMul, da::ops::Op_mul},
                                                            {ops::kNameDiv, da::ops::Op_div},
                                                            {ops::kNameMatMul, da::ops::Op_matmul},
                                                            {ops::kNameNorm, da::ops::Op_norm},
                                                            {ops::kNameReLU, da::ops::Op_relu},
                                                            {ops::kNameGeLU, da::ops::Op_gelu},
                                                            {ops::kNameSoftmax, da::ops::Op_softmax}};

static da::ops::Op ConvertPrimitiveOp(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);

  auto iter = primitive_op_map.find(prim->name());
  if (iter != primitive_op_map.end()) {
    return (*iter).second;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unexpected Primitive " << prim->name();
  }
}

static std::map<TypeId, da::tensor::Type> number_data_type_map = {{kNumberTypeInt16, da::tensor::Type_I16},
                                                                {kNumberTypeInt32, da::tensor::Type_I32},
                                                                {kNumberTypeInt64, da::tensor::Type_I64},
                                                                {kNumberTypeFloat16, da::tensor::Type_F16},
                                                                {kNumberTypeFloat32, da::tensor::Type_F32},
                                                                {kNumberTypeFloat64, da::tensor::Type_F64},
                                                                {kNumberTypeBFloat16, da::tensor::Type_BF16}};

static da::tensor::Type ConvertNumberDataType(const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);

  auto iter = number_data_type_map.find(type->type_id());
  if (iter != number_data_type_map.end()) {
    return (*iter).second;
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unexpected Type " << type->type_name();
  }
}

static void SetTensorType(const TypePtr &dtype, const ShapeVector &shape, da::tensor::DATensor *tensor) {
  MS_EXCEPTION_IF_NULL(dtype);
  MS_EXCEPTION_IF_NULL(tensor);

  tensor->type = ConvertNumberDataType(dtype);
  // TODO: Convert shape
}

static void ConvertTensor(const tensor::TensorPtr &tensor, da::tensor::DATensor *da_value) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(da_value);

  SetTensorType(tensor->Dtype(), tensor->shape(), da_value);
  // TODO: Convert data
}

static void SetValue(const BaseRef &val, da::tensor::DATensor *da_value) {
  MS_EXCEPTION_IF_NULL(val);
  MS_EXCEPTION_IF_NULL(da_value);

  if (utils::isa<tensor::Tensor>(val)) {
    tensor::TensorPtr tensor_ptr = utils::cast<tensor::TensorPtr>(val);
    ConvertTensor(tensor_ptr, da_value);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unsupported type " << val.ToString();
  }
}

static void SetNodeOutputType(const AnfNodePtr &node, da::tensor::DATensor *tensor) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(tensor);

  const TypePtr &type = node->Type();
  MS_EXCEPTION_IF_NULL(type);
  const BaseShapePtr &shape = node->Shape();
  MS_EXCEPTION_IF_NULL(shape);

  if (type->isa<TensorType>()) {
    SetTensorType(dyn_cast<TensorType>(type)->element(), shape->GetShapeVector(), tensor);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Unsupported type: " << type->type_name();
  }
}

void GraphAdapter::ConvertGraph() {
  // parameters DATensor should be created before BeginGraph, added as parameters after BeginGraph
  ConvertParameters();

  graph_executor_.BeginGraph(func_graph_->ToString());
  InsertParameters();
  ConvertCNodes();
  graph_executor_.EndGraph();

  graph_executor_.DumpGraph();
}

void GraphAdapter::RunGraph(const VectorRef &inputs, VectorRef *outputs) {
  if (AnfAlgo::IsGraphOutputValueNodeOrParameter(func_graph_->output(), inputs, outputs)) {
    return;
  }

  ConvertInputs(inputs);
  graph_executor_.RunGraph();
  ConvertOutputs(outputs);
}

void GraphAdapter::ConvertInputs(const VectorRef &inputs) {
  const auto &params = func_graph_->parameters();
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == params.size(),
                             "The inputs size is not equal to graph params size.");

  for (size_t i = 0; i < inputs.size(); ++i) {
    auto iter = parameter_map_.find(params[i]);
    if (iter == parameter_map_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Can not find parameter '" << params[i]->ToString()
                                 << "' in parameter_map_";
    }
    da::tensor::DATensor *da_param = iter->second;
    SetValue(inputs[i], da_param);
  }
}

void GraphAdapter::ConvertOutputs(VectorRef *outputs)  {
  // TODO: Convert outputs
}

void GraphAdapter::ConvertParameters() {
  for (auto &param : func_graph_->parameters()) {
    const ParameterPtr param_ptr = dyn_cast<Parameter>(param);
    MS_EXCEPTION_IF_NULL(param_ptr);

    da::tensor::DATensor *da_param = graph_executor_.AddTensor();
    SetNodeOutputType(param, da_param);
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
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode != func_graph_->get_return()) { // discard return node
      ConvertCNode(cnode);
    }
  }
}

void GraphAdapter::ConvertCNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);

  auto &inputs = node->inputs();
  if (inputs.size() < 1) {
    MS_LOG(INTERNAL_EXCEPTION) << "Inputs of CNode is empty" << node->ToString();
  }

  // process Op
  AnfNodePtr op = inputs[0];
  if (!IsValueNode<Primitive>(op)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Operator must be a primitive" << node->ToString();
  }
  da::ops::Op da_op = ConvertPrimitiveOp(GetValueNode<PrimitivePtr>(op));

  // process Inputs
  std::vector<da::tensor::DATensor *> da_inputs;
  for (size_t i = 1; i < inputs.size(); ++i) {
    (void)da_inputs.emplace_back(GetNodeDATensor(inputs[i]));
  }

  da::tensor::DATensor *da_cnode = graph_executor_.AddTensor(da_op, da_inputs);
  SetNodeOutputType(node, da_cnode);
  apply_map_[node] = da_cnode;
}

da::tensor::DATensor * GraphAdapter::GetNodeDATensor(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);

  if (node->isa<ValueNode>()) {
    auto iter = const_map_.find(node);
    if (iter == const_map_.end()) {
      da::tensor::DATensor *da_value = graph_executor_.AddTensor();
      SetValue(GetValueNode(node), da_value);
      const_map_[node] = da_value;
    }
    return const_map_[node];
  }

  if (node->isa<CNode>()) {
    auto iter = apply_map_.find(node);
    if (iter == apply_map_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Can not find node '" << node->ToString() << "' in apply_map_";
    }
    return iter->second;
  }

  if (node->isa<Parameter>()) {
    auto iter = parameter_map_.find(node);
    if (iter == parameter_map_.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Can not find node '" << node->ToString() << "' in parameter_map_";
    }
    return iter->second;
  }

  MS_LOG(INTERNAL_EXCEPTION) << "Unknown node type. node is '" << node->ToString() << "'";
}

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
