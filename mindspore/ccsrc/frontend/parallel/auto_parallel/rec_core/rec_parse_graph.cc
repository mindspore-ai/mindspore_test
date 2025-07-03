/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/auto_parallel/rec_core/rec_parse_graph.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ir/value.h"
#include "ops_utils/op_constants.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_tensor.h"
#include "frontend/parallel/ops_info/operator_info.h"

#include "frontend/parallel/ops_info/transpose_info.h"

namespace mindspore {
namespace parallel {
const TensorParam MakeTensor(int64_t n, int64_t c, int64_t h, int64_t w) {
  TensorParam new_tensor;
  new_tensor.tensor_type = kFloat32;
  new_tensor.tensor_shape.shape_n = n;
  new_tensor.tensor_shape.shape_c = c;
  new_tensor.tensor_shape.shape_h = h;
  new_tensor.tensor_shape.shape_w = w;
  const TensorParam &tensor = new_tensor;
  return tensor;
}

void UpdateStrategy(const int64_t n, const int64_t c, const int64_t h, const int64_t w, TensorParam *tensor_param) {
  MS_EXCEPTION_IF_NULL(tensor_param);
  if (n == 0 || c == 0 || h == 0 || w == 0) {
    MS_LOG(WARNING) << "input shape should not be 0";
    return;
  }

  tensor_param->tensor_str.str_n = 1.0 / n;
  tensor_param->tensor_str.str_c = 1.0 / c;
  tensor_param->tensor_str.str_h = 1.0 / h;
  tensor_param->tensor_str.str_w = 1.0 / w;
}

std::vector<int64_t> ReshapeDecompose(std::vector<int64_t> input_shape, std::vector<int64_t> output_shape) {
  std::vector<int64_t> dependency(output_shape.size(), INT_MAX);
  size_t input_index = 0;
  size_t output_index = 0;
  while (input_index < input_shape.size() && output_index < output_shape.size()) {
    if (input_shape[input_index] >= output_shape[output_index]) {
      if (output_shape[output_index] != 0 && (input_shape[input_index] % output_shape[output_index] == 0)) {
        input_shape[input_index] /= output_shape[output_index];
        output_shape[output_index] = 1;
        dependency[output_index] = SizeToLong(input_index);
      }
      output_index++;
      if (input_shape[input_index] == 1) {
        input_index++;
      }
    } else {
      input_index++;
    }
  }

  if (std::any_of(output_shape.begin(), output_shape.end(), [](int64_t shape) { return shape != 1; })) {
    return std::vector<int64_t>{};
  }
  if (std::any_of(dependency.begin(), dependency.end(), [](int64_t shape) { return shape == INT_MAX; })) {
    return std::vector<int64_t>{};
  }

  return dependency;
}

std::vector<std::vector<int64_t>> ReshapeDecomToCombine(const std::vector<int64_t> &dependency) {
  std::vector<std::vector<int64_t>> combine;
  if (dependency.empty()) {
    return combine;
  }
  auto dependency_index = 0;
  auto output_index = 0;
  for (auto combine_index = 0; combine_index < REC_NODE_DIMS_SIZE; combine_index++) {
    std::vector<int64_t> tmp;
    while (combine_index == dependency[dependency_index]) {
      while (combine_index == dependency[dependency_index]) {
        tmp.push_back(output_index);
        output_index++;
        dependency_index++;
      }
    }
    combine.push_back(tmp);
  }
  return combine;
}

std::vector<int64_t> complete_shape_to_4D(const std::vector<int64_t> &shape) {
  std::vector<int64_t> completed(SIZE_FOUR, 1);
  size_t shape_size = shape.size();
  size_t completed_size = completed.size();
  size_t elements_to_copy = (shape_size > completed_size) ? completed_size : shape_size;
  for (size_t i = 0; i < elements_to_copy; ++i) {
    completed[completed_size - elements_to_copy + i] = shape[shape_size - elements_to_copy + i];
  }
  return completed;
}

void HandleShapeRelatedOp(Graph::NodeType *new_op, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                          size_t iter_ops) {
  if (ops.size() <= iter_ops) {
    MS_LOG(EXCEPTION) << "iter_ops out of range";
  }
  MS_EXCEPTION_IF_NULL(ops[iter_ops]);

  if (ops[iter_ops]->type() == TRANSPOSE) {
    auto transpose = std::static_pointer_cast<TransposeInfo>(ops[iter_ops]);
    new_op->transpose_mapping = transpose->axis_v();
    MS_LOG(INFO) << "The perturbation information of TRANSPOSE is : " << new_op->transpose_mapping;
  }

  if (ops[iter_ops]->type() == RESHAPE) {
    auto input_size = ops[iter_ops]->inputs_shape()[0].size();
    auto output_size = ops[iter_ops]->outputs_shape()[0].size();
    auto input_shape = complete_shape_to_4D(ops[iter_ops]->inputs_shape()[0]);
    auto output_shape = complete_shape_to_4D(ops[iter_ops]->outputs_shape()[0]);

    std::vector<std::vector<int64_t>> reshape_op;
    std::vector<int64_t> decom;
    if (input_size <= output_size) {
      decom = ReshapeDecompose(input_shape, output_shape);
      for (size_t i = 0; i < decom.size(); i++) {
        std::vector<int64_t> tmp;
        tmp.push_back(decom[i]);
        reshape_op.push_back(tmp);
      }
    } else {
      decom = ReshapeDecompose(output_shape, input_shape);
      reshape_op = ReshapeDecomToCombine(decom);
    }

    new_op->reshape_mapping = reshape_op;
    MS_LOG(INFO) << "The dimension mapping information of RESHAPE is : " << new_op->reshape_mapping;
  }
}

void SetDefaultOutStrategy(Graph::NodeType *new_op) {
  MS_EXCEPTION_IF_NULL(new_op);
  if (new_op->apply.op_type == OperatorType::kRecMatMul) {
    new_op->tensor_parm.tensor_str.str_h = new_op->apply.arguments[0].tensor_str.str_h;
    new_op->tensor_parm.tensor_str.str_w = new_op->apply.arguments[1].tensor_str.str_w;
    MS_LOG(INFO) << "The user does not define the output strategy for " << new_op->name
                 << ". The output strategy should be [" << new_op->tensor_parm.tensor_str.str_n << ","
                 << new_op->tensor_parm.tensor_str.str_c << "," << new_op->tensor_parm.tensor_str.str_h << ","
                 << new_op->tensor_parm.tensor_str.str_w;
  } else {
    new_op->tensor_parm.tensor_str.str_n = new_op->apply.arguments[0].tensor_str.str_n;
    new_op->tensor_parm.tensor_str.str_c = new_op->apply.arguments[0].tensor_str.str_c;
    new_op->tensor_parm.tensor_str.str_h = new_op->apply.arguments[0].tensor_str.str_h;
    new_op->tensor_parm.tensor_str.str_w = new_op->apply.arguments[0].tensor_str.str_w;
    MS_LOG(INFO) << "The user does not define the output strategy for " << new_op->name
                 << ". The default output strategy is [" << new_op->tensor_parm.tensor_str.str_n << ","
                 << new_op->tensor_parm.tensor_str.str_c << "," << new_op->tensor_parm.tensor_str.str_h << ","
                 << new_op->tensor_parm.tensor_str.str_w;
  }
}

void HandleMatMulTranspose(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t iter_ops,
                           const Graph::NodeType &new_op, size_t idx, std::vector<int64_t> *nchw) {
  if (nchw == nullptr) {
    MS_LOG(EXCEPTION) << "Input nchw must not be nullptr";
    return;
  }

  if (new_op.apply.op_type == OperatorType::kRecMatMul) {
    auto input_value = ops[iter_ops]->input_value();
    MS_EXCEPTION_IF_NULL(input_value[kIndex2]->cast<BoolImmPtr>());
    bool transpose_a = input_value[2]->cast<BoolImmPtr>()->value();
    MS_EXCEPTION_IF_NULL(input_value[kIndex3]->cast<BoolImmPtr>());
    bool transpose_b = input_value[3]->cast<BoolImmPtr>()->value();
    if (nchw->size() != SIZE_FOUR) {
      MS_LOG(ERROR) << "The length of nchw must be 4";
      return;
    }

    if (transpose_a && idx == 0) {
      MS_LOG(INFO) << "The transpose_a attribute is found.";
      auto tmp = nchw->at(INDEX_TWO);
      nchw->at(INDEX_TWO) = nchw->at(INDEX_THREE);
      nchw->at(INDEX_THREE) = tmp;
    }
    if (transpose_b && idx == 1) {
      MS_LOG(INFO) << "The transpose_b attribute is found.";
      auto tmp = nchw->at(INDEX_TWO);
      nchw->at(INDEX_TWO) = nchw->at(INDEX_THREE);
      nchw->at(INDEX_THREE) = tmp;
    }
  }
}

OperatorType GetOperatorType(const std::shared_ptr<OperatorInfo> &op) {
  auto pos = op->name().find("Info");
  auto name = op->name().substr(0, pos);
  auto op_type = op->type();
  auto idx = DictOpType.find(op_type);
  if (idx != DictOpType.end()) {
    return DictOpType.at(op_type);
  } else if (name == STAND_ALONE) {
    MS_LOG(INFO) << op->type() << ": standalone operator.";
    return OperatorType::kRecStandAlone;
  } else if (name == BATCH_PARALLEL) {
    MS_LOG(INFO) << op->type() << ": batch parallel operator.";
    return OperatorType::kRecBatchParallel;
  } else {
    MS_LOG(INFO) << op->name() << ": Unknown operator type " << op_type;
    return OperatorType::kRecUnknownType;
  }
}

TensorParam SetTensorParam(const std::vector<Shape> &output_shape) {
  if (output_shape.empty()) {
    MS_LOG(EXCEPTION) << "outputs shape is empty : " << output_shape;
  } else {
    size_t shape_size = output_shape[0].size();
    switch (shape_size) {
      case SIZE_FOUR:
        return MakeTensor(output_shape[0][INDEX_ZERO], output_shape[0][INDEX_ONE], output_shape[0][INDEX_TWO],
                          output_shape[0][INDEX_THREE]);
      case SIZE_THREE:
        return MakeTensor(1, output_shape[0][INDEX_ZERO], output_shape[0][INDEX_ONE], output_shape[0][INDEX_TWO]);
      case SIZE_TWO:
        return MakeTensor(1, 1, output_shape[0][INDEX_ZERO], output_shape[0][INDEX_ONE]);
      case SIZE_ONE:
        return MakeTensor(1, 1, 1, output_shape[0][INDEX_ZERO]);
      case SIZE_ZERO:
        return MakeTensor(1, 1, 1, 1);
      default:
        MS_LOG(WARNING) << "Output tensor shape is unexpected, return default tensor shape (1,1,1,1)";
        return MakeTensor(1, 1, 1, 1);
    }
  }
}

Graph::NodeType MakeNewOperator(const std::vector<std::shared_ptr<OperatorInfo>> &ops, size_t iter_ops) {
  MS_LOG(INFO) << "Creating Node " << ops[iter_ops]->name() << "...";
  Graph::NodeType NewOp;
  NewOp.name = ops[iter_ops]->name();
  NewOp.info = InfoType::kApplication;
  NewOp.apply.op_type = GetOperatorType(ops[iter_ops]);
  NewOp.tensor_parm = SetTensorParam(ops[iter_ops]->outputs_shape());

  CompleteOperatorInputs(ops, iter_ops, &NewOp);

  auto prim_anf_node = GetValueNode<PrimitivePtr>(ops[iter_ops]->cnode()->input(0));
  auto sapp_env = getenv(INTERFERED_SAPP);
  if (sapp_env != nullptr && std::string(sapp_env) == "1") {
    MS_LOG(INFO) << "environment variable MS_INTERFERED_SAPP is set.";
    auto out_stra = prim_anf_node->GetAttr(OUT_STRATEGY);
    if (out_stra != nullptr) {
      NewOp.interfered_sapp = true;
      auto out_stra_var = GetValue<std::vector<Shape>>(out_stra);
      if (out_stra_var.size() > 1) {
        MS_LOG(WARNING) << "operator should have only one output";
      }
      Dimensions sub_out_stra = out_stra_var.at(0);
      std::vector<int64_t> nchw(sub_out_stra);
      if (sub_out_stra.size() > SIZE_FOUR) {
        MS_LOG(EXCEPTION) << "Operator: " << NewOp.name << "'s output strategy has more than 4 dimensions, "
                          << "which is not supported by recursive_programming";
      }
      for (size_t i = 0; i < SIZE_FOUR - sub_out_stra.size(); i++) {
        nchw.insert(nchw.begin(), 1);
      }
      UpdateStrategy(nchw[INDEX_ZERO], nchw[INDEX_ONE], nchw[INDEX_TWO], nchw[INDEX_THREE], &NewOp.tensor_parm);
      MS_LOG(INFO) << "user-defined out strategy: [" << NewOp.tensor_parm.tensor_str.str_n << ","
                   << NewOp.tensor_parm.tensor_str.str_c << "," << NewOp.tensor_parm.tensor_str.str_h << ","
                   << NewOp.tensor_parm.tensor_str.str_w << "]";
    }

    auto in_stra = prim_anf_node->GetAttr(IN_STRATEGY);
    if (in_stra != nullptr) {
      NewOp.interfered_sapp = true;
      auto in_stra_var = GetValue<std::vector<Shape>>(in_stra);
      for (size_t i = 0; i < in_stra_var.size(); i++) {
        Dimensions sub_in_stra = in_stra_var.at(i);
        std::vector<int64_t> nchw(sub_in_stra);
        if (sub_in_stra.size() > SIZE_FOUR) {
          MS_LOG(EXCEPTION) << "Operator: " << NewOp.name << "'s input strategy has more than 4 dimensions, "
                            << "which is not supported by recursive_programming";
        }
        for (size_t j = 0; j < SIZE_FOUR - sub_in_stra.size(); j++) {
          nchw.insert(nchw.begin(), 1);
        }
        HandleMatMulTranspose(ops, iter_ops, NewOp, i, &nchw);
        UpdateStrategy(nchw[INDEX_ZERO], nchw[INDEX_ONE], nchw[INDEX_TWO], nchw[INDEX_THREE],
                       &NewOp.apply.arguments[i]);
        MS_LOG(INFO) << "user-defined " << i << "th in strategy: [" << NewOp.apply.arguments[i].tensor_str.str_n << ","
                     << NewOp.apply.arguments[i].tensor_str.str_c << "," << NewOp.apply.arguments[i].tensor_str.str_h
                     << "," << NewOp.apply.arguments[i].tensor_str.str_w << "]";
      }

      // Infer the output strategy for an operator with a custom input strategy but no custom output strategy
      if (out_stra == nullptr) {
        SetDefaultOutStrategy(&NewOp);
      }
    }
  }
  HandleShapeRelatedOp(&NewOp, ops, iter_ops);
  MS_LOG(INFO) << "Node " << NewOp.name << " created successfully,"
               << " its input is " << ops[iter_ops]->inputs_shape() << " and its output is "
               << ops[iter_ops]->outputs_shape() << ".";
  return NewOp;
}

void CompleteOperatorInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                            Graph::NodeType *NewTensor) {
  size_t input_tensor_size = ops[iter_ops]->inputs_shape().size();
  if (ops[iter_ops]->type() == STACK) {
    input_tensor_size = 1;
  }
  if (input_tensor_size > MAX_INPUT_NUM) {
    MS_LOG(EXCEPTION) << ops[iter_ops]->name() << " input tensor " << input_tensor_size << " num exceeds limit("
                      << MAX_INPUT_NUM << ").";
  }

  for (size_t iter_input_tensors = 0; iter_input_tensors < input_tensor_size; iter_input_tensors++) {
    if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == SIZE_FOUR) {
      Complete4DInputs(ops, iter_ops, iter_input_tensors, NewTensor);
    } else if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == SIZE_THREE) {
      NewTensor->apply.arguments[iter_input_tensors] =
        MakeTensor(1, ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ZERO],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ONE],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_TWO]);
    } else if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == SIZE_TWO) {
      Complete2DInputs(ops, iter_ops, iter_input_tensors, NewTensor);
    } else if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == SIZE_ONE) {
      NewTensor->apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ZERO]);
    } else if (ops[iter_ops]->inputs_shape()[iter_input_tensors].size() == 0) {
      NewTensor->apply.arguments[iter_input_tensors] = MakeTensor(1, 1, 1, 1);
    } else {
      MS_LOG(WARNING) << ops[iter_ops]->name() << ": input tensor shape is unexpected.";
    }
  }
}

void Complete2DInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                      const size_t iter_input_tensors, Graph::NodeType *NewTensor) {
  if (NewTensor->apply.op_type == OperatorType::kRecMatMul) {
    auto input_value = ops[iter_ops]->input_value();
    bool transpose_a = input_value[2]->cast<BoolImmPtr>()->value();
    bool transpose_b = input_value[3]->cast<BoolImmPtr>()->value();
    if (transpose_a && (iter_input_tensors == 0)) {
      NewTensor->apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][1],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][0]);
    } else if (transpose_b && (iter_input_tensors == 1)) {
      NewTensor->apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][1],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][0]);
    } else {
      NewTensor->apply.arguments[iter_input_tensors] =
        MakeTensor(1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][0],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][1]);
    }
  } else {
    NewTensor->apply.arguments[iter_input_tensors] = MakeTensor(
      1, 1, ops[iter_ops]->inputs_shape()[iter_input_tensors][0], ops[iter_ops]->inputs_shape()[iter_input_tensors][1]);
  }
}

void Complete4DInputs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                      const size_t iter_input_tensors, Graph::NodeType *NewTensor) {
  if (NewTensor->apply.op_type == OperatorType::kRecBatchMatMul) {
    auto input_value = ops[iter_ops]->input_value();
    bool transpose_a = input_value[2]->cast<BoolImmPtr>()->value();
    bool transpose_b = input_value[3]->cast<BoolImmPtr>()->value();
    if (transpose_a && (iter_input_tensors == 0)) {
      NewTensor->apply.arguments[iter_input_tensors] =
        MakeTensor(ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ZERO],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ONE],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_THREE],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_TWO]);
    } else if (transpose_b && (iter_input_tensors == 1)) {
      NewTensor->apply.arguments[iter_input_tensors] =
        MakeTensor(ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ZERO],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ONE],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_THREE],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_TWO]);
    } else {
      NewTensor->apply.arguments[iter_input_tensors] =
        MakeTensor(ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ZERO],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ONE],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_TWO],
                   ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_THREE]);
    }
  } else {
    NewTensor->apply.arguments[iter_input_tensors] =
      MakeTensor(ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ZERO],
                 ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_ONE],
                 ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_TWO],
                 ops[iter_ops]->inputs_shape()[iter_input_tensors][INDEX_THREE]);
  }
}

std::shared_ptr<Graph> ParseGraph(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                  const std::vector<std::vector<std::string>> &input_tensor_names) {
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  constexpr size_t MAX_OP_NUM = SIZE_MAX / 2;
  if (ops.size() > MAX_OP_NUM) {
    MS_LOG(EXCEPTION) << "Total number of operators is bigger than " << MAX_OP_NUM;
  }

  for (size_t iter_ops = 0; iter_ops < ops.size(); iter_ops++) {
    Graph::NodeType NewOp = MakeNewOperator(ops, iter_ops);
    NewOp.param_name = ops[iter_ops]->get_involved_param_name();
    graph->nodes.push_back(NewOp);
  }
  MakeEdge(input_tensor_names, graph);

  return graph;
}

void MakeEdge(const std::vector<std::vector<std::string>> &input_tensor_names, const std::shared_ptr<Graph> &graph) {
  MS_LOG(INFO) << "Creating Edges ...";
  for (size_t iter_i = 0; iter_i < input_tensor_names.size(); iter_i++) {
    for (size_t iter_j = 1; iter_j < input_tensor_names[iter_i].size(); iter_j++) {
      size_t head_node_index = GetIndexInInputTensorNames(input_tensor_names, input_tensor_names[iter_i][iter_j]);
      if (head_node_index < SIZE_MAX / SIZE_TWO && head_node_index != iter_i) {
        graph->nodes[iter_i].node_in.push_back(head_node_index);
        NodeDep iter_node;
        iter_node.idx = iter_i;
        graph->nodes[head_node_index].node_out.push_back(iter_node);
        MS_LOG(INFO) << "Edge " << graph->nodes[head_node_index].name << "-" << graph->nodes[iter_i].name
                     << " created successfully.";
      }
    }
  }
}

size_t GetIndexInInputTensorNames(const std::vector<std::vector<std::string>> &input_tensor_name,
                                  const std::string &input_name) {
  for (size_t index = 0; index < input_tensor_name.size(); index++) {
    if (input_tensor_name[index][0] == input_name) {
      return index;
    }
  }
  MS_LOG(INFO) << "Get index failed, using SIZE_MAX instead";
  return SIZE_MAX;
}

bool IsTransposeRelatedOp(const Graph::NodeType &node) {
  if (!node.transpose_mapping.empty()) {
    MS_LOG(INFO) << "found a transpose related op when eliminating: " << node.name;
    return true;
  } else {
    return false;
  }
}

bool IsReshapeRelatedOp(const Graph::NodeType &node) {
  if (!node.reshape_mapping.empty()) {
    MS_LOG(INFO) << "found a reshape related op when eliminating: " << node.name;
    return true;
  } else {
    return false;
  }
}

NodeDep ShapeMappingCombine(const Graph::NodeType &node, NodeDep outgoing_index) {
  if (IsTransposeRelatedOp(node) || !outgoing_index.transpose_mapping.empty()) {
    std::vector<int64_t> updated = TransposeCombine(node.transpose_mapping, outgoing_index.transpose_mapping);
    outgoing_index.transpose_mapping = updated;
    MS_LOG(INFO) << "perturbation information (" << node.transpose_mapping << " and "
                 << outgoing_index.transpose_mapping << ") combined : " << outgoing_index.transpose_mapping;
  }

  if (IsReshapeRelatedOp(node) || !outgoing_index.reshape_mapping.empty()) {
    std::vector<std::vector<int64_t>> updated = ReshapeCombine(node.reshape_mapping, outgoing_index.reshape_mapping);
    outgoing_index.reshape_mapping = updated;
    MS_LOG(INFO) << "dimension mapping information (" << node.reshape_mapping << " and "
                 << outgoing_index.reshape_mapping << ") combined : " << outgoing_index.reshape_mapping;
  }
  return outgoing_index;
}

void Eliminate_Aux(size_t node_index, const std::shared_ptr<Graph> &graph,
                   const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(eli_list);
  std::vector<size_t> eli;
  eli.push_back(node_index);
  for (size_t i = 0; i < graph->nodes[node_index].node_out.size(); i++) {
    auto outgoing_node_idx = graph->nodes[node_index].node_out[i].idx;
    eli.push_back(outgoing_node_idx);
    if (!graph->nodes[node_index].param_name.empty() &&
        graph->nodes[node_index].apply.op_type == OperatorType::kRecCast &&
        (graph->nodes[outgoing_node_idx].apply.op_type == OperatorType::kRecMatMul ||
         graph->nodes[outgoing_node_idx].apply.op_type == OperatorType::kRecBatchMatMul)) {
      graph->nodes[outgoing_node_idx].param_name = graph->nodes[node_index].param_name;
    }
  }
  eli_list->push_back(eli);

  // Iterate over all input operators of the current node
  for (size_t i = 0; i < graph->nodes[node_index].node_in.size(); i++) {
    auto *incoming_outputs = &graph->nodes[graph->nodes[node_index].node_in[i]].node_out;
    auto it = find_if(incoming_outputs->begin(), incoming_outputs->end(),
                      [node_index](const NodeDep &out_node) { return out_node.idx == node_index; });
    if (it != incoming_outputs->end()) {
      it = incoming_outputs->erase(it);
      for (auto outgoing_index : graph->nodes[node_index].node_out) {
        it = find_if(incoming_outputs->begin(), incoming_outputs->end(),
                     [outgoing_index](const NodeDep &out_node) { return out_node.idx == outgoing_index.idx; });
        if (it == incoming_outputs->end()) {
          incoming_outputs->push_back(ShapeMappingCombine(graph->nodes[node_index], outgoing_index));
        }
      }
    }
  }

  // Iterate over all aux_input operators of the current node
  for (size_t i = 0; i < graph->nodes[node_index].node_in_aux.size(); i++) {
    auto *aux_incoming_outputs = &graph->nodes[graph->nodes[node_index].node_in_aux[i]].node_out;
    auto it = find_if(aux_incoming_outputs->begin(), aux_incoming_outputs->end(),
                      [node_index](const NodeDep &out_node) { return out_node.idx == node_index; });
    if (it != aux_incoming_outputs->end()) {
      it = aux_incoming_outputs->erase(it);
      for (auto outgoing_index : graph->nodes[node_index].node_out) {
        it = find_if(aux_incoming_outputs->begin(), aux_incoming_outputs->end(),
                     [outgoing_index](const NodeDep &out_node) { return out_node.idx == outgoing_index.idx; });
        if (it == aux_incoming_outputs->end()) {
          aux_incoming_outputs->push_back(ShapeMappingCombine(graph->nodes[node_index], outgoing_index));
        }
      }
    }
  }

  // Iterate over all output operators of the current node
  Eliminate_Aux_Outgoing(node_index, graph);
}

void EliminateAuxOutgoingInput(size_t node_index, const std::shared_ptr<Graph> &graph, size_t i) {
  MS_EXCEPTION_IF_NULL(graph);
  auto *outgoing_inputs = &graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in;
  MS_EXCEPTION_IF_NULL(outgoing_inputs);
  // Check if the current node is the input operator of the current node's output operator
  auto it = find(outgoing_inputs->begin(), outgoing_inputs->end(), node_index);
  if (it != outgoing_inputs->end()) {
    if (graph->nodes[node_index].node_in.size() > 0) {
      // If the current node has input operator, then add input[0] of the current node to the input of the current
      // node's output operator (if input[0] is also in the aux_input of the current node's output operator, then remove
      // it from the aux_input and keep it only in the input)
      auto exist_in_outgoing_auxinputs = find(
        graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.begin(),
        graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.end(), graph->nodes[node_index].node_in[0]);
      if (exist_in_outgoing_auxinputs != graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.end()) {
        size_t index_remove_node = LongToSize(std::distance(
          graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.begin(), exist_in_outgoing_auxinputs));
        if (graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux_idx.size() > index_remove_node) {
          (void)graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux_idx.erase(
            graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux_idx.begin() + index_remove_node);
        } else {
          MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_remove_node << ", out of range!";
        }
        if (graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.size() > index_remove_node) {
          (void)graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.erase(exist_in_outgoing_auxinputs);
        } else {
          MS_LOG(DEBUG) << "Trying to erase vector element at index " << index_remove_node
                        << ", which is out of range!";
        }
      }
      size_t idx = LongToSize(std::distance(outgoing_inputs->begin(), it));
      if (outgoing_inputs->size() > idx) {
        outgoing_inputs->at(idx) = graph->nodes[node_index].node_in[0];
      } else {
        MS_LOG(DEBUG) << "Trying to index vector element at index " << idx << ", out of range!";
      }
      // Then add the other input operators of the current node to the aux_input of the current node's output operator
      for (size_t j = 1; j < graph->nodes[node_index].node_in.size(); j++) {
        exist_in_outgoing_auxinputs = find(graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.begin(),
                                           graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.end(),
                                           graph->nodes[node_index].node_in[j]);
        if (exist_in_outgoing_auxinputs == graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.end()) {
          size_t index_aux = LongToSize(std::distance(outgoing_inputs->begin(), it));
          graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux_idx.push_back(index_aux);
          graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.push_back(
            graph->nodes[node_index].node_in[j]);
        }
      }
      // Then add all the operators in the aux_input of the current node to the aux_input of the output operator of the
      // current node
      for (size_t j = 0; j < graph->nodes[node_index].node_in_aux.size(); j++) {
        exist_in_outgoing_auxinputs = find(graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.begin(),
                                           graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.end(),
                                           graph->nodes[node_index].node_in_aux[j]);
        if (exist_in_outgoing_auxinputs == graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.end()) {
          size_t index_aux = LongToSize(std::distance(outgoing_inputs->begin(), it));
          graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux_idx.push_back(index_aux);
          graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux.push_back(
            graph->nodes[node_index].node_in_aux[j]);
        }
      }
    } else {
      auto idx = LongToSize(std::distance(outgoing_inputs->begin(), it));
      if (outgoing_inputs->size() > idx) {
        (void)outgoing_inputs->erase(it);
      } else {
        MS_LOG(DEBUG) << "Trying to erase vector element at index " << idx << ", out of range!";
      }
    }
  }
}

void EraseAuxInputAndIndex(std::vector<size_t> *aux_inputs, std::vector<size_t> *aux_input_indices,
                           std::vector<size_t>::iterator it, size_t index) {
  if (aux_inputs->size() > index) {
    (void)aux_inputs->erase(it);
  } else {
    MS_LOG(DEBUG) << "Trying to erase vector element at index " << index << ", out of range!";
  }

  if (aux_input_indices->size() > index) {
    (void)aux_input_indices->erase(aux_input_indices->begin() + index);
  } else {
    MS_LOG(DEBUG) << "Trying to erase vector element at index " << index << ", out of range!";
  }
}

void AddAuxInputToOutgoing(std::vector<size_t> *aux_inputs, std::vector<size_t> *aux_input_indices, size_t aux_input,
                           size_t index) {
  aux_inputs->push_back(aux_input);
  if (aux_input_indices->size() > index) {
    aux_input_indices->push_back(aux_input_indices->at(index));
  } else {
    MS_LOG(DEBUG) << "Trying to index vector element at index " << index << ", out of range!";
  }
}

void EliminateAuxOutgoingAuxInput(size_t node_index, const std::shared_ptr<Graph> &graph, size_t i) {
  MS_EXCEPTION_IF_NULL(graph);
  auto *outgoing_auxinputs = &graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux;
  MS_EXCEPTION_IF_NULL(outgoing_auxinputs);
  auto *outgoing_auxinputs_index = &graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in_aux_idx;
  // Check if the current node is the aux_input operator of the current node's output operator
  auto it = find(outgoing_auxinputs->begin(), outgoing_auxinputs->end(), node_index);
  if (it == outgoing_auxinputs->end()) {
    return;
  }
  size_t index_entree = LongToSize(std::distance(outgoing_auxinputs->begin(), it));
  if (!graph->nodes[node_index].node_in.empty()) {
    // If the current node has input operator, and if the input[0] of the current node is in
    // the input of the output operator of the current node, then delete it
    // from the aux_input of the output of the current node, otherwise add the input[0]
    // to the auxinput of the output of the current node
    auto exist_in_outgoing_inputs =
      find(graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in.begin(),
           graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in.end(), graph->nodes[node_index].node_in[0]);
    if (exist_in_outgoing_inputs != graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in.end()) {
      index_entree = LongToSize(std::distance(outgoing_auxinputs->begin(), it));
      EraseAuxInputAndIndex(outgoing_auxinputs, outgoing_auxinputs_index, it, index_entree);
    } else {
      size_t idx = LongToSize(std::distance(outgoing_auxinputs->begin(), it));
      if (outgoing_auxinputs->size() > idx) {
        outgoing_auxinputs->at(idx) = graph->nodes[node_index].node_in[0];
      } else {
        MS_LOG(DEBUG) << "Trying to index vector element at index " << idx << ", out of range!";
      }
      index_entree = LongToSize(std::distance(
        outgoing_auxinputs->begin(),
        find(outgoing_auxinputs->begin(), outgoing_auxinputs->end(), graph->nodes[node_index].node_in[0])));
    }
    // Determine whether the other input operator of the current node is in the input of the output operator,
    // and if not, add it to the aux_input of the output operator
    for (size_t j = 1; j < graph->nodes[node_index].node_in.size(); j++) {
      exist_in_outgoing_inputs =
        find(graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in.begin(),
             graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in.end(), graph->nodes[node_index].node_in[j]);
      if (exist_in_outgoing_inputs == graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in.end()) {
        AddAuxInputToOutgoing(outgoing_auxinputs, outgoing_auxinputs_index, graph->nodes[node_index].node_in[j],
                              index_entree);
      }
    }
    // Determine if the aux_input operator of the current node is in the input of the output operator,
    // and if not, add it to the aux_input of the output operator
    for (size_t j = 0; j < graph->nodes[node_index].node_in_aux.size(); j++) {
      exist_in_outgoing_inputs = find(graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in.begin(),
                                      graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in.end(),
                                      graph->nodes[node_index].node_in_aux[j]);
      if (exist_in_outgoing_inputs == graph->nodes[graph->nodes[node_index].node_out[i].idx].node_in.end()) {
        AddAuxInputToOutgoing(outgoing_auxinputs, outgoing_auxinputs_index, graph->nodes[node_index].node_in_aux[j],
                              index_entree);
      }
    }
  } else {
    EraseAuxInputAndIndex(outgoing_auxinputs, outgoing_auxinputs_index, it, index_entree);
  }
}

void Eliminate_Aux_Outgoing(size_t node_index, const std::shared_ptr<Graph> &graph) {
  for (size_t i = 0; i < graph->nodes[node_index].node_out.size(); i++) {
    // Handle the output operator connected to the current node via main edge
    EliminateAuxOutgoingInput(node_index, graph, i);
    // Handle the output operator connected to the current node via auxiliary edge
    EliminateAuxOutgoingAuxInput(node_index, graph, i);
  }
}

static void EraseEliminatedNode(std::vector<size_t> *nodes, const std::shared_ptr<std::vector<size_t>> &index_list) {
  for (size_t j = nodes->size(); j > 0; j--) {
    bool IsEliminated = (index_list->at(nodes->at(j - 1)) == SIZE_MAX);
    if (IsEliminated) {
      (void)nodes->erase(nodes->begin() + SizeToLong(j) - 1);
    } else {
      nodes->at(j - 1) = index_list->at(nodes->at(j - 1));
    }
  }
}

static void EraseEliminatedNode(std::vector<NodeDep> *nodes, const std::shared_ptr<std::vector<size_t>> &index_list) {
  MS_EXCEPTION_IF_NULL(nodes);
  for (size_t j = nodes->size(); j > 0; j--) {
    bool IsEliminated = (index_list->at(nodes->at(j - 1).idx) == SIZE_MAX);
    if (IsEliminated) {
      (void)nodes->erase(nodes->begin() + SizeToLong(j) - 1);
    } else {
      nodes->at(j - 1).idx = index_list->at(nodes->at(j - 1).idx);
    }
  }
}

std::shared_ptr<Graph> EliminateGraph(const std::shared_ptr<Graph> &graph,
                                      const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                                      const std::shared_ptr<std::vector<size_t>> &index_list,
                                      const bool dyn_shape_tmp_fix) {
  MS_EXCEPTION_IF_NULL(graph);
  for (size_t node_index = 0; node_index < graph->nodes.size(); node_index++) {
    auto type = graph->nodes[node_index].apply.op_type;
    bool interfered_sapp = graph->nodes[node_index].interfered_sapp;
    if (interfered_sapp == true) {
      MS_LOG(INFO) << "find a customized op when eliminating: " << graph->nodes[node_index].name;
    }
    if (dyn_shape_tmp_fix && type == OperatorType::kRecBatchMatMul) {
      continue;
    } else if (!interfered_sapp && EliminateOpType.find(type) != EliminateOpType.end()) {
      Eliminate_Aux(node_index, graph, eli_list);
    }
  }
  index_list->reserve(graph->nodes.size());
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    index_list->push_back(i);
  }
  for (size_t i = 0; i < eli_list->size(); i++) {
    if (eli_list->at(i)[0] >= index_list->size()) {
      MS_LOG(EXCEPTION) << "Failure: Operators' elements out of range.";
    }
    index_list->at(eli_list->at(i)[0]) = SIZE_MAX;
    for (size_t j = eli_list->at(i)[0] + 1; j < index_list->size(); j++) {
      index_list->at(j)--;
    }
  }
  std::shared_ptr<Graph> new_graph = std::make_shared<Graph>();
  for (size_t i = 0; i < graph->nodes.size(); i++) {
    if (index_list->at(i) > SIZE_MAX / SIZE_TWO) {
      continue;
    }
    new_graph->nodes.push_back(graph->nodes[i]);
    auto *node_in = &new_graph->nodes[index_list->at(i)].node_in;
    EraseEliminatedNode(node_in, index_list);
    auto *node_in_aux = &new_graph->nodes[index_list->at(i)].node_in_aux;
    EraseEliminatedNode(node_in_aux, index_list);
    auto *node_out = &new_graph->nodes[index_list->at(i)].node_out;
    EraseEliminatedNode(node_out, index_list);
  }
  return new_graph;
}
}  // namespace parallel
}  // namespace mindspore
