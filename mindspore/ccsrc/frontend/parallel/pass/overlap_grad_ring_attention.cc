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

#include <map>
#include <memory>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <queue>
#include <utility>
#include "frontend/parallel/pass/overlap_grad_ring_attention.h"
#include "op_def/sequence_ops.h"
#include "op_def/other_ops.h"
#include "op_def/array_ops.h"
#include "op_def/nn_ops.h"
#include "op_def/math_ops.h"
#include "op_def/framework_ops.h"
#include "infer/make_tuple.h"
#include "utils/anf_utils.h"
#include "ir/tensor.h"
#include "utils/trace_base.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/utils/comm_manager.h"
#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "frontend/parallel/tensor_layout/tensor_info.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "pipeline/jit/ps/action.h"
#include "mindspore/ccsrc/frontend/parallel/graph_util/generate_graph.h"
#include "mindspore/ops/op_def/op_enum.h"
#include "mindspore/ops/infer/ops_func_impl/flash_attention_score.h"
#include "mindspore/ccsrc/frontend/parallel/ops_info/flash_attention_score_info.h"
#include "frontend/parallel/pass/flash_sp.h"
#include "frontend/parallel/parameter_manager.h"

struct FaGradCompareMethod {
  bool operator()(const std::string &a, const std::string &b) const {
    int a_1, a_2, b_1, b_2;
    std::istringstream a_stream(a);
    std::istringstream b_stream(b);
    char underline;
    a_stream >> a_1 >> underline >> a_2;
    b_stream >> b_1 >> underline >> b_2;

    if (a_1 < b_1) return true;
    if (a_1 > b_1) return false;

    return a_2 < b_2;
  }
};

namespace mindspore {
namespace parallel {
namespace {

std::string GetNewStr(std::string origin_str) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string first_number_str = origin_str.substr(0, underscore_pos);
  int first_number = std::stoi(first_number_str);

  std::string second_number_str = origin_str.substr(underscore_pos + 1);
  int second_number = std::stoi(second_number_str) + 1;

  std::stringstream ss;
  ss << first_number << '_' << second_number;

  std::string new_str = ss.str();
  return new_str;
}

std::string GetFirstStr(std::string origin_str, int64_t sp_num) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string first_number_str = origin_str.substr(0, underscore_pos);
  int64_t first_number = std::stoi(first_number_str);

  int64_t second_number = sp_num - 1;

  std::stringstream ss;
  ss << first_number << '_' << second_number;

  std::string new_str = ss.str();
  return new_str;
}

void FindFAGradInputNode(const CNodePtr &node, std::map<int64_t, AnfNodePtr> *attention_out_map,
                         std::map<int64_t, AnfNodePtr> *softmax_max_map, std::map<int64_t, AnfNodePtr> *softmax_sum_map,
                         std::map<int64_t, AnfNodePtr> *dout_map) {
  if (node != nullptr) {
    if (node->HasPrimalAttr(RING_ATTENTION_UPDATE_MUL) && node->HasPrimalAttr("forward_unique_id")) {
      auto flash_index = GetValue<int>(node->GetPrimalAttr(RING_ATTENTION_UPDATE_MUL));
      dout_map->insert({flash_index, node});
    }

    if (node->HasPrimalAttr(RING_ATTENTION_UPDATE_MAX)) {
      auto flash_index = GetValue<int>(node->GetPrimalAttr(RING_ATTENTION_UPDATE_MAX));
      softmax_max_map->insert({flash_index, node});
    }

    if (node->HasPrimalAttr(RING_ATTENTION_UPDATE_SUM)) {
      auto flash_index = GetValue<int>(node->GetPrimalAttr(RING_ATTENTION_UPDATE_SUM));
      softmax_sum_map->insert({flash_index, node});
    }

    if (node->HasPrimalAttr(RING_ATTENTION_UPDATE_ATTN)) {
      auto flash_index = GetValue<int>(node->GetPrimalAttr(RING_ATTENTION_UPDATE_ATTN));
      attention_out_map->insert({flash_index, node});
    }
  }
}

void FindTargetNode(std::vector<AnfNodePtr> *origin_nodes_topological, std::map<std::string, AnfNodePtr> *fwd_fas,
                    std::map<std::string, AnfNodePtr, FaGradCompareMethod> *grad_fa_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_map, std::map<std::string, AnfNodePtr> *grad_send_map,
                    CNodePtr *loss_node, std::map<int64_t, AnfNodePtr> *attention_out_map,
                    std::map<int64_t, AnfNodePtr> *softmax_max_map, std::map<int64_t, AnfNodePtr> *softmax_sum_map,
                    std::map<int64_t, AnfNodePtr> *dout_map) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  for (auto &anf_node : *origin_nodes_topological) {
    CNodePtr node = anf_node->cast<CNodePtr>();
    if (node != nullptr && node->HasPrimalAttr(FLASH_LOSS_NODE) && pipeline_stages <= 1) {
      (*loss_node) = node;
    }
    FindFAGradInputNode(node, attention_out_map, softmax_max_map, softmax_sum_map, dout_map);
    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScore)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX)) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      fwd_fas->insert({flash_index, node});
    }

    if (!IsPrimitiveCNode(node, prim::kPrimReceive) && !IsPrimitiveCNode(node, prim::kPrimSend) &&
        !IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      continue;
    }

    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_fa_map)[flash_index] = node;
    }

    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_recv_map).insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_send_map).insert({flash_index, node});
    }
  }
}

void DynFindTargetNode(std::vector<AnfNodePtr> *origin_nodes_topological,
                       std::map<std::string, AnfNodePtr> *grad_fa_map, std::map<std::string, AnfNodePtr> *grad_recv_map,
                       std::map<std::string, AnfNodePtr> *grad_send_map, CNodePtr *loss_node) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  for (auto &anf_node : *origin_nodes_topological) {
    CNodePtr node = anf_node->cast<CNodePtr>();
    if (node != nullptr && node->HasPrimalAttr(FLASH_LOSS_NODE) && pipeline_stages <= 1) {
      (*loss_node) = node;
    }
    if (!IsPrimitiveCNode(node, prim::kPrimReceive) && !IsPrimitiveCNode(node, prim::kPrimSend) &&
        !IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      continue;
    }

    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_fa_map).insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_recv_map).insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
      if (!node->HasPrimalAttr(RING_ATTENTION_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(RING_ATTENTION_INDEX));
      (*grad_send_map).insert({flash_index, node});
    }
  }
}

CNodePtr NewSendNode(const AnfNodePtr &send_data, int64_t tag, int64_t dest_rank, const Shape &send_shape,
                     TypeId type_id, const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(send_data);
  Attr attr_tag = std::make_pair(parallel::SR_TAG, MakeValue((tag)));
  Attr attr_rank = std::make_pair(parallel::DEST_RANK, MakeValue(dest_rank));
  Attr attr_group = std::make_pair(parallel::GROUP, MakeValue(group_name));
  Attr attr_group_back = std::make_pair(parallel::GROUP_BACK, MakeValue(group_name));
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_group, attr_group_back};
  auto send_input = ConvertToRealInputs("Send", "Send", AnfNodePtrList{send_data}, attrs);
  auto send_node = send_data->func_graph()->NewCNode(send_input);
  MS_EXCEPTION_IF_NULL(send_node);

  common::AnfAlgo::SetNodeAttr(parallel::SR_TAG, MakeValue(tag), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::DEST_RANK, MakeValue(dest_rank), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP, MakeValue(group_name), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP_BACK, MakeValue(group_name), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::SHAPE, MakeValue(send_shape), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::DTYPE, TypeIdToType(type_id), send_node);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(send_data, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(send_data, 0);
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, send_node.get());
  send_node->set_scope(send_data->scope());
  MS_EXCEPTION_IF_NULL(send_node->abstract());

  return send_node;
}

CNodePtr NewReceiveNode(const AnfNodePtr &parameter, int64_t tag, int64_t src_rank, const Shape &recv_shape,
                        TypeId type_id, const std::string &group_name) {
  MS_EXCEPTION_IF_NULL(parameter);
  Attr attr_tag = std::make_pair(parallel::SR_TAG, MakeValue((tag)));
  Attr attr_rank = std::make_pair(parallel::SRC_RANK, MakeValue(src_rank));
  Attr attr_shape = std::make_pair(parallel::SHAPE, MakeValue(recv_shape));
  Attr attr_dtype = std::make_pair(parallel::DTYPE, TypeIdToType(type_id));
  Attr attr_group = std::make_pair(parallel::GROUP, MakeValue(group_name));
  Attr attr_group_back = std::make_pair(parallel::GROUP_BACK, MakeValue(group_name));
  OperatorAttrs attrs = {attr_tag, attr_rank, attr_shape, attr_dtype, attr_group, attr_group_back};
  auto recv_inputs = ConvertToRealInputs("Receive", "Receive", AnfNodePtrList{parameter}, attrs);
  auto recv_node = parameter->func_graph()->NewCNode(recv_inputs);
  MS_EXCEPTION_IF_NULL(recv_node);

  common::AnfAlgo::SetNodeAttr(parallel::SR_TAG, MakeValue(tag), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::SRC_RANK, MakeValue(src_rank), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP, MakeValue(group_name), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP_BACK, MakeValue(group_name), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::SHAPE, MakeValue(recv_shape), recv_node);
  common::AnfAlgo::SetNodeAttr(parallel::DTYPE, TypeIdToType(type_id), recv_node);
  common::AnfAlgo::SetNodeAttr("flash_tag", MakeValue("True"), recv_node);

  common::AnfAlgo::SetOutputInferTypeAndShape({type_id}, {recv_shape}, recv_node.get());
  recv_node->set_scope(parameter->scope());
  MS_EXCEPTION_IF_NULL(recv_node->abstract());

  return recv_node;
}

CNodePtr NewConcatNode(const AnfNodePtr &input_node, size_t concat_dim) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto concat_dim_node = NewValueNode(MakeValue(static_cast<int64_t>(concat_dim)));
  concat_dim_node->set_abstract(std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(concat_dim)));
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(prim::kPrimConcat), input_node, concat_dim_node};
  auto concat = input_node->func_graph()->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_node, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  if (shape[concat_dim] > 0) {
    shape[concat_dim] *= 2;
  }
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, concat.get());
  concat->set_scope(input_node->scope());
  MS_EXCEPTION_IF_NULL(concat->abstract());
  return concat;
}

CNodePtr NewMakeTupleNode(const std::vector<AnfNodePtr> &input_nodes) {
  // input_nodes are getitem nodes
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < input_nodes.size(); ++i) {
    make_tuple_inputs.push_back(input_nodes[i]);
  }
  auto make_tuple = input_nodes[0]->func_graph()->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);

  auto dtype = common::AnfAlgo::GetOutputInferDataType(input_nodes[0], 0);
  std::vector<TypeId> dtypes(input_nodes.size(), dtype);
  auto shape = common::AnfAlgo::GetOutputInferShape(input_nodes[0], 0);
  std::vector<ShapeVector> shapes(input_nodes.size(), shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, make_tuple.get());
  make_tuple->set_scope(input_nodes[0]->scope());
  MS_EXCEPTION_IF_NULL(make_tuple->abstract());
  return make_tuple;
}

CNodePtr NewSplitNode(const AnfNodePtr &split_node, size_t split_dim, size_t split_num) {
  if (split_num == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "split_num should not be zero.";
  }
  MS_EXCEPTION_IF_NULL(split_node);
  auto split_dim_node = NewValueNode<int64_t>(split_dim);
  split_dim_node->set_abstract(std::make_shared<abstract::AbstractScalar>(SizeToLong(split_dim)));
  auto split_num_node = NewValueNode<int64_t>(split_num);
  split_num_node->set_abstract(std::make_shared<abstract::AbstractScalar>(SizeToLong(split_num)));
  std::vector<AnfNodePtr> split_inputs = {NewValueNode(prim::kPrimSplit), split_node, split_dim_node, split_num_node};
  auto split = split_node->func_graph()->NewCNode(split_inputs);
  MS_EXCEPTION_IF_NULL(split);

  auto dtype = common::AnfAlgo::GetOutputInferDataType(split_node, 0);
  std::vector<TypeId> dtypes(split_num, dtype);
  auto shape = common::AnfAlgo::GetOutputInferShape(split_node, 0);
  if (shape[split_dim] > 0) {
    shape[split_dim] /= SizeToLong(split_num);
  }
  std::vector<ShapeVector> shapes(split_num, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());
  split->set_scope(split_node->scope());
  MS_EXCEPTION_IF_NULL(split->abstract());
  return split;
}

CNodePtr NewTupleGetItemNode(const AnfNodePtr &input_node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto idx = NewValueNode(SizeToLong(output_index));
  MS_EXCEPTION_IF_NULL(idx);
  idx->set_abstract(std::make_shared<abstract::AbstractScalar>(SizeToLong(output_index)));
  auto getitem = input_node->func_graph()->NewCNode({NewValueNode(prim::kPrimTupleGetItem), input_node, idx});
  MS_EXCEPTION_IF_NULL(getitem);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_node, output_index)};
  auto shapes = {common::AnfAlgo::GetOutputInferShape(input_node, output_index)};
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, getitem.get());
  getitem->set_scope(input_node->scope());
  MS_EXCEPTION_IF_NULL(getitem->abstract());
  return getitem;
}

CNodePtr CreateDepend(const AnfNodePtr &latter_node, const AnfNodePtr &former_node, const CNodePtr &node) {
  if (former_node == nullptr) {
    return latter_node->cast<CNodePtr>();
  }
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), latter_node, former_node};
  auto depend_node = node->func_graph()->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->set_abstract(latter_node->abstract()->Clone());
  return depend_node;
}

void GetPreNode(CNodePtr *pre_grad_recv_node, CNodePtr *pre_grad_send_node, const std::string &new_str,
                std::map<std::string, AnfNodePtr> *grad_recv_map, std::map<std::string, AnfNodePtr> *grad_send_map) {
  if ((*grad_recv_map).find(new_str) != (*grad_recv_map).end() &&
      (*grad_send_map).find(new_str) != (*grad_send_map).end()) {
    (*pre_grad_recv_node) = (*grad_recv_map).at(new_str)->cast<CNodePtr>();
    (*pre_grad_send_node) = (*grad_send_map).at(new_str)->cast<CNodePtr>();
  }
}

bool IsFwdBeginningNode(const std::string &origin_str) {
  size_t underscore_pos = origin_str.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "Flash_Index ERROR";
  }

  std::string second_number_str = origin_str.substr(underscore_pos + 1);
  int second_number = std::stoi(second_number_str);
  if (second_number == 0) {
    return true;
  }
  return false;
}

std::shared_ptr<OperatorInfo> GetAttentionInfo(const std::map<std::string, AnfNodePtr> &fa_map) {
  CNodePtr fwd_score;
  for (auto &fa : fa_map) {
    auto fa_score_node_i = fa.second;
    auto fa_score_node_prim = GetCNodePrimitive(fa_score_node_i);
    MS_EXCEPTION_IF_NULL(fa_score_node_prim);
    if ((!fa_score_node_prim->HasAttr(parallel::ENABLE_RING_ATTENTION) ||
         !GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_RING_ATTENTION)))) &&
        (!fa_score_node_prim->HasAttr(parallel::ENABLE_FLASH_SP) ||
         !GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_FLASH_SP))))) {
      continue;
    }
    fwd_score = fa_score_node_i->cast<CNodePtr>();
  }

  std::shared_ptr<OperatorInfo> operator_info = fwd_score->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  return operator_info;
}

void ChangeFAGradInput(std::map<std::string, AnfNodePtr, FaGradCompareMethod> *grad_fa_map, const FuncGraphPtr &graph,
                       std::map<int64_t, AnfNodePtr> *attention_out_map, std::map<int64_t, AnfNodePtr> *softmax_max_map,
                       std::map<int64_t, AnfNodePtr> *softmax_sum_map, std::map<int64_t, AnfNodePtr> *dout_map) {
  auto manager = graph->manager();
  for (auto it = (*grad_fa_map).rbegin(); it != (*grad_fa_map).rend(); ++it) {
    auto cur_grad_fa_node = it->second->cast<CNodePtr>();
    size_t underscore_pos = (it->first).find('_');
    if (underscore_pos == std::string::npos) {
      MS_LOG(ERROR) << "Flash_Index ERROR";
    }

    std::string first_number_str = (it->first).substr(0, underscore_pos);
    int first_number = std::stoi(first_number_str);
    auto dout_node = dout_map->find(first_number)->second;
    auto softmax_max_node = softmax_max_map->find(first_number)->second;
    auto softmax_sum_node = softmax_sum_map->find(first_number)->second;
    auto attention_out_node = attention_out_map->find(first_number)->second;
    MS_EXCEPTION_IF_NULL(dout_node);
    MS_EXCEPTION_IF_NULL(softmax_max_node);
    MS_EXCEPTION_IF_NULL(softmax_sum_node);
    MS_EXCEPTION_IF_NULL(attention_out_node);
    manager->SetEdge(cur_grad_fa_node, 4, dout_node->cast<CNodePtr>()->input(2));
    manager->SetEdge(cur_grad_fa_node, 9, softmax_max_node);
    manager->SetEdge(cur_grad_fa_node, 10, softmax_sum_node);
    manager->SetEdge(cur_grad_fa_node, 11, cur_grad_fa_node->input(7));
    manager->SetEdge(cur_grad_fa_node, 12, attention_out_node);
  }
}

void WaitForAllInputs(CNodePtr *later_node, CNodePtr *fromer_node, const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  auto later_node_input = (*later_node)->input(1);
  for (size_t i = 0; i < (*fromer_node)->size(); i++) {
    auto fromer_input_node = (*fromer_node)->input(i);
    if (fromer_input_node != nullptr && fromer_input_node->func_graph() != nullptr) {
      later_node_input = CreateDepend(later_node_input, fromer_input_node, (*later_node));
    }
  }
  manager->SetEdge((*later_node), 1, later_node_input);
}

void GradFirstStepCommKV(const std::string &cur_str, std::map<std::string, AnfNodePtr> *fa_map,
                         const FuncGraphPtr &graph, CNodePtr *grad_fa_node, int64_t pos, int64_t send_rank_id,
                         int64_t recv_rank_id, const Shape &neigh_shape, TypeId output_type_id,
                         std::map<std::string, AnfNodePtr> *grad_send_kv_map,
                         std::map<std::string, AnfNodePtr> *grad_recv_kv_map) {
  auto manager = graph->manager();
  auto fwd_fa_node = (*fa_map).at(cur_str)->cast<CNodePtr>();
  auto key_node = fwd_fa_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputKeyIndex + 1);
  auto value_node = fwd_fa_node->input(ops::FlashAttentionScoreInputIndex::kFlashAttentionScoreInputValueIndex + 1);
  manager->SetEdge((*grad_fa_node), 2, key_node);
  manager->SetEdge((*grad_fa_node), 3, value_node);
  std::vector<AnfNodePtr> kv_nodes = {key_node, value_node};
  auto kv_tuple = NewMakeTupleNode(kv_nodes);
  auto kv_concat = NewConcatNode(kv_tuple, 0);
  AnfNodePtr send_node, recv_node;
  auto dout = (*grad_fa_node)->input(4);
  auto sp_num = GetValue<int64_t>((*grad_fa_node)->GetPrimalAttr("sp_num"));
  auto first_str = GetFirstStr(cur_str, sp_num);
  auto fwd_last_fa_node = (*fa_map).at(first_str)->cast<CNodePtr>();
  dout = CreateDepend(dout, fwd_last_fa_node, fwd_last_fa_node);
  if (pos % kIndex2 == kIndex0) {
    send_node = NewSendNode(CreateDepend(kv_concat, dout, kv_concat), 0, send_rank_id, neigh_shape, output_type_id,
                            g_device_manager->world_group());
    recv_node =
      NewReceiveNode(send_node, 0, recv_rank_id, neigh_shape, output_type_id, g_device_manager->world_group());
  } else {
    recv_node = NewReceiveNode(CreateDepend(kv_concat, dout, kv_concat), 0, recv_rank_id, neigh_shape, output_type_id,
                               g_device_manager->world_group());
    send_node = NewSendNode(CreateDepend(kv_concat, recv_node, kv_concat), 0, send_rank_id, neigh_shape, output_type_id,
                            g_device_manager->world_group());
  }
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(cur_str + "grad"), send_node);
  common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(cur_str + "grad"), recv_node);
  (*grad_send_kv_map).insert({cur_str, send_node});
  (*grad_recv_kv_map).insert({cur_str, recv_node});
}

void GetCommInfo(int64_t *send_rank_id, int64_t *recv_rank_id, std::shared_ptr<OperatorInfo> *operator_info,
                 int64_t *pos) {
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>((*operator_info));

  auto rankList = flash_score_info_ptr->GetSPRankList();
  (*pos) = -1;
  size_t dev_rank_id = g_device_manager->global_rank();
  for (size_t i = 0; i < rankList.size(); ++i) {
    if (dev_rank_id == LongToSize(rankList[i])) {
      (*pos) = i;
    }
  }
  (*recv_rank_id) = rankList[((*pos) + 1) % rankList.size()];
  (*send_rank_id) = rankList[((*pos) + rankList.size() - 1) % rankList.size()];
}

void CreateCommForRAGrad(const FuncGraphPtr &graph, const CNodePtr &pre_grad_recv_kv_node,
                         const CNodePtr &pre_grad_send_kv_node, const CNodePtr &last_fa_grad, const CNodePtr &loss_node,
                         int64_t pos, int64_t send_rank_id, int64_t recv_rank_id, const Shape &neigh_shape,
                         TypeId output_type_id, CNodePtr *grad_fa_node, CNodePtr *grad_recv_node,
                         CNodePtr *grad_send_node, AnfNodePtr *send_node, AnfNodePtr *recv_node) {
  auto manager = graph->manager();
  auto kv_split = NewSplitNode(pre_grad_recv_kv_node, kIndex0, kIndex2);
  auto key_node = NewTupleGetItemNode(kv_split, kIndex0);
  auto value_node = NewTupleGetItemNode(kv_split, kIndex1);
  std::vector<AnfNodePtr> kv_nodes = {key_node, value_node};
  auto kv_tuple = NewMakeTupleNode(kv_nodes);
  auto kv_concat = NewConcatNode(kv_tuple, 0);

  key_node = CreateDepend(key_node, last_fa_grad, *grad_send_node);  // cur fa wait pre fa
  kv_concat = CreateDepend(kv_concat, last_fa_grad, kv_concat);
  if (pos % kIndex2 == 0) {
    kv_concat = CreateDepend(kv_concat, *grad_recv_node, *grad_send_node);
    *send_node = NewSendNode(CreateDepend(kv_concat, pre_grad_recv_kv_node, *grad_send_node), 0, send_rank_id,
                             neigh_shape, output_type_id, g_device_manager->world_group());
    *recv_node =
      NewReceiveNode(*send_node, 0, recv_rank_id, neigh_shape, output_type_id, g_device_manager->world_group());
    auto grad_send_input = (*grad_send_node)->input(1);
    grad_send_input = CreateDepend(grad_send_input, pre_grad_recv_kv_node, *grad_send_node);
    grad_send_input = CreateDepend(grad_send_input, loss_node, *grad_send_node);
    manager->SetEdge(*grad_send_node, 1, grad_send_input);
    WaitForAllInputs(grad_send_node, grad_fa_node, graph);
    WaitForAllInputs(grad_fa_node, grad_send_node, graph);

    // ensure grad split which is output of receive(grad send) sort after fa
    manager->Replace(*grad_send_node, CreateDepend(*grad_send_node, *grad_fa_node, *grad_send_node));

    auto grad_recv_input = (*grad_recv_node)->input(1);
    manager->SetEdge(*grad_recv_node, 1, CreateDepend(grad_recv_input, *grad_send_node, *grad_recv_node));

    // ensure fa sort after current grad send/recv
    key_node = CreateDepend(key_node, *grad_recv_node, *grad_recv_node);
    manager->SetEdge(*grad_fa_node, kIndex2, key_node);
    manager->SetEdge(*grad_fa_node, kIndex3, value_node);

    manager->Replace(*grad_fa_node, CreateDepend(*grad_fa_node, *grad_recv_node, *grad_fa_node));
  } else {
    kv_concat = CreateDepend(kv_concat, *grad_send_node, *grad_send_node);
    *recv_node = NewReceiveNode(CreateDepend(kv_concat, pre_grad_send_kv_node, kv_concat), 0, recv_rank_id, neigh_shape,
                                output_type_id, g_device_manager->world_group());
    *send_node = NewSendNode(CreateDepend(kv_concat, *recv_node, kv_concat), 0, send_rank_id, neigh_shape,
                             output_type_id, g_device_manager->world_group());
    auto grad_recv_input = (*grad_recv_node)->input(1);
    grad_recv_input = CreateDepend(grad_recv_input, pre_grad_send_kv_node, *grad_recv_node);
    grad_recv_input = CreateDepend(grad_recv_input, loss_node, *grad_recv_node);
    manager->SetEdge(*grad_recv_node, 1, grad_recv_input);
    WaitForAllInputs(grad_recv_node, grad_fa_node, graph);
    WaitForAllInputs(grad_fa_node, grad_recv_node, graph);

    // ensure grad split which is output of receive(grad send) sort after fa
    manager->Replace(*grad_send_node, CreateDepend(*grad_send_node, *grad_fa_node, *grad_send_node));

    auto grad_send_input = (*grad_send_node)->input(1);
    manager->SetEdge(*grad_send_node, 1, CreateDepend(grad_send_input, *grad_recv_node, *grad_send_node));

    // ensure fa sort after current grad send/recv
    key_node = CreateDepend(key_node, *grad_send_node, *grad_send_node);
    manager->SetEdge(*grad_fa_node, kIndex2, key_node);
    manager->SetEdge(*grad_fa_node, kIndex3, value_node);

    manager->Replace(*grad_fa_node, CreateDepend(*grad_fa_node, *grad_send_node, *grad_fa_node));
  }
}
}  // namespace

void DynOverlapGradRingAttention(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  std::map<std::string, AnfNodePtr> grad_fa_map;
  std::map<std::string, AnfNodePtr> grad_send_map;
  std::map<std::string, AnfNodePtr> grad_recv_map;
  auto ret = graph->get_return();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);
  CNodePtr loss_node;

  DynFindTargetNode(&origin_nodes_topological, &grad_fa_map, &grad_recv_map, &grad_send_map, &loss_node);
  for (auto it = grad_send_map.begin(); it != grad_send_map.end(); ++it) {
    auto grad_fa_node = grad_fa_map.at(it->first)->cast<CNodePtr>();
    auto grad_recv_node = grad_recv_map.at(it->first)->cast<CNodePtr>();
    auto grad_send_node = it->second->cast<CNodePtr>();

    CNodePtr pre_grad_recv_node;
    CNodePtr pre_grad_send_node;
    auto new_str = GetNewStr(it->first);
    GetPreNode(&pre_grad_recv_node, &pre_grad_send_node, new_str, &grad_recv_map, &grad_send_map);

    auto grad_recv_pos = GetValue<int64_t>(grad_recv_node->GetPrimalAttr(RING_ATTENTION_POS));
    if (grad_recv_pos % kIndex2 == 0) {
      auto grad_send_input = grad_send_node->input(1);
      for (size_t i = 0; i < grad_fa_node->size(); i++) {
        auto grad_fa_input_node = grad_fa_node->input(i);
        if (grad_fa_input_node != nullptr && grad_fa_input_node->func_graph() != nullptr) {
          grad_send_input = CreateDepend(grad_send_input, grad_fa_input_node, grad_send_node);
        }
      }
      grad_send_input = CreateDepend(grad_send_input, loss_node, grad_send_node);
      manager->SetEdge(grad_send_node, 1, grad_send_input);

      auto grad_fa_node_input = grad_fa_node->input(1);
      for (size_t i = 0; i < grad_send_node->size(); i++) {
        auto grad_send_input_node = grad_send_node->input(i);
        if (grad_send_input_node != nullptr && grad_send_input_node->func_graph() != nullptr) {
          grad_fa_node_input = CreateDepend(grad_fa_node_input, grad_send_input_node, grad_fa_node);
        }
      }
      manager->SetEdge(grad_fa_node, 1, grad_fa_node_input);

      if (pre_grad_recv_node != nullptr) {
        auto grad_send_input_new = grad_send_node->input(1);
        manager->SetEdge(grad_send_node, 1, CreateDepend(grad_send_input_new, pre_grad_recv_node, grad_send_node));
      }

      auto grad_recv_input = grad_recv_node->input(1);
      manager->SetEdge(grad_recv_node, 1, CreateDepend(grad_recv_input, grad_send_node, grad_recv_node));

      manager->Replace(grad_recv_node, CreateDepend(grad_recv_node, grad_fa_node, grad_recv_node));

      std::vector<AnfNodePtr> depend3_inputs{NewValueNode(prim::kPrimDepend), grad_fa_node, grad_recv_node};
      auto depend_node3 = grad_fa_node->func_graph()->NewCNode(depend3_inputs);
      MS_EXCEPTION_IF_NULL(depend_node3);
      depend_node3->set_abstract(grad_fa_node->abstract()->Clone());
      manager->Replace(grad_fa_node, CreateDepend(grad_fa_node, grad_recv_node, grad_fa_node));
    } else {
      auto grad_recv_input = grad_recv_node->input(1);
      for (size_t i = 0; i < grad_fa_node->size(); i++) {
        auto grad_fa_input_node = grad_fa_node->input(i);
        if (grad_fa_input_node != nullptr && grad_fa_input_node->func_graph() != nullptr) {
          grad_recv_input = CreateDepend(grad_recv_input, grad_fa_input_node, grad_recv_node);
        }
      }
      grad_recv_input = CreateDepend(grad_recv_input, loss_node, grad_recv_node);
      manager->SetEdge(grad_recv_node, 1, grad_recv_input);

      auto grad_fa_node_input = grad_fa_node->input(1);
      for (size_t i = 0; i < grad_recv_node->size(); i++) {
        auto grad_recv_input_node = grad_recv_node->input(i);
        if (grad_recv_input_node != nullptr && grad_recv_input_node->func_graph() != nullptr) {
          grad_fa_node_input = CreateDepend(grad_fa_node_input, grad_recv_input_node, grad_fa_node);
        }
      }
      manager->SetEdge(grad_fa_node, 1, grad_fa_node_input);

      if (pre_grad_send_node != nullptr) {
        auto grad_recv_input_new = grad_recv_node->input(1);
        manager->SetEdge(grad_recv_node, 1, CreateDepend(grad_recv_input_new, pre_grad_send_node, grad_recv_node));
      }

      auto grad_send_input = grad_send_node->input(1);
      manager->SetEdge(grad_send_node, 1, CreateDepend(grad_send_input, grad_recv_node, grad_send_node));
      manager->Replace(grad_send_node, CreateDepend(grad_send_node, grad_fa_node, grad_send_node));
      manager->Replace(grad_fa_node, CreateDepend(grad_fa_node, grad_send_node, grad_fa_node));
    }
  }
}

void StaticOverlapGradRingAttention(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  std::map<std::string, AnfNodePtr> fa_map, grad_send_map, grad_recv_map;
  std::map<std::string, AnfNodePtr, FaGradCompareMethod> grad_fa_map;
  std::map<int64_t, AnfNodePtr> attention_out_map, softmax_max_map, softmax_sum_map, dout_map;
  auto ret = graph->get_return();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);
  CNodePtr loss_node;

  FindTargetNode(&origin_nodes_topological, &fa_map, &grad_fa_map, &grad_recv_map, &grad_send_map, &loss_node,
                 &attention_out_map, &softmax_max_map, &softmax_sum_map, &dout_map);
  if (grad_fa_map.empty() || fa_map.empty()) return;

  auto operator_info = GetAttentionInfo(fa_map);
  int64_t pos, recv_rank_id, send_rank_id;
  GetCommInfo(&send_rank_id, &recv_rank_id, &operator_info, &pos);
  std::map<std::string, AnfNodePtr> grad_send_kv_map, grad_recv_kv_map;

  auto kv_shape = operator_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  auto neigh_shape = kv_shape;
  if (neigh_shape[kIndex0] != -1) {
    neigh_shape[kIndex0] = neigh_shape[kIndex0] * kIndex2;
  }
  ChangeFAGradInput(&grad_fa_map, graph, &attention_out_map, &softmax_max_map, &softmax_sum_map, &dout_map);

  for (auto it = grad_fa_map.rbegin(); it != grad_fa_map.rend(); ++it) {
    auto grad_fa_node = it->second->cast<CNodePtr>();
    auto output_type_id = common::AnfAlgo::GetOutputInferDataType(grad_fa_node, kIndex3);
    auto new_str = GetNewStr(it->first);

    CNodePtr grad_recv_node;
    CNodePtr grad_send_node;
    if (grad_recv_map.find(it->first) != grad_recv_map.end() && grad_send_map.find(it->first) != grad_send_map.end()) {
      grad_recv_node = grad_recv_map.at(it->first)->cast<CNodePtr>();
      grad_send_node = grad_send_map.at(it->first)->cast<CNodePtr>();
    }

    if (grad_recv_node == nullptr || grad_send_node == nullptr) {
      GradFirstStepCommKV(it->first, &fa_map, graph, &grad_fa_node, pos, send_rank_id, recv_rank_id, neigh_shape,
                          output_type_id, &grad_send_kv_map, &grad_recv_kv_map);
      continue;
    }

    CNodePtr pre_grad_recv_kv_node;
    CNodePtr pre_grad_send_kv_node;
    GetPreNode(&pre_grad_recv_kv_node, &pre_grad_send_kv_node, new_str, &grad_recv_kv_map, &grad_send_kv_map);
    auto pre_fa_grad = grad_fa_map.at(new_str)->cast<CNodePtr>();

    AnfNodePtr send_node;
    AnfNodePtr recv_node;
    CreateCommForRAGrad(graph, pre_grad_recv_kv_node, pre_grad_send_kv_node, pre_fa_grad, loss_node, pos, send_rank_id,
                        recv_rank_id, neigh_shape, output_type_id, &grad_fa_node, &grad_recv_node, &grad_send_node,
                        &send_node, &recv_node);
    common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(it->first + "grad"), send_node);
    common::AnfAlgo::SetNodeAttr(RING_ATTENTION_INDEX, MakeValue<std::string>(it->first + "grad"), recv_node);

    if (!IsFwdBeginningNode(it->first)) {
      grad_send_kv_map.insert({it->first, send_node});
      grad_recv_kv_map.insert({it->first, recv_node});
    }
  }
}

void OverlapGradRingAttention(const FuncGraphPtr &graph) {
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  if (pipeline::IsDynamicShapeGraph(graph)) {
    DynOverlapGradRingAttention(graph);
    return;
  }
  StaticOverlapGradRingAttention(graph);
}
}  // namespace parallel
}  // namespace mindspore
