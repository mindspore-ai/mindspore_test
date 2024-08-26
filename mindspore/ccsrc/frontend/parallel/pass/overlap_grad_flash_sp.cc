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

#include "frontend/parallel/pass/overlap_grad_flash_sp.h"
#include <map>
#include <memory>
#include <vector>
#include <list>
#include <algorithm>
#include <string>
#include <queue>
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/op_enum.h"
#include "mindspore/ops/infer/ops_func_impl/flash_attention_score_grad.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "mindspore/ccsrc/frontend/parallel/ops_info/flash_attention_score_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/pass/pass_utils.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace parallel {
namespace {
enum TagType {
  query = 0,
  kv_a = 1,
  kv_b = 2,
  oml = 3,
};

struct FaGradCompareMethod {
  bool operator()(const std::string &a, const std::string &b) const {
    int a_1, a_2, b_1, b_2;
    std::istringstream a_stream(a);
    std::istringstream b_stream(b);
    char underline;
    a_stream >> a_1 >> underline >> a_2;
    b_stream >> b_1 >> underline >> b_2;
    return a_1 == b_1 ? a_2 > b_2 : a_1 > b_1;
  }
};

std::vector<int64_t> GetCommOrder(std::string comm_order_str) {
  std::istringstream iss(comm_order_str);
  std::string token;
  std::vector<int64_t> res;

  while (std::getline(iss, token, '_')) {
    res.push_back(std::stoi(token));
  }
  return res;
}

std::string NextFlashIndex(std::string flash_index, int64_t sp_num) {
  size_t underscore_pos = flash_index.find('_');
  if (underscore_pos == std::string::npos) {
    MS_LOG(ERROR) << "FLASH_INDEX ERROR";
  }

  std::string first_number_str = flash_index.substr(0, underscore_pos);
  int first_number = std::stoi(first_number_str);

  std::string second_number_str = flash_index.substr(underscore_pos + 1);
  int second_number = std::stoi(second_number_str) + 1;
  if (second_number > sp_num) {
    return "";
  }

  std::stringstream ss;
  ss << first_number << '_' << second_number;

  std::string new_str = ss.str();
  return new_str;
}

void FindTargetNode(std::vector<AnfNodePtr> *origin_nodes_topological, std::map<std::string, AnfNodePtr> *fa_map,
                    std::map<std::string, AnfNodePtr, FaGradCompareMethod> *grad_fa_map,
                    std::map<std::string, AnfNodePtr> *grad_send_qkv_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_qkv_map,
                    std::map<std::string, AnfNodePtr> *grad_send_oml_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_oml_map, CNodePtr *loss_node) {
  auto pipeline_stages = ParallelContext::GetInstance()->pipeline_stage_split_num();
  for (auto &anf_node : *origin_nodes_topological) {
    CNodePtr node = anf_node->cast<CNodePtr>();
    if (node != nullptr && node->HasPrimalAttr(FLASH_LOSS_NODE) && pipeline_stages <= 1) {
      (*loss_node) = node;
    }

    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScore)) {
      if (!node->HasPrimalAttr(FLASH_INDEX)) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(FLASH_INDEX));
      fa_map->insert({flash_index, node});
    }

    if (IsPrimitiveCNode(node, prim::kPrimFlashAttentionScoreGrad)) {
      if (!node->HasPrimalAttr(FLASH_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(FLASH_INDEX));
      (*grad_fa_map)[flash_index] = node;
    }

    if (IsPrimitiveCNode(node, prim::kPrimSend)) {
      if (!node->HasPrimalAttr(FLASH_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(FLASH_INDEX));
      if (GetValue<std::string>(node->GetPrimalAttr(FLASH_SP_COMM_TYPE)) == FLASH_SP_COMM_QKV) {
        (*grad_recv_qkv_map).insert({flash_index, node});
      } else {
        (*grad_recv_oml_map).insert({flash_index, node});
      }
    }

    if (IsPrimitiveCNode(node, prim::kPrimReceive)) {
      if (!node->HasPrimalAttr(FLASH_INDEX) || !node->HasPrimalAttr("forward_unique_id")) {
        continue;
      }
      auto flash_index = GetValue<std::string>(node->GetPrimalAttr(FLASH_INDEX));
      if (GetValue<std::string>(node->GetPrimalAttr(FLASH_SP_COMM_TYPE)) == FLASH_SP_COMM_QKV) {
        (*grad_send_qkv_map).insert({flash_index, node});
      } else {
        (*grad_send_oml_map).insert({flash_index, node});
      }
    }
  }
}

CNodePtr CreateDepend(const AnfNodePtr &latter_node, const AnfNodePtr &former_node, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(latter_node);
  if (former_node == nullptr) {
    return latter_node->cast<CNodePtr>();
  }
  MS_EXCEPTION_IF_NULL(node);
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), latter_node, former_node};
  auto depend_node = node->func_graph()->NewCNode(depend_inputs);
  MS_EXCEPTION_IF_NULL(depend_node);
  MS_EXCEPTION_IF_NULL(latter_node->abstract());
  depend_node->set_abstract(latter_node->abstract()->Clone());
  return depend_node;
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
  shape[split_dim] /= SizeToLong(split_num);
  std::vector<ShapeVector> shapes(split_num, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, split.get());
  split->set_scope(split_node->scope());
  MS_EXCEPTION_IF_NULL(split->abstract());
  return split;
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

CNodePtr NewConcatNode(const AnfNodePtr &input_node, size_t concat_dim) {
  MS_EXCEPTION_IF_NULL(input_node);
  auto concat_dim_node = NewValueNode(MakeValue(static_cast<int64_t>(concat_dim)));
  concat_dim_node->set_abstract(std::make_shared<abstract::AbstractScalar>(static_cast<int64_t>(concat_dim)));
  std::vector<AnfNodePtr> concat_inputs = {NewValueNode(prim::kPrimConcat), input_node, concat_dim_node};
  auto concat = input_node->func_graph()->NewCNode(concat_inputs);
  MS_EXCEPTION_IF_NULL(concat);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(input_node, 0)};
  auto shape = common::AnfAlgo::GetOutputInferShape(input_node, 0);
  shape[concat_dim] *= 2;
  std::vector<ShapeVector> shapes(1, shape);
  common::AnfAlgo::SetOutputInferTypeAndShape(dtypes, shapes, concat.get());
  concat->set_scope(input_node->scope());
  MS_EXCEPTION_IF_NULL(concat->abstract());
  return concat;
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

  auto shape = common::AnfAlgo::GetOutputInferShape(send_data, 0);

  common::AnfAlgo::SetNodeAttr(parallel::SR_TAG, MakeValue(tag), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::DEST_RANK, MakeValue(dest_rank), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP, MakeValue(group_name), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::GROUP_BACK, MakeValue(group_name), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::SHAPE, MakeValue(shape), send_node);
  common::AnfAlgo::SetNodeAttr(parallel::DTYPE, TypeIdToType(type_id), send_node);

  std::vector<TypeId> dtypes = {common::AnfAlgo::GetOutputInferDataType(send_data, 0)};
  // auto shape = common::AnfAlgo::GetOutputInferShape(send_data, 0);
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

int64_t GetPosInSpDevice(std::shared_ptr<FlashAttentionScoreInfo> flash_score_info_ptr) {
  auto rankList = flash_score_info_ptr->GetSPRankList();
  int64_t pos = -1;
  size_t dev_rank_id = g_device_manager->global_rank();
  for (size_t i = 0; i < rankList.size(); ++i) {
    if (dev_rank_id == LongToSize(rankList[i])) {
      pos = i;
    }
  }
  return pos;
}

std::shared_ptr<FlashAttentionScoreInfo> GetAttentionInfo(const std::map<std::string, AnfNodePtr> &fa_map) {
  CNodePtr fwd_score;
  for (auto &fa : fa_map) {
    auto fa_score_node_i = fa.second;
    auto fa_score_node_prim = GetCNodePrimitive(fa_score_node_i);
    MS_EXCEPTION_IF_NULL(fa_score_node_prim);
    if (fa_score_node_prim->HasAttr(parallel::ENABLE_FLASH_SP) &&
        GetValue<bool>((fa_score_node_prim->GetAttr(parallel::ENABLE_FLASH_SP)))) {
      fwd_score = fa_score_node_i->cast<CNodePtr>();
      break;
    }
  }
  MS_EXCEPTION_IF_NULL(fwd_score);
  std::shared_ptr<OperatorInfo> operator_info = fwd_score->user_data<parallel::OperatorInfo>();
  MS_EXCEPTION_IF_NULL(operator_info);
  auto flash_score_info_ptr = std::dynamic_pointer_cast<FlashAttentionScoreInfo>(operator_info);
  return flash_score_info_ptr;
}

size_t GetRankIndex(int64_t rank_id, size_t step, size_t sp_size) {
  std::vector<int> rank_order;
  for (size_t i = 0; i < sp_size; i++) {
    if (i % (step + 1) == 0) {
      rank_order.push_back(i);
    }
  }
  for (size_t i = 1; i <= step; i++) {
    for (size_t j = i; j < sp_size; j += step + 1) {
      rank_order.push_back(j);
    }
  }
  MS_LOG(ERROR) << "grad rank_order:" << rank_order;
  size_t pos = -1;
  for (size_t rank_list_idx = 0; rank_list_idx < rank_order.size(); ++rank_list_idx) {
    if (rank_id == rank_order[rank_list_idx]) {
      pos = rank_list_idx;
    }
  }
  return pos;
}

void GetGradNode(std::map<std::string, AnfNodePtr> *grad_send_qkv_map,
                 std::map<std::string, AnfNodePtr> *grad_recv_qkv_map,
                 std::map<std::string, AnfNodePtr> *grad_send_oml_map,
                 std::map<std::string, AnfNodePtr> *grad_recv_oml_map, std::map<int64_t, CNodePtr> *grad_comm_map,
                 const std::string &new_str) {
  CNodePtr grad_send_qkv_node;
  CNodePtr grad_recv_qkv_node;
  CNodePtr grad_send_oml_node;
  CNodePtr grad_recv_oml_node;
  if ((*grad_send_qkv_map).find(new_str) != (*grad_send_qkv_map).end()) {
    grad_send_qkv_node = (*grad_send_qkv_map).at(new_str)->cast<CNodePtr>();
  }
  if ((*grad_recv_qkv_map).find(new_str) != (*grad_recv_qkv_map).end()) {
    grad_recv_qkv_node = (*grad_recv_qkv_map).at(new_str)->cast<CNodePtr>();
  }
  if ((*grad_send_oml_map).find(new_str) != (*grad_send_oml_map).end()) {
    grad_send_oml_node = (*grad_send_oml_map).at(new_str)->cast<CNodePtr>();
  }
  if ((*grad_recv_oml_map).find(new_str) != (*grad_recv_oml_map).end()) {
    grad_recv_oml_node = (*grad_recv_oml_map).at(new_str)->cast<CNodePtr>();
  }
  (*grad_comm_map).insert({0, grad_send_qkv_node});
  (*grad_comm_map).insert({1, grad_recv_qkv_node});
  (*grad_comm_map).insert({2, grad_send_oml_node});
  (*grad_comm_map).insert({3, grad_recv_oml_node});
}

void GetPreCommNode(const std::string &new_str, int64_t index, std::map<std::string, AnfNodePtr> *grad_send_qkv_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_qkv_map,
                    std::map<std::string, AnfNodePtr> *grad_send_oml_map,
                    std::map<std::string, AnfNodePtr> *grad_recv_oml_map, CNodePtr *pre_grad_comm_node) {
  if (index == kIndex0 && (*grad_send_qkv_map).find(new_str) != (*grad_send_qkv_map).end()) {
    (*pre_grad_comm_node) = (*grad_send_qkv_map).at(new_str)->cast<CNodePtr>();
  } else if (index == kIndex1 && (*grad_recv_qkv_map).find(new_str) != (*grad_recv_qkv_map).end()) {
    (*pre_grad_comm_node) = (*grad_recv_qkv_map).at(new_str)->cast<CNodePtr>();
  } else if (index == kIndex2 && (*grad_send_oml_map).find(new_str) != (*grad_send_oml_map).end()) {
    (*pre_grad_comm_node) = (*grad_send_oml_map).at(new_str)->cast<CNodePtr>();
  } else if (index == kIndex3 && (*grad_recv_oml_map).find(new_str) != (*grad_recv_oml_map).end()) {
    (*pre_grad_comm_node) = (*grad_recv_oml_map).at(new_str)->cast<CNodePtr>();
  }
}

void SplitKVNode(std::vector<AnfNodePtr> *kv_a_nodes, std::vector<AnfNodePtr> *kv_b_nodes, const AnfNodePtr &key_node,
                 const AnfNodePtr &value_node) {
  auto key_split_node = NewSplitNode(key_node, kIndex2, kIndex2);
  auto value_split_node = NewSplitNode(value_node, kIndex2, kIndex2);
  auto key_a_node = NewTupleGetItemNode(key_split_node, kIndex0);
  auto key_b_node = NewTupleGetItemNode(key_split_node, kIndex1);
  auto value_a_node = NewTupleGetItemNode(value_split_node, kIndex0);
  auto value_b_node = NewTupleGetItemNode(value_split_node, kIndex1);
  (*kv_a_nodes) = {key_a_node, value_a_node};
  (*kv_b_nodes) = {key_b_node, value_b_node};
}

int64_t GetSendRecvTag(int64_t src, int64_t dest, TagType data_type) {
  auto src_string = std::to_string(src + 1);
  auto dest_string = std::to_string(dest + 1);
  auto data_type_string = std::to_string(data_type);

  auto res_string = src_string + dest_string + data_type_string;
  return std::stoi(res_string);
}

int64_t GetSendQKVDstRank(size_t rank, size_t step, size_t sp_size) { return (rank + step + 1) % sp_size; }

int64_t GetRecvQKVSrcRank(size_t rank, size_t step, size_t sp_size) { return (rank + sp_size - step - 1) % sp_size; }

CNodePtr GetCurrentSendQKVNode(size_t pos, size_t step, size_t inner_step, size_t sp_num, const AnfNodePtr &query_node,
                               const std::vector<CNodePtr> &send_kv_nodes, const RankList &spRankList,
                               const std::string &qkv_group, const CNodePtr &pre_node, Shape q_shape, Shape kv_shape) {
  CNodePtr cur_send_qkv_node;
  if (step < (sp_num / kIndex2)) {  // [0, sp-1]
    auto send_qkv_dst_rank = GetSendQKVDstRank(pos, step, sp_num);
    if (pos + step + 1 >= sp_num) {  // send q
      if (inner_step == 0) {
        cur_send_qkv_node = NewSendNode(CreateDepend(query_node, pre_node, pre_node),
                                        GetSendRecvTag(pos, send_qkv_dst_rank, TagType::query),
                                        spRankList[send_qkv_dst_rank], q_shape, TypeId::kNumberTypeFloat16, qkv_group);
      }
    } else {                                                             // send kv
      auto kv_type = (inner_step == 0 ? TagType::kv_a : TagType::kv_b);  // grad should send kv_a first
      auto send_kv_node = send_kv_nodes[inner_step % 2];
      send_kv_node = CreateDepend(send_kv_node, pre_node, pre_node);
      auto tag = GetSendRecvTag(pos, send_qkv_dst_rank, kv_type);
      auto dest_rank = spRankList[send_qkv_dst_rank];
      cur_send_qkv_node = NewSendNode(send_kv_node, tag, dest_rank, kv_shape, TypeId::kNumberTypeFloat16, qkv_group);
    }
  }
  return cur_send_qkv_node;
}

CNodePtr GetCurrentRecvQKVNode(size_t pos, size_t step, size_t inner_step, size_t sp_num, const RankList &spRankList,
                               Shape q_shape, Shape kv_shape, const std::string &qkv_group,
                               const AnfNodePtr &pre_node) {
  CNodePtr cur_recv_qkv_node;
  if (step < (sp_num / kIndex2)) {  // [0, sp-1]
    auto recv_qkv_src_rank = GetRecvQKVSrcRank(pos, step, sp_num);
    if (pos < step + kIndex1) {
      if (inner_step == kIndex0) {  // recv q
        cur_recv_qkv_node =
          NewReceiveNode(pre_node, GetSendRecvTag(recv_qkv_src_rank, pos, TagType::query),
                         spRankList[recv_qkv_src_rank], q_shape, TypeId::kNumberTypeFloat16, qkv_group);
        cur_recv_qkv_node->AddPrimalAttr("recv_type", MakeValue<int64_t>(TagType::query));
      }
    } else {  // recv kv
      auto kv_type = inner_step == kIndex0 ? TagType::kv_a : TagType::kv_b;
      cur_recv_qkv_node =
        NewReceiveNode(pre_node, GetSendRecvTag(recv_qkv_src_rank, pos, kv_type), spRankList[recv_qkv_src_rank],
                       kv_shape, TypeId::kNumberTypeFloat16, qkv_group);
      cur_recv_qkv_node->AddPrimalAttr("recv_type", MakeValue<int64_t>(kv_type));
    }
  }
  return cur_recv_qkv_node;
}

void ChangeQKVToBNSD(const std::shared_ptr<FlashAttentionScoreInfo> &fa_info, Shape *q_shape, Shape *kv_shape) {
  auto fa_n1 = fa_info->head_num();
  auto input_layout = fa_info->input_layout();
  *q_shape = fa_info->inputs_tensor_info()[kIndex0].tensor_layout().base_slice_shape().array();
  *kv_shape = fa_info->inputs_tensor_info()[kIndex1].tensor_layout().base_slice_shape().array();
  if (input_layout == ops::FASInputLayoutMode::BSH) {
    *q_shape = {(*q_shape)[kIndex0], fa_n1, (*q_shape)[kIndex1], (*q_shape)[kIndex2] / fa_n1};
    *kv_shape = {(*kv_shape)[kIndex0], fa_n1, (*kv_shape)[kIndex1], (*kv_shape)[kIndex2] / fa_n1};
  }
}

void WaitForAllInputs(CNodePtr *later_node, CNodePtr *fromer_node, const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  if (later_node == nullptr) {
    return;
  }
  auto later_node_input = (*later_node)->input(1);
  for (size_t i = 0; i < (*fromer_node)->size(); i++) {
    auto fromer_input_node = (*fromer_node)->input(i);
    if (fromer_input_node != nullptr && fromer_input_node->func_graph() != nullptr) {
      later_node_input = CreateDepend(later_node_input, fromer_input_node, (*later_node));
    }
  }
  manager->SetEdge((*later_node), 1, later_node_input);
}

void GetFirstFAGradQKV(const std::map<std::string, AnfNodePtr, FaGradCompareMethod> &grad_fa_map,
                       std::vector<CNodePtr> *send_kv_nodes, AnfNodePtr *query_node) {
  // first grad fa, grad_fa_map has been sorted
  CNodePtr first_grad_fa = (grad_fa_map.begin())->second->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(first_grad_fa);
  *query_node = first_grad_fa->input(ops::FASGradInputIndex::kFASGradInputQueryIndex + 1);
  auto key_node = first_grad_fa->input(ops::FASGradInputIndex::kFASGradInputKeyIndex + 1);
  auto value_node = first_grad_fa->input(ops::FASGradInputIndex::kFASGradInputValueIndex + 1);

  std::vector<AnfNodePtr> kv_a_nodes;
  std::vector<AnfNodePtr> kv_b_nodes;
  SplitKVNode(&kv_a_nodes, &kv_b_nodes, key_node, value_node);
  auto kv_a_concat = NewConcatNode(NewMakeTupleNode(kv_a_nodes), kIndex2);
  auto kv_b_concat = NewConcatNode(NewMakeTupleNode(kv_b_nodes), kIndex2);
  *send_kv_nodes = {kv_a_concat, kv_b_concat};
}

void ReplaceGradQKV(const FuncGraphPtr &graph, const CNodePtr &pre_grad_recv_qkv_node,
                    const CNodePtr &pre_grad_send_qkv_node, const CNodePtr &pre_grad_fa_node, CNodePtr *grad_fa_node) {
  auto manager = graph->manager();
  if (pre_grad_recv_qkv_node == nullptr && pre_grad_send_qkv_node == nullptr) {
    MS_LOG(EXCEPTION) << "Last step kv comm nodes have not been created.";
  }
  MS_EXCEPTION_IF_NULL(pre_grad_recv_qkv_node);
  MS_EXCEPTION_IF_NULL(*grad_fa_node);
  auto recv_type = GetValue<int64_t>(pre_grad_recv_qkv_node->GetPrimalAttr("recv_type"));
  if (recv_type == TagType::query) {
    CNodePtr query_node = pre_grad_recv_qkv_node;
    query_node = CreateDepend(query_node, pre_grad_send_qkv_node, pre_grad_send_qkv_node);
    query_node = CreateDepend(query_node, pre_grad_fa_node, pre_grad_fa_node);
    manager->SetEdge(*grad_fa_node, kIndex1, query_node);
  } else {
    auto kv_split = NewSplitNode(pre_grad_recv_qkv_node, kIndex2, kIndex2);
    auto key_node = NewTupleGetItemNode(kv_split, kIndex0);
    key_node = CreateDepend(key_node, pre_grad_recv_qkv_node, pre_grad_recv_qkv_node);
    key_node = CreateDepend(key_node, pre_grad_fa_node, pre_grad_fa_node);
    auto value_node = NewTupleGetItemNode(kv_split, kIndex1);
    manager->SetEdge(*grad_fa_node, kIndex2, key_node);
    manager->SetEdge(*grad_fa_node, kIndex3, value_node);
  }
}

void CreateCommNode(const FuncGraphPtr &graph, const std::shared_ptr<FlashAttentionScoreInfo> &fa_info,
                    const CNodePtr &grad_last_comm_node, const CNodePtr &pre_grad_comm_node,
                    const CNodePtr &pre_grad_fa_node, const AnfNodePtr &query_node,
                    const std::vector<CNodePtr> &send_kv_nodes, int64_t outer_step, int64_t inner_step, int64_t sp_num,
                    CNodePtr *pre_grad_send_qkv_node, CNodePtr *pre_grad_recv_qkv_node, CNodePtr *cur_send_qkv_node,
                    CNodePtr *cur_recv_qkv_node) {
  auto manager = graph->manager();
  int64_t pos = GetPosInSpDevice(fa_info);
  auto spRankList = fa_info->GetSPRankList();
  Shape q_shape;
  Shape kv_shape;
  ChangeQKVToBNSD(fa_info, &q_shape, &kv_shape);
  // current step has no dkv comm node
  int64_t grad_qkv_step = outer_step - 1;
  auto rank_ring_index = GetRankIndex(pos, grad_qkv_step, sp_num);
  CNodePtr depend_node = grad_last_comm_node == nullptr ? pre_grad_comm_node : grad_last_comm_node;
  if (rank_ring_index % kIndex2 == 0) {  // send first
    *cur_send_qkv_node =
      GetCurrentSendQKVNode(pos, grad_qkv_step, inner_step, sp_num, query_node, send_kv_nodes, spRankList,
                            g_device_manager->world_group(), depend_node, q_shape, kv_shape);
    if (*cur_send_qkv_node != nullptr) {
      auto cur_send_qkv_node_input = (*cur_send_qkv_node)->input(1);
      manager->SetEdge(*cur_send_qkv_node, 1,
                       CreateDepend(cur_send_qkv_node_input, pre_grad_fa_node, pre_grad_fa_node));
      depend_node = *cur_send_qkv_node;
    }
    *cur_recv_qkv_node = GetCurrentRecvQKVNode(pos, grad_qkv_step, inner_step, sp_num, spRankList, q_shape, kv_shape,
                                               g_device_manager->world_group(), depend_node);
  } else {
    *cur_recv_qkv_node = GetCurrentRecvQKVNode(pos, grad_qkv_step, inner_step, sp_num, spRankList, q_shape, kv_shape,
                                               g_device_manager->world_group(), depend_node);
    if (*cur_recv_qkv_node != nullptr) {
      auto cur_recv_qkv_node_input = (*cur_recv_qkv_node)->input(1);
      cur_recv_qkv_node_input = CreateDepend(cur_recv_qkv_node_input, depend_node, depend_node);
      cur_recv_qkv_node_input = CreateDepend(cur_recv_qkv_node_input, pre_grad_fa_node, pre_grad_fa_node);
      manager->SetEdge(*cur_recv_qkv_node, 1, cur_recv_qkv_node_input);
      depend_node = *cur_recv_qkv_node;
    }
    *cur_send_qkv_node =
      GetCurrentSendQKVNode(pos, grad_qkv_step, inner_step, sp_num, query_node, send_kv_nodes, spRankList,
                            g_device_manager->world_group(), depend_node, q_shape, kv_shape);
  }
  // if cur step receive nothing, use query which receive at latest step twice
  *pre_grad_recv_qkv_node = *cur_recv_qkv_node == nullptr ? *pre_grad_recv_qkv_node : *cur_recv_qkv_node;
  *pre_grad_send_qkv_node = *cur_send_qkv_node;
}

void AdjustCommNodeDependency(const FuncGraphPtr &graph, const CNodePtr &loss_node, const CNodePtr &pre_grad_comm_node,
                              const CNodePtr &pre_grad_fa_node, const CNodePtr &pre_grad_send_qkv_node,
                              const CNodePtr &pre_grad_recv_qkv_node, CNodePtr *grad_first_comm_node,
                              CNodePtr *grad_last_comm_node, CNodePtr *grad_fa_node) {
  auto manager = graph->manager();
  // ensure that fa and comm node start at the same time
  WaitForAllInputs(grad_first_comm_node, grad_fa_node, graph);
  WaitForAllInputs(grad_fa_node, grad_first_comm_node, graph);

  // first comm node sort after pre ring comm node
  auto grad_first_comm_node_input = (*grad_first_comm_node)->input(1);
  grad_first_comm_node_input = CreateDepend(grad_first_comm_node_input, loss_node, loss_node);
  grad_first_comm_node_input = CreateDepend(grad_first_comm_node_input, pre_grad_comm_node, pre_grad_comm_node);
  grad_first_comm_node_input = CreateDepend(grad_first_comm_node_input, pre_grad_fa_node, pre_grad_fa_node);
  grad_first_comm_node_input = CreateDepend(grad_first_comm_node_input, pre_grad_recv_qkv_node, pre_grad_recv_qkv_node);
  grad_first_comm_node_input = CreateDepend(grad_first_comm_node_input, pre_grad_send_qkv_node, pre_grad_send_qkv_node);
  manager->SetEdge(*grad_first_comm_node, 1, grad_first_comm_node_input);

  // grad split sort after fa
  manager->Replace(*grad_first_comm_node, CreateDepend(*grad_first_comm_node, *grad_fa_node, *grad_fa_node));

  // last comm node sort after first comm node
  if (*grad_first_comm_node != *grad_last_comm_node) {
    auto grad_last_comm_node_input = (*grad_last_comm_node)->input(1);
    manager->SetEdge(*grad_last_comm_node, 1,
                     CreateDepend(grad_last_comm_node_input, *grad_first_comm_node, *grad_first_comm_node));
  }
}
}  // namespace

void OverlapGradFlashSP(const FuncGraphPtr &graph) {
  auto manager = graph->manager();
  std::map<std::string, AnfNodePtr> fa_map;
  std::map<std::string, AnfNodePtr, FaGradCompareMethod> grad_fa_map;
  std::map<std::string, AnfNodePtr> grad_send_qkv_map;
  std::map<std::string, AnfNodePtr> grad_recv_qkv_map;
  std::map<std::string, AnfNodePtr> grad_send_oml_map;
  std::map<std::string, AnfNodePtr> grad_recv_oml_map;
  auto ret = graph->get_return();
  auto origin_nodes_topological = DeepScopedGraphSearch(ret);
  CNodePtr loss_node;
  FindTargetNode(&origin_nodes_topological, &fa_map, &grad_fa_map, &grad_send_qkv_map, &grad_recv_qkv_map,
                 &grad_send_oml_map, &grad_recv_oml_map, &loss_node);
  if (grad_fa_map.empty() || fa_map.empty()) {
    return;
  }
  auto fa_info = GetAttentionInfo(fa_map);
  MS_EXCEPTION_IF_NULL(fa_info);

  AnfNodePtr send_query_node;
  std::vector<CNodePtr> send_kv_nodes;
  GetFirstFAGradQKV(grad_fa_map, &send_kv_nodes, &send_query_node);

  int step = 0;
  // pre ring kv comm node
  CNodePtr pre_grad_recv_qkv_node;
  CNodePtr pre_grad_send_qkv_node;
  for (auto it = grad_fa_map.begin(); it != grad_fa_map.end(); ++it) {
    CNodePtr grad_fa_node = it->second->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(grad_fa_node);
    std::map<int64_t, CNodePtr> grad_comm_map;
    GetGradNode(&grad_send_qkv_map, &grad_recv_qkv_map, &grad_send_oml_map, &grad_recv_oml_map, &grad_comm_map,
                it->first);

    // pre ring dkv comm node
    CNodePtr pre_grad_comm_node;
    // pre ring fa grad
    CNodePtr pre_grad_fa_node;
    auto sp_num = GetValue<int64_t>(grad_fa_node->GetPrimalAttr("sp_num"));
    auto new_str = NextFlashIndex(it->first, sp_num);
    while (new_str != "") {
      if (grad_fa_map.find(new_str) != grad_fa_map.end()) {
        pre_grad_fa_node = grad_fa_map.at(new_str)->cast<CNodePtr>();
        if (GetValue<std::string>(pre_grad_fa_node->GetPrimalAttr("comm_order")) != "") {
          auto pre_comm_order = GetValue<std::string>(pre_grad_fa_node->GetPrimalAttr("comm_order"));
          auto pre_comm_order_list = GetCommOrder(pre_comm_order);
          GetPreCommNode(new_str, pre_comm_order_list[0], &grad_send_qkv_map, &grad_recv_qkv_map, &grad_send_oml_map,
                         &grad_recv_oml_map, &pre_grad_comm_node);
          break;
        }
      }
      new_str = NextFlashIndex(new_str, sp_num);
    }

    // cur ring dkv comm node
    CNodePtr grad_first_comm_node;
    CNodePtr grad_last_comm_node;
    // cur ring kv comm node
    CNodePtr cur_send_qkv_node;
    CNodePtr cur_recv_qkv_node;
    auto comm_order = GetValue<std::string>(grad_fa_node->GetPrimalAttr("comm_order"));
    auto comm_order_list = GetCommOrder(comm_order);
    if (comm_order_list.size() != 0) {
      grad_first_comm_node = grad_comm_map.at(comm_order_list[comm_order_list.size() - 1]);
      grad_last_comm_node = grad_comm_map.at(comm_order_list[0]);

      AdjustCommNodeDependency(graph, loss_node, pre_grad_comm_node, pre_grad_fa_node, pre_grad_send_qkv_node,
                               pre_grad_recv_qkv_node, &grad_first_comm_node, &grad_last_comm_node, &grad_fa_node);
    }

    int64_t outer_step = (sp_num - step) / kIndex2;
    int64_t inner_step = step == 0 ? 0 : (step + 1) % kIndex2;
    int64_t first_step = sp_num / kIndex2;
    if (outer_step >= 0) {                              // outer step
      if (outer_step != first_step && step < sp_num) {  // no need to replace grad qkv
        ReplaceGradQKV(graph, pre_grad_recv_qkv_node, pre_grad_send_qkv_node, pre_grad_fa_node, &grad_fa_node);
      }

      // create kv comm node when is not last outer_step
      if (outer_step != 0) {
        CreateCommNode(graph, fa_info, grad_last_comm_node, pre_grad_comm_node, pre_grad_fa_node, send_query_node,
                       send_kv_nodes, outer_step, inner_step, sp_num, &pre_grad_send_qkv_node, &pre_grad_recv_qkv_node,
                       &cur_send_qkv_node, &cur_recv_qkv_node);
      }
    }
    // no need create send/recv kv
    // ensure fa sort after cur dkv comm
    auto grad_fa_node_input = grad_fa_node->input(1);
    grad_fa_node_input = CreateDepend(grad_fa_node_input, grad_last_comm_node, grad_fa_node);
    grad_fa_node_input = CreateDepend(grad_fa_node_input, pre_grad_fa_node, grad_fa_node);
    grad_fa_node_input = CreateDepend(grad_fa_node_input, cur_send_qkv_node, grad_fa_node);
    grad_fa_node_input = CreateDepend(grad_fa_node_input, cur_recv_qkv_node, grad_fa_node);
    manager->SetEdge(grad_fa_node, 1, grad_fa_node_input);
    step++;
  }
}
}  // namespace parallel
}  // namespace mindspore
