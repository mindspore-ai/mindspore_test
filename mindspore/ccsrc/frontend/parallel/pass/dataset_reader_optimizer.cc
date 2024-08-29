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

#include "frontend/parallel/pass/dataset_reader_optimizer.h"
#include <algorithm>
#include <stack>
#include <queue>
#include <set>
#include <string>
#include "mindspore/core/ops/other_ops.h"
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "mindspore/core/ops/structure_ops.h"
#include "mindspore/core/ops/sequence_ops.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "utils/hash_set.h"
#include "mindspore/core/utils/tensor_construct_utils.h"
#include "frontend/parallel/pass/overlap_opt_shard_in_pipeline.h"

namespace mindspore {
namespace parallel {
bool DatasetReaderOptimizer::Init() {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    return false;
  }
  opt_level_ = ms_context->get_param<int>(MS_CTX_DATASET_BROADCAST_OPT_LEVEL);
  if (ParallelInit() != SUCCESS) {
    return false;
  }
  if (root_ == nullptr) {
    return false;
  }
  auto all_nodes = TopoSort(root_->get_return(), SuccDeeperSimple);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimVirtualDataset)) {
      continue;
    }
    virtual_dataset_ = node;
    break;
  }
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimGetNext)) {
      continue;
    }
    get_next_ = node;
    break;
  }
  return true;
}

RankList DatasetReaderOptimizer::InferReapteDataRankThroughDataStrategy(const Strategies &data_stra) {
  auto max_shard = 1;
  RankList rank_list = {};
  if (virtual_dataset_ == nullptr) {
    return rank_list;
  }
  for (const auto &each_stra : data_stra) {
    auto cur_max_shard = *max_element(each_stra.begin(), each_stra.end());
    max_shard = (max_shard > cur_max_shard) ? max_shard : cur_max_shard;
  }
  auto dev_num = g_device_manager->stage_device_num();
  auto repeat_num = dev_num / max_shard;
  if (repeat_num == 1) {
    return rank_list;
  }
  auto cnode = virtual_dataset_->cast<CNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  uint64_t repeat_dim = 0;
  DeviceMatrix device_matrix;
  if (prim->HasAttr(REPEAT_DIM_DIRECT) && GetValue<std::string>(prim->GetAttr(REPEAT_DIM_DIRECT)) == RIGHT) {
    device_matrix = DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(),
                                 {max_shard, repeat_num});
    repeat_dim = 1;
  } else {
    device_matrix = DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(),
                                 {repeat_num, max_shard});
  }
  if (device_matrix.GetDevicesAlongDim(repeat_dim, &rank_list)) {
    MS_LOG(WARNING) << "Failed to get dataset repeat rank list within pipeline stage.";
  }
  return rank_list;
}

RankList DatasetReaderOptimizer::InferRepeatRankListWithinStage() {
  RankList rank_list = {};
  if (opt_level_ != WITHIN_STAGE && opt_level_ != OPT_ALL) {
    return rank_list;
  }
  auto data_stra = ParallelContext::GetInstance()->dataset_strategy();
  if (!data_stra.empty()) {
    return InferReapteDataRankThroughDataStrategy(data_stra);
  }
  bool full_batch = ParallelContext::GetInstance()->full_batch();
  if (full_batch) {
    return g_device_manager->GetDeviceListInThisStage();
  }
  return rank_list;
}

AnfNodePtr DatasetReaderOptimizer::FindDatasetParameter(const AnfNodePtr &node, const NodeUsersMap &node_users_map) {
  std::stack<AnfNodePtr> st;
  HashSet<AnfNodePtr> visited;
  st.push(node);
  visited.insert(node);
  while (!st.empty()) {
    auto cur_node = st.top()->cast<CNodePtr>();
    st.pop();
    auto cur_node_users = node_users_map.at(cur_node);
    for (const auto &node_pair : cur_node_users) {
      auto user_node = node_pair.first->cast<CNodePtr>();
      if (IsValueNode<FuncGraph>(user_node->input(0))) {
        auto fg = GetValueNode<FuncGraphPtr>(user_node->input(0));
        if (fg->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
          auto fg_params = fg->parameters();
          return fg_params.at(node_pair.second - 1);
        }
      }
      if (visited.find(user_node) == visited.end() && !IsPrimitiveCNode(user_node, prim::kPrimReturn)) {
        st.push(user_node);
        visited.insert(user_node);
      }
    }
  }
  return nullptr;
}

RankList DatasetReaderOptimizer::FindAllStageIdUsedDataParameter(const AnfNodePtr &node,
                                                                 const NodeUsersMap &node_users_map) {
  RankList rank_list = {};
  if (opt_level_ != BETWEEN_STAGE && opt_level_ != OPT_ALL) {
    return rank_list;
  }
  std::queue<AnfNodePtr> queue;
  HashSet<AnfNodePtr> visited;
  queue.push(node);
  visited.insert(node);
  std::set<int64_t> stage_set;
  while (!queue.empty()) {
    auto cur_node = queue.front();
    queue.pop();
    auto cur_node_users = node_users_map.at(cur_node);
    for (const auto &node_pair : cur_node_users) {
      auto user_node = node_pair.first->cast<CNodePtr>();
      if (IsValueNode<FuncGraph>(user_node->input(0))) {
        auto fg = GetValueNode<FuncGraphPtr>(user_node->input(0));
        auto stage = fg->stage();
        if (stage != -1) {
          stage_set.insert(stage);
        }
        continue;
      }
      if (visited.find(user_node) == visited.end() && !IsPrimitiveCNode(user_node, prim::kPrimReturn)) {
        queue.push(user_node);
        visited.insert(user_node);
      }
    }
  }
  rank_list.assign(stage_set.begin(), stage_set.end());
  return rank_list;
}

RankList DatasetReaderOptimizer::InferRepeatRankList(const RankList &within_stage, const RankList &between_stage) {
  RankList rank_list;
  auto local_stage = g_device_manager->stage_id();
  auto stage_device_num = g_device_manager->stage_device_num();
  if (between_stage.empty()) {
    return within_stage;
  }
  if (within_stage.empty()) {
    std::transform(between_stage.begin(), between_stage.end(), std::back_inserter(rank_list),
                   [&](const auto &stage_id) {
                     auto global_rank = g_device_manager->global_rank();
                     auto tar_rank = global_rank + SizeToLong(stage_device_num) * (stage_id - local_stage);
                     return tar_rank;
                   });
    return rank_list;
  }
  for (const auto &stage_id : between_stage) {
    std::transform(within_stage.begin(), within_stage.end(), std::back_inserter(rank_list),
                   [&](const auto &repeat_rank) {
                     auto tar_rank = repeat_rank + SizeToLong(stage_device_num) * (stage_id - local_stage);
                     return tar_rank;
                   });
  }
  return rank_list;
}

AnfNodePtr DatasetReaderOptimizer::CreateZeroNode(const AnfNodePtr &node) {
  auto abs = node->abstract();
  if (abs == nullptr) {
    return nullptr;
  }
  auto tensor_abs = abs->cast<abstract::AbstractTensorPtr>();
  if (tensor_abs == nullptr) {
    return nullptr;
  }
  TypePtr tensor_type_ptr = tensor_abs->element()->BuildType();
  Shape tensor_shape = tensor_abs->shape()->shape();
  auto zero_tensor = TensorConstructUtils::CreateZerosTensor(tensor_type_ptr, tensor_shape);
  if (zero_tensor == nullptr) {
    return nullptr;
  }
  return NewValueNode(zero_tensor);
}

void DatasetReaderOptimizer::InsertBroadcast(const NodeUsersMap &node_user_map, const RankList &rank_list,
                                             int64_t index) {
  auto node_users = node_user_map.at(get_next_);
  auto global_rank = g_device_manager->global_rank();
  auto iter = std::find(rank_list.begin(), rank_list.end(), global_rank);
  constexpr int64_t place_holder = 0;
  if (iter == rank_list.end()) {
    return;
  }
  for (const auto &node_pair : node_users) {
    auto user_node = node_pair.first->cast<CNodePtr>();
    if (!IsPrimitiveCNode(user_node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto input_index = GetTupleGetItemIndex(user_node);
    if (input_index != index) {
      continue;
    }
    Group data_repeat_group;
    if (g_device_manager->CreateGroup(rank_list, &data_repeat_group) != SUCCESS) {
      MS_LOG(WARNING) << "Create dataset repeat group failed, rank list is: " << rank_list;
      return;
    }
    auto fg = user_node->func_graph();
    std::vector<AnfNodePtr> make_tuple_input = {NewValueNode(prim::kPrimMakeTuple->Clone())};
    if (iter == rank_list.begin()) {
      (void)make_tuple_input.emplace_back(user_node);
    } else {
      auto zero_tensor = CreateZeroNode(user_node);
      (void)make_tuple_input.emplace_back(zero_tensor);
    }
    auto make_tuple = fg->NewCNode(make_tuple_input);
    std::vector<AnfNodePtr> broadcast_input = {NewValueNode(prim::kPrimBroadcast->Clone()), make_tuple};
    auto broadcast = fg->NewCNode(broadcast_input);
    auto prim = GetCNodePrimitive(broadcast);
    prim->set_attr(ROOT_RANK, MakeValue(BROADCAST_ROOT_RANK));
    prim->set_attr(GROUP, MakeValue(data_repeat_group.name()));
    prim->set_attr(DATASET_BROADCAST, MakeValue(True));
    (void)broadcast_ops.emplace_back(broadcast);
    if (iter == rank_list.begin()) {
      std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), user_node,
                                              NewValueNode(MakeValue(place_holder))};
      auto depend = fg->NewCNode(depend_input);
      (void)manager_->Replace(user_node, depend);
      (void)manager_->SetEdge(depend, SIZE_TWO, broadcast);
      return;
    }
    std::vector<AnfNodePtr> getitem_input = {NewValueNode(prim::kPrimTupleGetItem), broadcast,
                                             NewValueNode(MakeValue(INT64_ZERO))};
    auto getitem = fg->NewCNode(getitem_input);
    (void)manager_->Replace(user_node, getitem);
    return;
  }
}

CNodePtr BroadcastReorder(const std::vector<CNodePtr> &broadcast_ops) {
  auto op_num = broadcast_ops.size();
  if (broadcast_ops.empty()) {
    return nullptr;
  }
  auto first_broadcast = broadcast_ops.front()->cast<CNodePtr>();
  auto first_broadcast_prim = GetCNodePrimitive(first_broadcast);
  first_broadcast_prim->set_attr(FIRST_BROADCAST, MakeValue(True));
  if (op_num == 1) {
    auto cnode = broadcast_ops.front()->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(cnode);
    prim->set_attr(LAST_BROADCAST, MakeValue(True));
    return first_broadcast;
  }
  for (size_t index = 0; index < op_num - 1; ++index) {
    auto prior_node = broadcast_ops.at(index)->cast<CNodePtr>();
    auto last_node = broadcast_ops.at(index + 1)->cast<CNodePtr>();
    if (index == op_num - 2) {
      auto prim = GetCNodePrimitive(last_node);
      prim->set_attr(LAST_BROADCAST, MakeValue(True));
    }
    std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), last_node->input(1), prior_node};
  }
  return first_broadcast;
}

void DatasetReaderOptimizer::BroadcastDataset() {
  if (get_next_ == nullptr) {
    MS_LOG(WARNING) << "For now on, only dataset sink mode support dataset reader optimizer.";
    return;
  }
  if (virtual_dataset_ == nullptr) {
    return;
  }
  auto reapte_rank_within_stage = InferRepeatRankListWithinStage();
  const auto &node_users_map = manager_->node_users();
  auto dataset_users = node_users_map.at(virtual_dataset_);
  for (const auto &node_pair : dataset_users) {
    auto cnode = node_pair.first->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto input_index = GetTupleGetItemIndex(cnode);
    auto cur_input_parameter = FindDatasetParameter(cnode, node_users_map);
    if (cur_input_parameter == nullptr) {
      continue;
    }
    auto cur_input_used_stage = FindAllStageIdUsedDataParameter(cur_input_parameter, node_users_map);
    auto rank_list = InferRepeatRankList(reapte_rank_within_stage, cur_input_used_stage);
    if (rank_list.empty()) {
      continue;
    }
    InsertBroadcast(node_users_map, rank_list, input_index);
  }
}

void ControlOptShardCommAndDataBroadcastOrder(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    return;
  }
  if (ms_context->get_param<int>(MS_CTX_DATASET_BROADCAST_OPT_LEVEL) == 0) {
    return;
  }
  if (graph == nullptr) {
    return;
  }
  if (!IsTraining(graph->manager())) {
    return;
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    return;
  }
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  if (parallel_mode != parallel::kSemiAutoParallel && parallel_mode != parallel::kAutoParallel) {
    return;
  }
  auto all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
  std::vector<CNodePtr> broadcast_ops;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimBroadcast)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(cnode);
    if (!prim->HasAttr(DATASET_BROADCAST)) {
      continue;
    }
    broadcast_ops.emplace_back(cnode);
  }
  if (broadcast_ops.empty()) {
    return;
  }
  auto first_broadcast = BroadcastReorder(broadcast_ops);
  if (first_broadcast == nullptr) {
    return;
  }
  std::vector<CNodePtr> opt_shard_comm_list;
  for (const auto &node : all_nodes) {
    if (!is_allgather_comm_ops(node)) {
      continue;
    }
    auto all_gather_cnode = node->cast<CNodePtr>();
    (void)opt_shard_comm_list.emplace_back(all_gather_cnode);
  }
  if (opt_shard_comm_list.empty()) {
    return;
  }
  std::vector<AnfNodePtr> make_tuple_inputs{NewValueNode(prim::kPrimMakeTuple)};
  (void)std::copy(opt_shard_comm_list.begin(), opt_shard_comm_list.end(), std::back_inserter(make_tuple_inputs));
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), first_broadcast->input(1),
                                        graph->NewCNode(make_tuple_inputs)};
  auto depend_node = graph->NewCNode(depend_inputs);
  depend_node->set_abstract(first_broadcast->input(1)->abstract()->Clone());
  (void)manager->Replace(first_broadcast->input(1), depend_node);
}

void ControlPipelineCommAndDataBroadcastOrder(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    return;
  }
  if (ms_context->get_param<int>(MS_CTX_DATASET_BROADCAST_OPT_LEVEL) == 0) {
    return;
  }
  if (parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kSemiAutoParallel &&
      parallel::ParallelContext::GetInstance()->parallel_mode() != parallel::kAutoParallel) {
    return;
  }
  if (g_device_manager == nullptr) {
    return;
  }
  auto stage_num = g_device_manager->stage_num();
  if (stage_num <= 1) {
    return;
  }
  auto manager = graph->manager();
  if (manager == nullptr) {
    return;
  }
  auto all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
  CNodePtr first_recv;
  for (const auto &node : all_nodes) {
    if (is_first_receive((node))) {
      first_recv = node->cast<CNodePtr>();
      break;
    }
  }
  if (first_recv == nullptr) {
    return;
  }
  CNodePtr last_broadcast;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimBroadcast)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(cnode);
    if (!prim->HasAttr(LAST_BROADCAST)) {
      continue;
    }
    last_broadcast = cnode;
    break;
  }
  if (last_broadcast == nullptr) {
    return;
  }
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), first_recv->input(1), last_broadcast};
  auto depend_node = graph->NewCNode(depend_inputs);
  depend_node->set_abstract(first_recv->input(1)->abstract()->Clone());
  (void)manager->Replace(first_recv->input(1), depend_node);
}
}  // namespace parallel
}  // namespace mindspore
