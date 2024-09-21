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
#include <string>
#include "mindspore/ops/op_def/other_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "utils/hash_set.h"
#include "utils/tensor_construct_utils.h"
#include "frontend/parallel/pass/overlap_opt_shard_in_pipeline.h"

namespace mindspore {
namespace parallel {
bool DatasetReaderOptimizer::Init() {
  auto ms_context = MsContext::GetInstance();
  if (ms_context == nullptr) {
    return false;
  }
  opt_level_ = ms_context->get_param<int>(MS_CTX_DATASET_BROADCAST_OPT_LEVEL);
  auto jit_level = ms_context->get_param<std::string>(MS_CTX_JIT_LEVEL);
  if (jit_level != "O0" && jit_level != "O1") {
    MS_LOG(WARNING) << "Now, Dataset broadcast optimize pass only support O0 and O1 jit level.";
    return false;
  }
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

void DatasetReaderOptimizer::FindAllStageIdUsedDataParameter(const AnfNodePtr &node, const NodeUsersMap &node_users_map,
                                                             std::set<int64_t> *const data_used_stage) {
  if (opt_level_ != BETWEEN_STAGE && opt_level_ != OPT_ALL) {
    return;
  }
  std::queue<AnfNodePtr> queue;
  HashSet<AnfNodePtr> visited;
  queue.push(node);
  visited.insert(node);
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
          data_used_stage->insert(stage);
        }
        continue;
      }
      if (visited.find(user_node) == visited.end() && !IsPrimitiveCNode(user_node, prim::kPrimReturn)) {
        queue.push(user_node);
        visited.insert(user_node);
      }
    }
  }
  return;
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

bool DatasetReaderOptimizer::CreateZeroNode(const Shapes &shapes, const std::vector<TypePtr> &types,
                                            std::vector<AnfNodePtr> *const input_vec) {
  auto data_stra = ParallelContext::GetInstance()->dataset_strategy();
  if (shapes.size() != types.size()) {
    return false;
  }
  for (size_t i = 0; i < shapes.size(); ++i) {
    tensor::TensorPtr zero_tensor = nullptr;
    auto cur_input_shape = shapes.at(i);
    auto cur_input_type = types.at(i);
    Shape slice_shape;
    if (!data_stra.empty()) {
      if (data_stra.size() != shapes.size()) {
        return false;
      }
      auto cur_input_stra = data_stra.at(i);
      if (cur_input_stra.size() != cur_input_shape.size()) {
        return false;
      }
      for (size_t j = 0; j < cur_input_stra.size(); ++j) {
        slice_shape.emplace_back(cur_input_shape.at(j) / cur_input_stra.at(j));
      }
    } else {
      slice_shape = cur_input_shape;
      auto full_batch = ParallelContext::GetInstance()->full_batch();
      if (!full_batch) {
        auto dev_num = g_device_manager->stage_device_num();
        slice_shape[0] = slice_shape[0] / dev_num;
      }
    }

    zero_tensor = TensorConstructUtils::CreateZerosTensor(cur_input_type, slice_shape);
    if (zero_tensor == nullptr) {
      return false;
    }
    input_vec->emplace_back(NewValueNode(zero_tensor));
  }

  return true;
}

void DatasetReaderOptimizer::InsertBroadcast(const RankList &rank_list) {
  auto global_rank = g_device_manager->global_rank();
  auto iter = std::find(rank_list.begin(), rank_list.end(), global_rank);
  if (iter == rank_list.end()) {
    return;
  }
  std::vector<AnfNodePtr> broadcast_input = {NewValueNode(prim::kPrimBroadcast->Clone())};
  AnfNodePtr broadcast;
  if (iter == rank_list.begin()) {
    (void)broadcast_input.emplace_back(get_next_);
    broadcast = root_->NewCNode(broadcast_input);
    std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), get_next_, broadcast};
    auto depend = root_->NewCNode(depend_input);
    (void)manager_->Replace(get_next_, depend);
  } else {
    auto prim = GetCNodePrimitive(get_next_);
    auto shape_attr = prim->GetAttr(SHAPES);
    auto type_attr = prim->GetAttr(TYPES);
    if (shape_attr == nullptr || type_attr == nullptr) {
      return;
    }
    std::vector<ValuePtr> shape = shape_attr->isa<ValueTuple>() ? shape_attr->cast<ValueTuplePtr>()->value()
                                                                : shape_attr->cast<ValueListPtr>()->value();
    Shapes shapes;
    for (const auto &element : shape) {
      std::vector<ValuePtr> element_list =
        element->isa<ValueTuple>() ? element->cast<ValueTuplePtr>()->value() : element->cast<ValueListPtr>()->value();
      Shape shape_vec;
      (void)std::transform(element_list.begin(), element_list.end(), std::back_inserter(shape_vec),
                           [](const ValuePtr &v) -> int64_t { return GetValue<int64_t>(v); });
      shapes.emplace_back(shape_vec);
    }
    auto types = GetValue<std::vector<TypePtr>>(type_attr);
    std::vector<AnfNodePtr> make_tuple_input = {NewValueNode(prim::kPrimMakeTuple->Clone())};
    if (!CreateZeroNode(shapes, types, &make_tuple_input)) {
      return;
    }
    if (make_tuple_input.size() == 1) {
      return;
    }
    auto make_tuple = root_->NewCNode(make_tuple_input);
    (void)broadcast_input.emplace_back(make_tuple);
    broadcast = root_->NewCNode(broadcast_input);
    (void)manager_->Replace(get_next_, broadcast);
  }
  Group data_repeat_group;
  if (g_device_manager->CreateGroup(rank_list, &data_repeat_group) != SUCCESS) {
    MS_LOG(WARNING) << "Create dataset repeat group failed, rank list is: " << rank_list;
    return;
  }
  if (broadcast == nullptr) {
    return;
  }
  auto prim = GetCNodePrimitive(broadcast);
  prim->set_attr(ROOT_RANK, MakeValue(BROADCAST_ROOT_RANK));
  prim->set_attr(GROUP, MakeValue(data_repeat_group.name()));
  prim->set_attr(DATASET_BROADCAST, MakeValue(True));
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
  std::set<int64_t> data_used_stage;
  for (const auto &node_pair : dataset_users) {
    auto cnode = node_pair.first->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cnode, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto cur_input_parameter = FindDatasetParameter(cnode, node_users_map);
    if (cur_input_parameter != nullptr) {
      FindAllStageIdUsedDataParameter(cur_input_parameter, node_users_map, &data_used_stage);
    }
  }
  RankList reapte_rank_between_stage;
  reapte_rank_between_stage.assign(data_used_stage.begin(), data_used_stage.end());
  auto rank_list = InferRepeatRankList(reapte_rank_within_stage, reapte_rank_between_stage);
  if (rank_list.size() <= 1) {
    return;
  }
  InsertBroadcast(rank_list);
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
  CNodePtr broadcast_op;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimBroadcast)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(cnode);
    if (!prim->HasAttr(DATASET_BROADCAST)) {
      continue;
    }
    broadcast_op = cnode;
    break;
  }
  if (broadcast_op == nullptr) {
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
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), broadcast_op->input(1),
                                        graph->NewCNode(make_tuple_inputs)};
  auto depend_node = graph->NewCNode(depend_inputs);
  depend_node->set_abstract(broadcast_op->input(1)->abstract()->Clone());
  (void)manager->Replace(broadcast_op->input(1), depend_node);
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
  CNodePtr broadcast_op;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimBroadcast)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetCNodePrimitive(cnode);
    if (!prim->HasAttr(DATASET_BROADCAST)) {
      continue;
    }
    broadcast_op = cnode;
    break;
  }
  if (broadcast_op == nullptr) {
    return;
  }
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), first_recv->input(1), broadcast_op};
  auto depend_node = graph->NewCNode(depend_inputs);
  depend_node->set_abstract(first_recv->input(1)->abstract()->Clone());
  (void)manager->Replace(first_recv->input(1), depend_node);
}
}  // namespace parallel
}  // namespace mindspore
