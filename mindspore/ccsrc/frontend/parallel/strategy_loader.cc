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

#include "frontend/parallel/strategy_loader.h"

#include <vector>
#include <string>
#include "mindspore/ops/op_def/other_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/dynamic_shape/dynamic_shape.h"
#include "frontend/parallel/tensor_layout/tensor_transform.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/strategy_utils.h"

namespace mindspore {
namespace parallel {
void StrategyLoader::LoadStrategyFromFile(const std::vector<AnfNodePtr> &all_nodes) {
  if (!StrategyCheckpoint::GetInstance().LoadAutoOpStrategyOn()) {
    return;
  }
  StrategyMap stra_map;
  StrategyMap out_stra_map;
  if ((StrategyCheckpoint::GetInstance().LoadAutoOpStrategy(&stra_map, &out_stra_map) != SUCCESS)) {
    return;
  }
  MS_LOG(INFO) << "Load strategies map from json successfully";
  SetStridedSliceSplitStrategy(all_nodes);
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!StrategyUtils::CheckExtractInformation(cnode) || IsPrimitiveCNode(node, prim::kPrimSend)) {
      continue;
    }
    StrategyUtils::SetVirtualDatasetStrategy(cnode);

    OperatorInfoPtr operator_info = CreateOperatorInfo(cnode);
    MS_EXCEPTION_IF_NULL(operator_info);

    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);

    if (prim->name() == RESHAPE) {
      cnode->set_user_data<OperatorInfo>(operator_info);
      continue;
    }

    std::string strategy_key_name = cnode->fullname_with_scope();
    if (stra_map.find(strategy_key_name) == stra_map.end()) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "Not found strategy for " << strategy_key_name;
    }
    StrategyPtr s_strategy = stra_map[strategy_key_name];
    operator_info->SetSelectedStrategy(s_strategy, 0);

    StrategyPtr o_strategy = nullptr;
    if (out_stra_map.find(strategy_key_name) != out_stra_map.end()) {
      o_strategy = out_stra_map[strategy_key_name];
      operator_info->set_out_strategy(o_strategy);
    }
    Status initRet = operator_info->InitSelectedStrategy(s_strategy, o_strategy);
    if (initRet == SUCCESS) {
      MS_LOG(INFO) << "Init selected strategy succeeded.";
    } else {
      MS_LOG(EXCEPTION) << "Init selected strategy failed.";
    }
    cnode->set_user_data<OperatorInfo>(operator_info);
    cnode->AddAttr(OP_INFO_CREATED, MakeValue(true));
  }
  MS_LOG(INFO) << "End load strategies from file";
}

void StrategyLoader::SaveStrategyToFile(const std::vector<AnfNodePtr> &all_nodes) {
  if (!StrategyCheckpoint::GetInstance().SaveAutoOpStrategyOn() || !CheckShardingPropagation()) {
    return;
  }
  StrategyMap stra_map;
  StrategyMap out_stra_map;

  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    std::string strategy_key_name = cnode->fullname_with_scope();
    OperatorInfoPtr op = cnode->user_data<OperatorInfo>();
    if (op == nullptr) {
      continue;
    }
    StrategyPtr s_strategy = op->selected_strategy();
    if (s_strategy != nullptr) {
      stra_map[strategy_key_name] = s_strategy;
    }
    StrategyPtr o_strategy = op->out_strategy();
    if (o_strategy != nullptr) {
      out_stra_map[strategy_key_name] = o_strategy;
    }
  }
  if (StrategyCheckpoint::GetInstance().SaveAutoOpStrategy(stra_map, out_stra_map) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Save strategy checkpoint failed";
  }
  MS_LOG(INFO) << "Success save strategies to file.";
}
}  // namespace parallel
}  // namespace mindspore
