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
#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_DATASET_READER_OPTIMIZER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_DATASET_READER_OPTIMIZER_H_

#include <vector>
#include <set>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/manager.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
constexpr int64_t BROADCAST_ROOT_RANK = 0;
constexpr int64_t BETWEEN_STAGE = 1;
constexpr int64_t WITHIN_STAGE = 2;
constexpr int64_t OPT_ALL = 3;
class DatasetReaderOptimizer {
 public:
  DatasetReaderOptimizer(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root)
      : manager_(manager), root_(root) {}
  virtual ~DatasetReaderOptimizer() = default;

  bool Init();
  void BroadcastDataset();

 private:
  RankList InferReapteDataRankThroughDataStrategy(const Strategies &data_stra);
  std::vector<RankList> InferRepeatDataRankThroughLayout();
  std::vector<RankList> InferRepeatRankListWithinStage();
  AnfNodePtr FindDatasetParameter(const AnfNodePtr &node, const NodeUsersMap &node_users_map);
  void FindAllStageIdUsedDataParameter(const AnfNodePtr &node, const NodeUsersMap &node_users_map,
                                       std::set<int64_t> *const data_used_stage);
  RankList InferRepeatRankList(const RankList &within_stage, const RankList &between_stage);
  void InsertBroadcast(const RankList &rank_list);
  std::vector<CNodePtr> broadcast_ops;
  int64_t opt_level_ = 0;
  FuncGraphManagerPtr manager_ = nullptr;
  FuncGraphPtr root_ = nullptr;
  AnfNodePtr virtual_dataset_ = nullptr;
  AnfNodePtr get_next_ = nullptr;
};
void FreezeParallelOptimizerCommOrder(const FuncGraphPtr &graph);
void ReplaceGetnextWithBroadcast(const FuncGraphPtr &graph);
void ControlOptShardCommAndDataBroadcastOrder(const FuncGraphPtr &graph);
void ControlPipelineCommAndDataBroadcastOrder(const FuncGraphPtr &graph);
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_DATASET_READER_OPTIMIZER_H_
