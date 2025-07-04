/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_PIPELINE_SPLIT_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_PIPELINE_SPLIT_UTILS_H_

#include <utility>
#include <vector>
#include <string>
#include "ir/anf.h"
#include "ir/manager.h"

namespace mindspore {
namespace parallel {
using PipelinePair = std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>>;
using PipelinePairVector = std::vector<std::vector<mindspore::parallel::PipelinePair>>;
AnfNodePtr FindAccuGrad(const CNodePtr &cnode);
bool IsFirstStage();
bool IsLastStage();
int64_t InferStage();
void InsertVirtualAssignAdd(const std::pair<AnfNodePtr, int> &node_user, const FuncGraphManagerPtr &manager,
                            const AnfNodePtr &accu_parameter, const NodeUsersMap &node_user_map);
void InsertVirtualAccuGrad(const AnfNodePtr &recv, const FuncGraphManagerPtr &manager, const AnfNodePtr &param);
AnfNodePtr FindGradAccuParameter(const std::vector<AnfNodePtr> &parameters, const std::string &name,
                                 const std::string &user_name = "");
void HandleReceiveParam(const FuncGraphPtr &root);
void AddVirtualAssignAdd(const FuncGraphPtr &root);
void SetParameterStartForCellShare(const FuncGraphPtr &root);
bool CompFunc(const AnfNodePtr &node1, const AnfNodePtr &node2);
void ReorderForForward(const std::vector<AnfNodePtr> &forward_start, const std::vector<AnfNodePtr> &forward_end,
                       const FuncGraphPtr &root);
void ReorderForBackward(const PipelinePair &forward_start_pair, const PipelinePair &forward_end_pair,
                        const PipelinePair &backward_start_pair, const PipelinePair &backward_end_pair,
                        const PipelinePair &forward_end_before_pair, const FuncGraphPtr &root);
void ReorderForParams(const PipelinePair &backward_params_pair, const PipelinePair &forward_params_pair,
                      const PipelinePair &backward_end_pair, const PipelinePair &forward_start_pair,
                      const FuncGraphPtr &root);
int64_t GetMicroBatch(const AnfNodePtr &node);
void InsertDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node, const FuncGraphManagerPtr &manager,
                  const FuncGraphPtr &root, const std::string &attr_tag = "", const size_t post_node_input_index = 1);
AnfNodePtr GetActualOp(const AnfNodePtr &node);
void GetBorderNode(std::vector<AnfNodePtr> *forward_start, std::vector<AnfNodePtr> *forward_end,
                   std::vector<AnfNodePtr> *backward_start, std::vector<AnfNodePtr> *backward_end,
                   std::vector<AnfNodePtr> *forward_params, std::vector<AnfNodePtr> *backward_params,
                   std::vector<AnfNodePtr> *allreduce_params, const FuncGraphPtr &root);
void Reorder(const FuncGraphPtr &root);
void ReorderForPredict(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager);
void HandleMicroBatch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager);
void BroadCastMicroBatch(const CNodePtr &node, NodeUsersMap *node_users_map, const ValuePtr &value, size_t max_depth);
void BroadCastSeqChunk(const FuncGraphPtr &root);
void AddVirtualAssignKvCache(const FuncGraphPtr &root);
void LabelNeedGrad(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root);
void BroadCastNeedGrad(const AnfNodePtr &node, NodeUsersMap *node_user_map, const FuncGraphPtr &root);
void LastStageEndNode(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager,
                      const FuncGraphPtr &root);
void SetStridedSliceStrategy(const AnfNodePtr &node);
void ParameterStartNode(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphManagerPtr &manager);
bool IsValidNode(const AnfNodePtr &node, const AnfNodePtr &return_node, const NodeUsersMap &node_user_map);
ValuePtr Micro(const CNodePtr &cnode, NodeUsersMap *node_users_map, size_t max_depth);
void CheckBorderNode(const PipelinePair &forward_start_pair, const PipelinePair &forward_end_pair,
                     const PipelinePair &backward_start_pair, const PipelinePair &backward_end_pair,
                     std::vector<int64_t> seg_micro_max);
void CommonDeduplicate(const std::vector<AnfNodePtr> &node_vector, std::vector<AnfNodePtr> *out_vec_begin,
                       std::vector<AnfNodePtr> *out_vec_end, const FuncGraphPtr &root, int64_t micro_max,
                       int64_t seg_max, int64_t h, bool is_train);
PipelinePair GetForwardEndBeforePair(const PipelinePair &forward_end_pair);
int64_t GetMicroMax(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &forward_end);
int64_t GetSegment(const AnfNodePtr &node);
std::string GetWorldGroup();
int64_t GetRank();
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_PIPELINE_SPLIT_UTILS_H_
