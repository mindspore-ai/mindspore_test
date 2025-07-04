/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAD_ACCUMULATION_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_GRAD_ACCUMULATION_UTILS_H_

#include <utility>
#include <vector>
#include <unordered_map>
#include <string>
#include "ir/anf.h"
#include "ir/manager.h"

namespace mindspore {
namespace parallel {
void SetGradAccumulationStep(const std::vector<AnfNodePtr> &all_nodes);
void TagMicroBatchStart(const FuncGraphManagerPtr &manager, const std::vector<AnfNodePtr> &all_nodes);
void TagMicroBatchEnd(const FuncGraphManagerPtr &manager, const std::vector<AnfNodePtr> &all_nodes);
void TagMicroBatchBpEndInCellShare(const FuncGraphPtr &root, const FuncGraphManagerPtr &manager);
void ExtractMicroBatchBorderNodes(const FuncGraphPtr &root,
                                  std::unordered_map<int64_t, std::vector<CNodePtr>> *forward_start,
                                  std::unordered_map<int64_t, std::vector<CNodePtr>> *backward_end);
void ReorderGradAccumulation(const FuncGraphPtr &root,
                             const std::unordered_map<int64_t, std::vector<CNodePtr>> &forward_start,
                             const std::unordered_map<int64_t, std::vector<CNodePtr>> &backward_end);
void TagMicroBatchBpEndPrim(const FuncGraphPtr &root);
void TagMicroBatchBpEnd(const FuncGraphPtr &root);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_GRAPH_UTIL_PIPELINE_SPLIT_UTILS_H_
