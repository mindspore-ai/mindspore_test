/**
 * Copyright 2024-2025Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_TENSORDUMP_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_TENSORDUMP_H_

#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include <string>
#include "ir/anf.h"
#include "ir/manager.h"

namespace mindspore {
namespace parallel {
class ParallelTensorDumpHandler {
 public:
  explicit ParallelTensorDumpHandler(
    const std::vector<std::pair<std::pair<AnfNodePtr, int>, std::vector<int>>> &next_nodes);
  void HandleParallelTensorDump();

 private:
  std::vector<std::pair<std::pair<AnfNodePtr, int>, std::vector<int>>> nodes_need_redistribution_;
  std::unordered_map<AnfNodePtr, std::vector<std::pair<AnfNodePtr, int>>> parent_to_successors_;
  AnfNodePtrList CollectSuccessorDumpNodes(const AnfNodePtr &parent_of_dump_nodes, const FuncGraphManagerPtr &manager);

  void InsertNewTensorDump(const CNodePtr &dump_cnode, const AnfNodePtr &last_insert_redistribution_op,
                           const CNodePtr &node, const size_t pos_u, const FuncGraphPtr &func_graph,
                           const ScopePtr &scope, const std::string &dump_mode);
  void ProcessTensorDumps(const std::vector<AnfNodePtr> &dumps, const CNodePtr &node, const size_t pos_u,
                          const AnfNodePtr &last_insert_op, const FuncGraphPtr &func_graph, const ScopePtr &scope);
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_TENSORDUMP_H_
