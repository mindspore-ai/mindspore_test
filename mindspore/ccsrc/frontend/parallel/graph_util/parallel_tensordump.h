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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_TENSORDUMP_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_TENSORDUMP_H_

#include <vector>
#include <memory>
#include <utility>
#include <set>
#include <unordered_map>
#include <string>
#include "ir/anf.h"
#include "ir/manager.h"

namespace mindspore {
namespace parallel {

constexpr char IN_MODE[] = "in";
constexpr char OUT_MODE[] = "out";
constexpr char IN_INSERTED[] = "in_inserted";
constexpr char INPUT_OUTPUT[] = "input_output";
constexpr char VISITED_DUMP[] = "visited_dump";

std::string GetInModeSuffixedDumpPath(const std::string &ori_path);
std::string GetDumpInputOutputAttr(const AnfNodePtr &dump_node);
std::string GetDumpHookInputOutputAttr(const AnfNodePtr &dump_gradient);

// DumpGradient format in graph: CNode(kPrimDumpGradient, dump_path, x, input_output).
// DumpGradient pass through 'x' when forwarding, and generate TensorDump for gradient which is passed to 'x'.
constexpr int kDumpGradientSkipIndex = 2;

class RedistributionParallelTensorDumpHandler {
 public:
  explicit RedistributionParallelTensorDumpHandler(
    const std::vector<AnfNodePtr> &pre_nodes,
    const std::vector<std::pair<std::pair<AnfNodePtr, int>, std::vector<int>>> &next_nodes,
    const FuncGraphManagerPtr &fg_manager);
  void HandleDumpAfterRedistributionNode();

 private:
  void InsertNewTensorDump(const CNodePtr &dump_cnode, const AnfNodePtr &last_insert_redistribution_op,
                           const CNodePtr &node, const size_t pos_u, const FuncGraphPtr &func_graph,
                           const ScopePtr &scope, const std::string &dump_mode);

  AnfNodePtrList CollectNodePathBetween(AnfNodePtr start, std::pair<AnfNodePtr, int> end);
  AnfNodePtrList CollectDumpNodesAlongPath(const AnfNodePtrList &path, const FuncGraphManagerPtr &manager);
  AnfNodePtrList CollectBwdDumpHookAlongPath(const AnfNodePtrList &path);

  mindspore::CompactSet<std::string> GetScopeSetFromNodes(const std::vector<std::pair<AnfNodePtr, int>> &nodes);
  AnfNodePtrList DoFilterByScopeSet(const mindspore::CompactSet<std::string> &scope_set, const ScopePtr &cur_node_scope,
                                    const AnfNodePtrList &collects);

  void MakeOutModeDumpBwdHookAfterRedistribution(const std::vector<AnfNodePtr> &bwd_dump_hooks, const CNodePtr &node,
                                                 const size_t pos_u, const AnfNodePtr &last_insert_op);
  void MakeInModeDumpAfterRedistribution(const std::vector<AnfNodePtr> &dumps, const CNodePtr &node, const size_t pos_u,
                                         const AnfNodePtr &last_insert_op, const FuncGraphPtr &func_graph,
                                         const ScopePtr &scope);
  AnfNodePtr prenode_redistribution_;
  std::vector<std::pair<std::pair<AnfNodePtr, int>, std::vector<int>>> nodes_need_redistribution_;
  std::unordered_map<AnfNodePtr, std::vector<std::pair<AnfNodePtr, int>>> parent_to_successors_;
  FuncGraphManagerPtr fg_manager_;
};
using RedistributionDumpHandlerPtr = std::shared_ptr<RedistributionParallelTensorDumpHandler>;

class FwdCommunicationParallelTensorDumpHandler {
 public:
  explicit FwdCommunicationParallelTensorDumpHandler(const AnfNodePtr &node) : prior_(node) {}
  void MakeOutModeDumpBeforeFwdComm();
  void MakeInModeBwdHookBeforeFwdComm();
  void CollectDumpNodes(const AnfNodePtr &anchor, const bool is_multi_output);

 private:
  void CollectDumpNodesRecursively(const AnfNodePtr &start, const bool first_recursive,
                                   std::set<AnfNodePtr> *collect_visited);
  AnfNodePtrList dump_nodes_;
  AnfNodePtrList bwd_dump_hooks_;
  AnfNodePtr prior_;
};

using FwdCommDumpHandlerPtr = std::shared_ptr<FwdCommunicationParallelTensorDumpHandler>;
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_TENSORDUMP_H_
