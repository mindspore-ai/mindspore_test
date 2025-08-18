/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_ARGUMENTS_OPTIMIZER_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_ARGUMENTS_OPTIMIZER_H

#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <utility>
#include <vector>
#include "frontend/jit/pi/graph_capture/graph.h"

namespace mindspore {
namespace pijit {

class GraphArgumentOptimizer;
using GraphArgumentOptimizerPtr = std::shared_ptr<GraphArgumentOptimizer>;
using ArgsStatusMap = std::unordered_map<const AObject *, std::pair<ValueNode *, int>>;

class GraphArgumentOptimizer {
 public:
  explicit GraphArgumentOptimizer(Graph *graph, ArgsStatusMap &status) : graph_(graph), status_(status) {}
  virtual ~GraphArgumentOptimizer() = default;

  /// \brief Get a new instance of graph argument optimizer.
  ///
  /// \param[in] graph The top graph will be optimized.
  ///
  /// \return The instance to optimize the arguments of graph.
  static GraphArgumentOptimizerPtr GetNewInstance(Graph *graph);
  /// \brief Set the bytecode nodes corresponding to the graph execution.
  ///
  /// \param[in] nodes The nodes corresponding to the graph.
  ///
  /// \note The nodes must be belong to the top graph.
  void SetGraphNodes(const std::vector<ValueNode *> &nodes) { nodes_ = nodes; }
  /// \brief The processing entry of graph arguments optimizer.
  ///
  /// \param[in] outputs The return value of the graph.
  ///
  /// \note All of the duplicate and unused parameters will be removed.
  void Run(const std::vector<ValueNode *> &outputs);
  /// \brief Get the optimized graph arguments.
  ///
  /// \return The optimized graph arguments.
  ///
  /// \note Function Run must be called before calling this function.
  const std::vector<ValueNode *> &GetArguments() { return arguments_; }

 private:
  /// \brief Determine whether the graph is top graph.
  ///
  /// \return True : The graph is top graph, False : The graph is not top graph.
  bool IsTopGraph() const { return graph_->GetParent() == nullptr || graph_->GetParent() == graph_; }
  /// \brief Determine whether the variable is a argument or contains a argument.
  ///
  /// \param[int] vobj The object info corresponding to the variable.
  ///
  /// \return Whether the variable is a argument or contains a argument.
  bool IsUsingAnyArgument(AObject *vobj) const;
  /// \brief Initialize the usage status of the graph arguments.
  ///
  /// \note Initialize data for the environment.
  void InitializeArgumentsUsageStatus();
  /// \brief Collect the captured nodes.
  ///
  /// \return The captured nodes.
  ///
  /// \note The nodes executed in the graph, need to be analyzed.
  std::vector<ValueNode *> CollectCapturedNodes() const;
  /// \brief Collect the nodes that use the argument(s).
  ///
  /// \param[in] nodes All the nodes will be executed in the graph.
  ///
  /// \return The nodes that use the argument(s).
  std::vector<ValueNode *> CollectNodesUsingArgument(const std::vector<ValueNode *> &nodes);
  /// \brief Mark all arguments used.
  ///
  /// \param[in] vobj The vobj of node.
  void MarkAllArguments(AObject *vobj);
  /// \brief Analyze the call node.
  ///
  /// \param[in] call_node The node will be analyzed.
  ///
  /// \return Ture : if all the input need mark used, else False.
  bool AnalyzeCallNode(CallNode *call_node);
  /// \brief Analysis the usage status of the arguments.
  ///
  /// \param[in] nodes All the nodes that use the argument(s).
  void AnalyzeArgumentsUsageStatus(const std::vector<ValueNode *> &nodes);
  /// \brief Collect the used arguments in graph exec.
  ///
  /// \return The used arguments in graph exec.
  std::vector<ValueNode *> CollectUsedInputsInGraph() const;
  /// \brief Collect the arguments used as constant in graph exec.
  ///
  /// \return The arguments used as constant in graph exec.
  std::vector<ValueNode *> CollectConstantArguments() const;
  /// \brief Collect the duplicate arguments of the top graph.
  ///
  /// \return The duplicate arguments of the top graph.
  std::unordered_map<ValueNode *, ValueNode *> CollectDuplicateArguments() const;
  /// \brief Add len guard for expand parameters.
  ///
  /// \note Add len guard for expand parameters.
  void GuardExpandParameters();
  /// \brief Remove the unused arguments from the graph.
  ///
  /// \param[in] constant_args The arguments used as constant in graph exec.
  /// \param[in] args The duplicate arguments.
  ///
  /// \note The function Will modify the computational graph and guards.
  void EliminateRedundantArguments(const std::vector<ValueNode *> &constant_args,
                                   const std::unordered_map<ValueNode *, ValueNode *> &args);

  /// \brief The graph corresponding to the nodes being captured.
  Graph *graph_;
  /// \brief The arguments of the top graph.
  std::vector<ValueNode *> arguments_;
  /// \brief The bytecode nodes corresponding to the graph execution.
  std::vector<ValueNode *> nodes_;
  /// \brief The usage status of the arguments.
  ArgsStatusMap status_;
  /// \brief The duplicate arguments of the top graph.
  std::unordered_map<const AObject *, std::unordered_set<ValueNode *>> duplicate_args_;
  /// \brief The nodes used the arguments.
  std::vector<ValueNode *> related_nodes_;
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_ARGUMENT_OPTIMIZER_H
