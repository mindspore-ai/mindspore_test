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
#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_ANALYZER_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_ANALYZER_H

#include <set>
#include <vector>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "frontend/jit/pi/graph_capture/cfg.h"
#include "frontend/jit/pi/graph_capture/abstract_object.h"
#include "frontend/jit/pi/graph_capture/graph_build.h"
namespace mindspore {
namespace pijit {

class Graph;
class AbstractNode;
class ValueNode;
class CallNode;
class GraphAnalyzer;
using GraphAnalyzerPtr = std::shared_ptr<GraphAnalyzer>;

class GraphAnalyzer {
 public:
  // escaped_locals and captured.values do not intersect
  struct CapturedInfo {
    struct Info {
      // contains inputs and operations, fast to find
      mindspore::CompactSet<ValueNode *> values;
      // the inputs of operations
      std::vector<ValueNode *> inputs;
      // bytecode operations
      std::vector<ValueNode *> operations;
      // ordered outputs, used to restore stack and locals
      std::vector<ValueNode *> outputs;

      void clear();
      std::string ToString();
    };

    /**
     * for captured inputs, it's parameters, maybe unordered.
     * for captured outputs, it's ordered by stack values and alive locals.
     */
    Info captured_;

    /**
     * for captured outputs, it's maybe invalid type as graph's outputs.
     * if type is number/string/tensor/tuple, it's valid graph's output.
     * if type is list, convert to tuple as graph's output, then convert to tuple again in python.
     * if type is dict, convert to keys tuple and values tuple as graph's output, then convert to dict again in python.
     * Others, recreate in python.
     */
    Info outputs_optimize_;

    // a map of reconstruct alive node
    std::map<ValueNode *, ValueNode *> replaced_nodes_;

    /**
     * for interpret inputs, it's ordered and same as original function arguments.
     * if not break graph, outputs is return value, else outputs is ordered by stack values and alive locals.
     */
    Info interpret_;

    bool has_grad_ = false;

    void clear();
    std::string ToString();
  };

  struct GraphBreakInfo {
    bool is_break_at_call = false;
    // Contains only subgraphs, and top-graph is not in it.
    // The topmost subgraph is at the beginning of the list and the bottommost subgraph at the end.
    std::list<Graph *> captured_subgraphs;
  };

 public:
  explicit GraphAnalyzer(const GraphBuilderPtr &graph_builder)
      : graph_(graph_builder->GetGraph()), graph_builder_(graph_builder), info_(), graph_break_info_() {}

  void Analyze();

  auto &GetCaptureInfo() { return info_; }
  const auto &GetCaptureInfo() const { return info_; }

  bool NeedInterpret() const { return need_interpret_; }

  const GraphBreakInfo &graph_break_info() const { return graph_break_info_; }

 private:
  void BeforeAnalyze();

  // Collect top-graph closure side-effect nodes.
  void CollectClosureSideEffect();

  // UD analyze
  void UseDefAnalyze();
  void GraphArgumentOpt();
  bool AnalyzeTopGraphAliveNodes(const std::vector<ValueNode *> &alive_nodes);
  void UpdateCapturedOrder();
  void AnalyzeSubGraphBreakRecursive(CallNode *call_node);
  std::vector<ValueNode *> SubGraphUseDefAnalyze(Graph *graph);
  bool AnalyzeSubGraphAliveNodes(const std::vector<ValueNode *> &alive_nodes, Graph *graph,
                                 std::vector<ValueNode *> *graph_outputs);

  void CollectCapturedAndInterpret();

  void ExpandGraphOutput();
  void UpdateUseDefNode();

  bool NeedSkipAddGraphOutput(ValueNode *node);
  ValueNode *MutateSequenceNode(ValueNode *node);
  ValueNode *MutateNamedtupleNode(ValueNode *tuple_node, ValueNode *namedtuple_node);
  std::pair<ValueNode *, ValueNode *> MutateDictNode(ValueNode *node);
  // find or insert
  ValueNode *GetBuiltinMethodNode(std::vector<ValueNode *> *operations, const std::string &method,
                                  const std::string &cls_method = "");

  Graph *graph_;
  GraphBuilderPtr graph_builder_;
  CapturedInfo info_;
  GraphBreakInfo graph_break_info_;
  bool need_interpret_{false};
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_ANALYZER_H
