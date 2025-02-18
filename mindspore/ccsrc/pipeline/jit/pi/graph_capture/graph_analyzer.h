/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_PI_JIT_GRAPH_CAPTURE_GRAPH_ANALYZER_H
#define MINDSPORE_PI_JIT_GRAPH_CAPTURE_GRAPH_ANALYZER_H

#include <set>
#include <vector>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "pipeline/jit/pi/graph_capture/cfg.h"
#include "pipeline/jit/pi/graph_capture/abstract_object.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include "utils/convert_utils_base.h"
namespace mindspore {
namespace pijit {

class Graph;
class AbstractNode;
class ValueNode;
class CallNode;
class GraphAnalyzer;
class MindGraphAnalyzer;
using GraphAnalyzerPtr = std::shared_ptr<GraphAnalyzer>;
using MindGraphAnalyzerPtr = std::shared_ptr<MindGraphAnalyzer>;

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

    struct GraphInputs {
      std::vector<ValueNode *> args;
      std::vector<ValueNode *> globals;
      ValueNode *vargs = nullptr;
      ValueNode *kwargs = nullptr;

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

    /**
     * Store all collected graph inputs.
     * If no graph is generated, graph_inputs_ should be empty.
     */
    GraphInputs graph_inputs_;

    bool has_grad_ = false;

    void clear();
    std::string ToString();
  };

  explicit GraphAnalyzer(Graph *g) : graph_(g) {}
  static GraphAnalyzerPtr Creator(const GraphBuilderPtr &g) {
    return g->trace_flag() ? std::static_pointer_cast<GraphAnalyzer>(std::make_shared<MindGraphAnalyzer>(g))
                           : std::make_shared<GraphAnalyzer>(g->GetGraph());
  }
  auto &GetCaptureInfo() { return info_; }
  const auto &GetCaptureInfo() const { return info_; }
  virtual void Analyze();
  bool HasTensorOperation() const;
  virtual bool NeedInterpret() const { return need_interpret_; }

  const auto &alive_locals() const { return alive_locals_; }

 protected:
  // optimize
  void OptimizeSideEffectRecord() const;

  // rollback
  virtual void ResetSideEffectRecord() const;

  void AddToEscaped(ValueNode *value);
  // UD analyze
  virtual void UseDefAnalyze();
  std::vector<ValueNode *> GetAliveLocals(Graph *g);
  virtual bool AnalyzeAliveLocals(std::vector<ValueNode *> aliveNodes);
  virtual void CollectCapturedInputs();
  virtual void CollectCapturedAndInterpret();
  virtual void CollectGraphInputs();
  bool ProduceInterpretValue(ValueNode *v);

  bool need_interpret_;
  Graph *graph_;
  CapturedInfo info_;
  std::vector<int> alive_locals_;

 private:
  bool AnalyzeRecursive(Graph *g);
  bool AnalyzeCall(CallNode *);
  bool TryToCapture(AbstractNode *value);
  bool AddToCaptured(ValueNode *value);
  bool HandleCallableToGraph(AObject *f);
  void CleanCapturedValue();
};

class MindGraphAnalyzer : public GraphAnalyzer {
 public:
  explicit MindGraphAnalyzer(const GraphBuilderPtr &g) : GraphAnalyzer(g->GetGraph()), graph_builder_(g) {}
  void Analyze() override;

 protected:
  // UD analyze
  void UseDefAnalyze() override;
  void CollectCapturedInputs() override;
  void CollectGraphInputs() override;
  void UpdateCapturedOrder();
  void CollectCapturedAndInterpret() override;
  bool AnalyzeAliveLocals(std::vector<ValueNode *> aliveNodes) override;
  void ResetSideEffectRecord() const override;
  void UpdateUseDefNode();

  bool NeedSkipAddGraphOutput(ValueNode *node);

  // find or insert
  ValueNode *GetBuiltinMethodNode(std::vector<ValueNode *> *operations, const std::string &method,
                                  const std::string &cls_method = "");

  GraphBuilderPtr graph_builder_ = nullptr;

 private:
  ValueNode *MutateSequenceNode(ValueNode *node);
  ValueNode *MutateNamedtupleNode(ValueNode *tuple_node, ValueNode *namedtuple_node);
  std::pair<ValueNode *, ValueNode *> MutateDictNode(ValueNode *node);
};
}  // namespace pijit
}  // namespace mindspore

#endif  // MINDSPORE_PI_JIT_GRAPH_CAPTURE_GRAPH_ANALYZER_H
