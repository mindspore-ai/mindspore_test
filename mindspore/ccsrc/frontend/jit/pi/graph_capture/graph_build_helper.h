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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_BUILD_HELPER_H
#define MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_BUILD_HELPER_H
#include <vector>
#include <utility>
#include <memory>

#include "frontend/jit/pi/utils/utils.h"
#include "frontend/jit/pi/utils/stop_trace_reason.h"

namespace mindspore {
namespace pijit {
class GraphBuilder;
using GraphBuilderPtr = std::shared_ptr<GraphBuilder>;
class ValueNode;
class CallNode;
class AbstractWrapper;
using AbstractWrapperPtr = std::shared_ptr<AbstractWrapper>;
using AbstractWrapperPtrList = std::vector<AbstractWrapperPtr>;
class FuncGraphBuilder;
using FuncGraphBuilderPtr = std::shared_ptr<FuncGraphBuilder>;

struct CallInfo {
  ValuePtr value;
  py::object object;
  AbstractWrapperPtrList inputs_abstract_wrapper;
};

class GraphBuildHelper : public std::enable_shared_from_this<GraphBuildHelper> {
 public:
  GraphBuildHelper() = default;

  size_t call_info_size() const { return call_info_list_.size(); }
  ValuePtr GetValue(size_t index);
  py::object GetObject(size_t index);
  AbstractWrapperPtrList GetInputsAbstractWrapper(size_t index);

  AbstractWrapperPtr Prepare(GraphBuilder *graph_builder, const CallInfo &call_info);
  virtual AbstractWrapperPtr Build(GraphBuilder *graph_builder, CallNode *call_node) = 0;

 protected:
  virtual AbstractWrapperPtr PrepareInner(GraphBuilder *graph_builder, const CallInfo &call_info) = 0;

  void AddCallInfo(const CallInfo &call_info);
  void CheckCallInfoListSize(size_t index);
  std::vector<CallInfo> call_info_list_;
};
using GraphBuildHelperPtr = std::shared_ptr<GraphBuildHelper>;

struct GradInfo {
  bool get_all_;
  bool get_by_list_;
  bool sens_param_;
  bool get_by_position_;
  bool has_aux_;
  bool get_value_;
  bool return_ids_;
  bool merge_forward_;
};

class GradGraphBuildHelper : public GraphBuildHelper {
 public:
  AbstractWrapperPtr Build(GraphBuilder *graph_builder, CallNode *call_node) override;

 protected:
  AbstractWrapperPtr PrepareInner(GraphBuilder *graph_builder, const CallInfo &call_info) override;

 private:
  std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>> BuildForwardGraph(GraphBuilder *graph_builder,
                                                                              CallNode *call_node);
  void HandleCustomBProp(const FuncGraphPtr &graph, const py::object &obj) const;
  void HandleGradForwardSideEffect(GraphBuilder *graph_builder, const FuncGraphPtr &forward_fg,
                                   const AbstractWrapperPtr &grad, const GraphBuilderPtr &subgraph_builder,
                                   CallNode *call_node);
  AbstractWrapperPtrList HandleInputsForGrad(GraphBuilder *graph_builder, CallNode *call_node,
                                             BindArgumentsHelper<ValueNode *> forward_inputs);
  AbstractWrapperPtr BuildGradNode(const FuncGraphBuilderPtr &func_graph_builder, const AbstractWrapperPtr &key,
                                   const FuncGraphPtr &forward_fg, const AbstractWrapperPtrList &inputs);
  AbstractWrapperPtr HandleGrad(const FuncGraphBuilderPtr &func_graph_builder, const AbstractWrapperPtr &key,
                                const FuncGraphPtr &forward_fg, const AbstractWrapperPtrList &inputs);
  FuncGraphPtr BuildCallForwardGraphForGrad(const FuncGraphPtr &fg, const std::vector<size_t> &arg_len, bool is_cell);
  void UpdateGradInfo(const ValuePtr &meta);
  GradInfo grad_info_;
};
using GradGraphBuildHelperPtr = std::shared_ptr<GradGraphBuildHelper>;

GraphBuildHelperPtr GraphBuildHelperFactory(const py::object &object);
GraphBuildHelperPtr GetCallNodeGraphBuildHelper(CallNode *call_node);
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PI_GRAPH_CAPTURE_GRAPH_BUILD_HELPER_H
