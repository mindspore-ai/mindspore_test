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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_GRAPH_COMPILER_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_GRAPH_COMPILER_H_

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <set>
#include "utils/hash_map.h"
#include "backend/ge_backend/runtime//actor/actor_common.h"
#include "backend/ge_backend/runtime//control_node_parser.h"
#include "backend/common/session/session_basic.h"
#include "backend/common/session/session_factory.h"
#include "ir/tensor.h"
#include "include/backend/visible.h"
#include "kernel/framework_utils.h"
#include "common/device_type.h"
#include "backend/ge_backend/executor/ge_graph_executor.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
using session::CallBackFunc;
using session::GraphOutputInfo;
using session::InputInfo;
using session::KernelGraph;
using session::KernelWithIndex;
using tensor::TensorPtr;

const char kModelNameRuntime[] = "Runtime";
const char kEventDeviceInit[] = "DeviceInit";
const char kEventCompileGraph[] = "CompileGraph";
const char kEventRunGraph[] = "RunGraph";
const char kStageDeviceInit[] = "DeviceInit";
const char kStageCompileGraphs[] = "CompileGraphs";
const char kStageGraphPartition[] = "GraphPartition";
const char kStageConstructKernelGraph[] = "ConstructKernelGraph";
const char kStageOptimizeGraph[] = "OptimizeGraph";
const char kStageCreateKernel[] = "CreateKernel";
const char kStageGraphTransform[] = "GraphTransform";
const char kStageBuild[] = "Build";
const char kStageLink[] = "Link";
const char kStageOptimize[] = "Optimize";
const char kStageRunGraph[] = "RunGraph";
const char kStageGetInputs[] = "GetInputs";
const char kStageRun[] = "Run";
const char kStageConstructOutputs[] = "ConstructOutputs";

// Position of kernel with index, the value pair<branch_id, vector<pos>> means the branch id of the kernel and the pos
// of the kernel. Generally, there is only one branch, and the branch id is 0 at this time. In control flow, there are
// multiple branch scenarios, and pos represents the position of the kernel in the branch.
using KernelMapPosition = std::map<KernelWithIndex, std::vector<size_t>, session::KernelWithIndexCmp>;

// The graph compiler info generated by graph compiler is the express of executable graph.
// The device context is unified interface of interaction with device of corresponding graph.
// The tensors mask is used to distinguish input tensor's type.
// The input tensor is used to link graphs in the dynamic build scenario.
// The control node is used to link graphs in the control flow scenario.
// The control node parser is used to parse the edge info in control nodes.
// The origin parameters order is used to correspond to the input args.
// The origin outputs order is used to correspond to the output args.
// The need_erase means need erase this GraphCompilerInfo object after run actor set.
struct BACKEND_EXPORT GraphCompilerInfo {
  GraphCompilerInfo(const std::vector<KernelGraphPtr> &graphs, const std::vector<std::vector<int64_t> *> &tensors_mask,
                    const std::vector<std::vector<TensorPtr> *> &input_tensors,
                    const std::vector<AnfNodePtr> &control_nodes,
                    const std::vector<AnfNodePtr> &origin_parameters_order, const ControlNodeParserPtr &parser,
                    const KernelMapPosition &origin_outputs_order, size_t outputs_num, size_t inputs_num,
                    const std::string &name, bool need_erase, GraphExecutionStrategy strategy,
                    const std::string &graph_phase, const FuncGraphPtr &root_graph,
                    const std::shared_ptr<backend::ge_backend::GeGraphExecutor> &graph_executor)
      : graphs_(graphs),
        tensors_mask_(tensors_mask),
        input_tensors_(input_tensors),
        control_nodes_(control_nodes),
        control_node_parser_(parser),
        origin_parameters_order_(origin_parameters_order),
        origin_outputs_order_(origin_outputs_order),
        outputs_num_(outputs_num),
        inputs_num_(inputs_num),
        name_(name),
        need_erase_(need_erase),
        exist_flatten_concat_(false),
        strategy_(strategy),
        graph_phase_(graph_phase),
        root_graph_(root_graph),
        graph_executor_(graph_executor) {}
  ~GraphCompilerInfo();
  std::vector<KernelGraphPtr> graphs_;
  std::vector<std::vector<int64_t> *> tensors_mask_;
  std::vector<std::vector<TensorPtr> *> input_tensors_;
  std::vector<AnfNodePtr> control_nodes_;
  ControlNodeParserPtr control_node_parser_;
  std::vector<AnfNodePtr> origin_parameters_order_;
  mutable mindspore::HashMap<AnfNodePtr, std::vector<std::pair<KernelWithIndex, KernelWithIndex>>>
    origin_parameters_to_backend_parameters_;
  KernelMapPosition origin_outputs_order_;
  size_t outputs_num_;
  size_t inputs_num_;
  std::string name_;
  bool need_erase_;
  mutable bool exist_flatten_concat_;
  mutable GraphExecutionStrategy strategy_;
  std::string graph_phase_;
  FuncGraphPtr root_graph_;
  std::shared_ptr<backend::ge_backend::GeGraphExecutor> graph_executor_;
};

class GraphCompiler {
 public:
  explicit GraphCompiler(std::shared_ptr<backend::ge_backend::GeGraphExecutor> graph_executor) {
    session_ = session::SessionFactory::Get().Create(kSessionBasic);
    graph_executor_ = graph_executor;
  }
  ~GraphCompiler() = default;

  // Construct kernel graph from anf nodes list and compile kernel graph in Graph mode,
  // the detailed implementation of compiling graph is in 'CompileGraphImpl'.
  GraphId CompileGraph(const GraphSegmentPtr &segment, const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                       const backend::BackendJitConfig &backend_jit_config);

  GraphId CompileGraph(const KernelGraphPtr &kernel_graph, const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes);

  // Get graph by graph id, if not exist return nullptr, used in Graph mode.
  KernelGraphPtr Fetch(GraphId graph_id) const;

  void ClearGraphBuildMember() {
    MS_EXCEPTION_IF_NULL(session_);
    session_->ClearGraphBuildMember();
  }

  // Register a summary callback function, which is called in the final stages of summary.
  void RegisterSummaryCallBackFunc(const CallBackFunc &callback) const;
  // Execute graph summary.
  void Summary(const std::vector<KernelGraphPtr> &graphs) const;

  // The implementation of compiling graph in Graph Mode, including optimizing graph,
  // setting operator info, creating kernel and transforming kernel graph to ActorSet.
  GraphId CompileGraphImpl(const KernelGraphPtr &graph) const;
  const session::SessionPtr &session_ptr() const { return session_; }

 private:
  DISABLE_COPY_AND_ASSIGN(GraphCompiler);

  // Create device address for all anf nodes of graph.
  void CreateDeviceAddress(const KernelGraphPtr &graph) const;

  // Set Graph's dependencies for pre_graph and post_graph
  void SetGraphDependency(const KernelGraphPtr &graph, const GraphSegmentPtr &segment) const;

  // The member variable 'session_' will be removed after removing session module.
  // Now all the GraphCompiler share the same 'session_'.
  session::SessionPtr session_;
  std::shared_ptr<backend::ge_backend::GeGraphExecutor> graph_executor_;
};
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_GRAPH_COMPILER_H_
