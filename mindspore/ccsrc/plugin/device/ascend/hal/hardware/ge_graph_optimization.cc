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

#include "plugin/device/ascend/hal/hardware/ge_graph_optimization.h"
#include <string>
#include <memory>
#include "mindspore/ops/op_def/framework_ops.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_optimization.h"
#include "plugin/device/ascend/optimizer/ge_backend_optimization.h"
#include "plugin/device/ascend/optimizer/backend_common_unify_mindir.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/debug/profiler/profiling.h"
#ifndef ENABLE_SECURITY
#include "include/common/debug/dump_proto.h"
#endif

namespace mindspore {
namespace device {
namespace ascend {
namespace {
void MarkRefGraph(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Mark graph is ref graph: " << kernel_graph->graph_id();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_kbk = ms_context->IsKByKExecutorMode();
  auto manager = kernel_graph->manager();
  if (manager == nullptr || kernel_graph->has_attr(kIsRefGraph)) {
    return;
  }
  for (const auto &node : TopoSort(kernel_graph->get_return(), SuccDeeperSimple, AlwaysInclude)) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto is_side_effect = common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, cnode) &&
                          common::AnfAlgo::GetNodeAttr<bool>(cnode, GRAPH_FLAG_SIDE_EFFECT_MEM);
    if (!(is_side_effect && cnode->fullname_with_scope().find("optimizer") != std::string::npos)) {
      continue;
    }
    for (const auto &node_pair : manager->node_users()[cnode]) {
      if (IsPrimitiveCNode(node_pair.first, prim::kPrimUpdateState)) {
        kernel_graph->set_attr(kIsRefGraph, MakeValue(true));
        MS_LOG(INFO) << "graph is ref graph: " << kernel_graph->graph_id();
        if (!is_kbk) {
          return;
        }
        common::AnfAlgo::SetNodeAttr(kFromRefGraph, MakeValue(true), cnode);
        break;
      }
    }
  }
}
}  // namespace

void GEGraphOptimization::OptimizeGEGraph(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(memo);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  MS_LOG(DEBUG) << "Status record: start optimize ge graph. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize ge graph. graph id: " << graph->graph_id();
  }
  MarkRefGraph(graph);
  opt::GEBackendOptimizeACL(graph);
  opt::GEBackendOptimization(graph);
  if (const auto &gk = graphkernel::GraphKernelFlags::GetInstance(); gk.IsEnableGraphKernel()) {
    if (gk.kernel_generator != "DVM") {
      graphkernel::GraphKernelOptimize(graph);
      graph->SetExecOrderByDefault();
    } else {
      MS_LOG(WARNING) << "In ge graph, GraphKernel fusion is not supported for the DVM kernel_generator.";
    }
  }
  for (auto &child_graph : graph->child_graph_order()) {
    if (child_graph.lock()->has_flag(kFlagGeKernel)) {
      continue;
    }
    OptimizeGEGraph(child_graph.lock(), memo);
  }
  MS_LOG(DEBUG) << "Status record: end optimize ge graph. graph id: " << graph->graph_id();
}

void GEGraphOptimization::OptimizeACLGraph(const KernelGraphPtr &graph, std::set<KernelGraphPtr> *const memo) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(memo);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph. graph id: " << graph->graph_id();
  }
  MarkRefGraph(graph);
  opt::AscendUnfoldInputsForSpecialNodes(graph);
  opt::GEBackendOptimizeACL(graph);
  for (auto &child_graph : graph->child_graph_order()) {
    if (child_graph.lock()->has_flag(kFlagGeKernel)) {
      OptimizeGEGraph(child_graph.lock(), memo);
    } else {
      OptimizeACLGraph(child_graph.lock(), memo);
    }
  }
  MS_LOG(DEBUG) << "Status record: end optimize acl graph. graph id: " << graph->graph_id();
}

void GEGraphOptimization::OptimizeACLGraphAfterKernelSelect(const KernelGraphPtr &graph,
                                                            std::set<KernelGraphPtr> *const memo) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(memo);
  if (memo->find(graph) != memo->end()) {
    return;
  }
  memo->insert(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph after kernel select. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph after kernel select. graph id: " << graph->graph_id();
  }
  if (!graph->is_from_single_op() && graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel()) {
    graphkernel::GraphKernelOptimize(graph);
    graph->SetExecOrderByDefault();
  }
  opt::GEBackendOptimizeACLAfterKernelSelect(graph);
  if (!graph->is_from_single_op() && graphkernel::GraphKernelFlags::GetInstance().IsEnableKernelPacket() &&
      common::AnfAlgo::IsDynamicGraph(graph)) {
    graphkernel::KernelPacketOptimize(graph);
    graph->SetExecOrderByDefault();
  }
  for (auto &child_graph : graph->child_graph_order()) {
    if (child_graph.lock()->has_flag(kFlagGeKernel)) {
      continue;
    }
    OptimizeACLGraphAfterKernelSelect(child_graph.lock(), memo);
  }
  MS_LOG(DEBUG) << "Status record: end optimize acl graph after kernel select. graph id: " << graph->graph_id();
}

void GEGraphOptimization::OptimizeACLGraphAfterCreateKernel(const KernelGraphPtr &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  int execution_mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  // pynaitve process the pass in GEBackendOptimizeACLAfterKernelSelect
  if (execution_mode == kPynativeMode) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph after create kernel. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph after create kernel. graph id: " << graph->graph_id();
  }
  opt::AclAfterCreateKernel(graph);
  MS_LOG(DEBUG) << "Status record: end optimize acl graph after create kernel. graph id: " << graph->graph_id();
}

void GEGraphOptimization::OptimizeACLGraphAfterInline(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(DEBUG) << "Status record: start optimize acl graph after inline. graph id: " << graph->graph_id();
  // empty graph dont entry to backend
  if (graph->execution_order().empty()) {
    MS_LOG(DEBUG) << graph->ToString() << " is empty graph.";
    AnfAlgo::InsertMakeTupleForOutput(NOT_NULL(graph));
    graph->set_executable(false);
    MS_LOG(DEBUG) << "Status record: end optimize acl graph after inline. graph id: " << graph->graph_id();
  }
  opt::GEAfterInlineOptimize(graph);
  MS_LOG(DEBUG) << "Status record: end optimize acl graph after inline. graph id: " << graph->graph_id();
}

void GEGraphOptimization::UnifyMindIR(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start unify mindir. graph id: " << graph->graph_id();
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(unify_mindir);
  opt::CommonUnifyMindIR(graph);
  opt::GEUnifyMindIR(graph);
  PROF_END(unify_mindir);
  (void)profiler::CollectHostInfo("Ascend", "Graph Optimization", "UnifyMindIR", start_time, profiler::GetClockSyscnt(),
                                  0);
  MS_LOG(INFO) << "Status record: end unify mindir. graph id: " << graph->graph_id();
}

void GEGraphOptimization::GEMindIRPass(const KernelGraphPtr &graph) const { opt::GEUnifyMindIR(graph); }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
