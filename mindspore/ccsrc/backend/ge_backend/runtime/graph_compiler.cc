/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License"){}
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

#include "backend/ge_backend/runtime/graph_compiler.h"
#include <numeric>
#include <map>
#include <utility>
#include <algorithm>
#include <functional>
#include <list>
#include <regex>
#include "backend/ge_backend/runtime/graph_scheduler.h"
#include "runtime/device/device_address_utils.h"
#include "include/backend/device_address.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "include/common/utils/convert_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/optimizer/common_backend_optimization.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "kernel/framework_utils.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/backend/optimizer/helper.h"
#include "base/base_ref_utils.h"
#include "include/common/debug/dump_proto.h"
#include "include/common/utils/parallel_context.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/anf_ir_dump.h"
#endif
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "include/backend/optimizer/graph_optimizer.h"
#include "include/common/profiler.h"
#include "include/common/utils/compile_cache_context.h"
#include "utils/phase.h"
#include "pipeline/jit/ps/base.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "runtime/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
namespace {
void SetSummaryNodesRefCount(const KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->summary_node_exist()) {
    return;
  }

  const std::map<std::string, std::pair<AnfNodePtr, int>> &summary_nodes = graph->summary_nodes();
  if (summary_nodes.empty()) {
    return;
  }

  for (const auto &item : summary_nodes) {
    const AnfNodePtr &node = item.second.first;
    size_t index = IntToSize(item.second.second);
    auto device_address = AnfAlgo::GetMutableOutputAddr(node, index, false);
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->set_original_ref_count(SIZE_MAX);
    device_address->ResetRefCount();
  }
}

bool IsEnableZeroCopy(bool run_in_pynative) {
  if (run_in_pynative) {
    return false;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool task_sink = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  bool is_multi_graph_sink = ms_context->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK);
  // If the run mode is not subgraph sink, the flag should not be set.
  if (!task_sink || is_multi_graph_sink) {
    // Jit level O2 in graph mode will execute ge and zero copy flag should be set.
    if (ms_context->get_param<std::string>(MS_CTX_JIT_LEVEL) != "O2" ||
        ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kGraphMode) {
      return false;
    }
  }

  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  bool is_parallel_mode = parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel ||
                          parallel_mode == parallel::kHybridParallel || parallel_mode == parallel::kDataParallel;
  // If there are auto parallel in graph, the flag should not be set. In parallel, the continue memory in communication
  // ops not support addr change.
  // force zero copy when use ge
  bool is_enable_ge = ms_context->backend_policy() == "ge";
  if (is_parallel_mode && !is_enable_ge) {
    return false;
  }
  return true;
}

void UseCacheToCompileGraphImpl(const KernelGraphPtr &graph, const DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);

  auto &compile_cache_context = CompileCacheContext::GetInstance();
  uint64_t start_time = profiler::GetClockSyscnt();
  compile_cache_context.SetFusionOpBuildInfoFlag(true);
  device_context->GetKernelExecutor(false)->CreateKernel(graph->execution_order());
  compile_cache_context.SetFusionOpBuildInfoFlag(false);
  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCreateKernel, start_time,
                                  profiler::GetClockSyscnt(), 1);

  // Update needed dump kernels for mindRT.
  DumpJsonParser::GetInstance().UpdateNeedDumpKernels(*graph.get());
  if (graph->is_dynamic_shape()) {
    auto profiler_manage_inst = profiler::ProfilerManager::GetInstance();
    MS_EXCEPTION_IF_NULL(profiler_manage_inst);
    profiler_manage_inst->SetNetDynamicShapeStatus();
  }
}

void ResetNodeId(const std::vector<KernelGraphPtr> &graphs) {
  static mindspore::HashMap<std::string, int> node_ids;
  for (const auto &graph : graphs) {
    MS_EXCEPTION_IF_NULL(graph);
    if (graph->memory_managed_by_ge()) {
      continue;
    }

#ifdef ENABLE_DUMP_IR
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    bool save_graphs = context->CanDump(kIntroductory);
    if (save_graphs) {
      std::string file_name = "graph_before_reset_id_" + std::to_string(graph->graph_id()) + ".ir";
      DumpIR(file_name, graph, true, kWholeStack);
    }
#endif
    const auto &all_nodes = TopoSort(graph->get_return(), SuccDeeperSimple);
    for (const auto &node : all_nodes) {
      if (node != nullptr && node->isa<CNode>()) {
        const auto &cnode = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        const auto &fullname = cnode->fullname_with_scope();
        auto op_index = fullname.rfind("-op");
        if (op_index != string::npos) {
          auto scope_prefix = fullname.substr(0, op_index);
          if (node_ids.find(scope_prefix) == node_ids.end()) {
            node_ids[scope_prefix] = 0;
          } else {
            node_ids[scope_prefix]++;
          }
          cnode->set_fullname_with_scope(scope_prefix + "-op" + std::to_string(node_ids[scope_prefix]));
        }
      }
    }
  }
}
}  // namespace

GraphId GraphCompiler::CompileGraph(const GraphSegmentPtr &segment,
                                    const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                                    const DeviceContext *device_context, device::RunMode run_mode,
                                    bool run_in_pynative) {
  MS_EXCEPTION_IF_NULL(segment);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_LOG(INFO) << "Status record: start compile graph.";
  auto nodes = segment->nodes_;
  auto device_target = device_context->GetDeviceType();
  // Generate kernel graph.
  uint64_t start_time = profiler::GetClockSyscnt();
  PROF_START(ConstructKernelGraph);
  auto kernel_graph =
    session_->ConstructKernelGraph(nodes, io_nodes.second, device_target, true, IsEnableZeroCopy(run_in_pynative));
  PROF_END(ConstructKernelGraph);

  (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageConstructKernelGraph, start_time,
                                  profiler::GetClockSyscnt(), 1);
  SetGraphDependency(kernel_graph, segment);
  return CompileGraph(kernel_graph, io_nodes, device_context, run_mode, run_in_pynative);
}

GraphId GraphCompiler::CompileGraph(const KernelGraphPtr &kernel_graph,
                                    const std::pair<AnfNodePtrList, AnfNodePtrList> &io_nodes,
                                    const DeviceContext *device_context, device::RunMode run_mode,
                                    bool run_in_pynative) {
  MS_EXCEPTION_IF_NULL(session_);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(kernel_graph);

  const auto &outputs = io_nodes.second;
  kernel_graph->UpdateGraphAquireGilAttr();

  auto manager = MakeManager({kernel_graph});
  if (manager) {
    manager->AddFuncGraph(kernel_graph);
    kernel_graph->set_manager(manager);
  }

  opt::OptimizationWithoutBackend(kernel_graph);
  // Unify the MindIR, must be before of the kernel_graph optimization.
  auto kernel_executor = device_context->GetKernelExecutor(false);
  if (kernel_executor != nullptr) {
    kernel_executor->AddMindIRPass(kernel_graph);
  }
  kernel_graph->SetInputNodes();
  kernel_graph->SetExecOrderByDefault();
  session_->SetInputNodeUsage(kernel_graph, manager);
  kernel_graph->SetOptimizerFlag();

  GraphId graph_id = CompileGraphImpl(kernel_graph, device_context, run_in_pynative);
  kernel_graph->set_front_outputs(outputs);
  kernel_graph->set_root_graph_id(graph_id);

  ResetNodeId({kernel_graph});
  session_->DumpGraphs({kernel_graph});

  // Cache the backend kernel_graph output nodes to front nodes with output index.
  auto backend_node = kernel_graph->output();
  MS_EXCEPTION_IF_NULL(backend_node);
  kernel_graph->CacheGraphOutputToFrontNodeWithIndex({backend_node}, outputs);

  AnfAlgo::UpdateGraphValidRefPair(kernel_graph);

  MS_LOG(INFO) << "Status record: end compile graph. graph id: " << graph_id;
  return graph_id;
}

GraphCompilerInfo::~GraphCompilerInfo() {
  GraphScheduler::GetInstance().Clear(name_, graphs_, origin_parameters_order_, control_node_parser_);
}

GraphId GraphCompiler::CompileGraphImpl(const KernelGraphPtr &graph, const DeviceContext *device_context,
                                        bool run_in_pynative) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(session_);
  const auto &context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (use_cache_to_compile_graph_) {
    UseCacheToCompileGraphImpl(graph, device_context);
  } else {
#ifdef ENABLE_DUMP_IR
    if (context->CanDump(kIntroductory)) {
      // Dump .pb graph before graph optimization.
      DumpIRProto(graph, "before_opt_" + std::to_string(graph->graph_id()));
    }
#endif
    MS_EXCEPTION_IF_NULL(device_context->GetKernelExecutor(false));
    // Execute optimization pass.
    uint64_t start_time = profiler::GetClockSyscnt();
    PROF_START(OptimizeGraph);
    device_context->GetKernelExecutor(false)->OptimizeGraph(graph);
    PROF_END(OptimizeGraph);
    (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageOptimizeGraph, start_time,
                                    profiler::GetClockSyscnt(), 1);
    // Generate 'KernelMod' for all kernels and set 'KernelMod' into kernel,
    // 'KernelMod' is real executive object of kernel.
    start_time = profiler::GetClockSyscnt();
    PROF_START(CreateKernel);
    graph->SetExecOrderByDefault();
    device_context->GetKernelExecutor(false)->CreateKernel(graph->execution_order());
    PROF_END(CreateKernel);
    (void)profiler::CollectHostInfo(kModelNameRuntime, kEventCompileGraph, kStageCreateKernel, start_time,
                                    profiler::GetClockSyscnt(), 1);

    // Read the output and input ref map and set to the kernel graph.
    AnfAlgo::AddOutInRefToGraph(graph);

    session_->RecurseSetSummaryNodesForAllGraphs(graph.get());
    // Update needed dump kernels for mindRT.
    DumpJsonParser::GetInstance().UpdateNeedDumpKernels(*graph.get());

    // dynamic shape pass of graphmode
    if (graph->is_dynamic_shape()) {
      auto profiler_manage_inst = profiler::ProfilerManager::GetInstance();
      MS_EXCEPTION_IF_NULL(profiler_manage_inst);
      profiler_manage_inst->SetNetDynamicShapeStatus();
    }
  }

  if (export_compile_cache_) {
    session_->CacheKernelGraph({graph});
  }
  // Adjust kernel graph before run graph.
  PROF_START(PreprocessBeforeRun);
  device_context->GetKernelExecutor(false)->PreprocessBeforeRun(graph);
  PROF_END(PreprocessBeforeRun);
  graph->UpdateInternalParameter();
  // Set device target for parameter affinity.
  AnfAlgo::SetParameterDeviceTarget(graph);

  PROF_START(CreateDeviceAddress);
  // Create device address for all anf nodes of graph.
  CreateDeviceAddress(graph, device_context);
  PROF_END(CreateDeviceAddress);

  SetSummaryNodesRefCount(graph.get());
#ifdef ENABLE_DUMP_IR
  // Dump .pb graph after graph optimization.
  if (context->CanDump(kIntroductory)) {
    DumpIRProto(graph, "after_opt_" + std::to_string(graph->graph_id()));
  }
#endif

#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  // Dump graph for GPU mindRT if dump is enabled.
  debugger->DumpInGraphCompiler(graph);
  if (debugger && debugger->DebuggerBackendEnabled()) {
    // Load graphs for GPU and Ascend mindRT.
    debugger->LoadGraphs(graph);
  }
#endif

  graph->EnableRuntimeCache();
  return graph->graph_id();
}

KernelGraphPtr GraphCompiler::Fetch(GraphId graph_id) const {
  MS_EXCEPTION_IF_NULL(session_);
  return session_->GetGraph(graph_id);
}

void GraphCompiler::CreateDeviceAddress(const KernelGraphPtr &graph, const DeviceContext *device_context) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Status record: start create device address. graph id: " << graph->graph_id();
  DeviceAddressUtils::CreateParameterDeviceAddress(device_context, graph);
  DeviceAddressUtils::CreateValueNodeDeviceAddress(device_context, graph);
  DeviceAddressUtils::CreateKernelOutputDeviceAddress(device_context, graph, false);
  DeviceAddressUtils::CreateKernelWorkspaceDeviceAddress(device_context, graph);
  DeviceAddressUtils::UpdateDeviceAddressForInplaceNode(graph);
  DeviceAddressUtils::UpdateDeviceAddressForRefNode(graph);
  MS_LOG(INFO) << "Status record: end create device address. graph id: " << graph->graph_id();
}

void GraphCompiler::RegisterSummaryCallBackFunc(const CallBackFunc &callback) const {
  MS_EXCEPTION_IF_NULL(session_);
  session_->RegisterSummaryCallBackFunc(callback);
}

void GraphCompiler::Summary(const std::vector<KernelGraphPtr> &graphs) const {
  MS_EXCEPTION_IF_NULL(session_);
  for (const auto &graph : graphs) {
    session_->Summary(graph.get());
  }
}

void GraphCompiler::SetGraphDependency(const KernelGraphPtr &graph, const GraphSegmentPtr &segment) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(segment);
  segment->graph_id_ = graph->graph_id();
  for (auto &pre_segment : segment->pre_segments_) {
    MS_EXCEPTION_IF_NULL(pre_segment);
    auto pre_graph = Fetch(pre_segment->graph_id_);
    MS_EXCEPTION_IF_NULL(pre_graph);
    pre_graph->AddPostGraph(graph);
    graph->AddPreGraph(pre_graph);
    MS_LOG(INFO) << "Link graph " << pre_segment->graph_id_ << " to " << graph->graph_id();
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
