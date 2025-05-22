/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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
#include "backend/ms_backend/ms_backend.h"

#include <algorithm>
#include <vector>
#include <map>
#include <stack>
#include <unordered_map>
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/nn_op_name.h"
#include "mindspore/ops/op_def/structure_op_name.h"
#include "include/common/utils/parallel_context.h"
#include "backend/graph_compiler/transform.h"
#include "backend/common/session/session_factory.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/op_compiler.h"
#include "include/backend/optimizer/helper.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "backend/common/session/jit_call_graph.h"
#include "ir/anf.h"
#include "frontend/ir/base_ref_py.h"
#include "pybind_api/pybind_patch.h"
#include "include/common/utils/callbacks.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/graph_scheduler/graph_compiler.h"
#include "runtime/pynative/op_runner.h"
#include "runtime/pynative/graph_adapter.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "include/backend/distributed/recovery/recovery_context.h"
#include "pybind_api/gil_scoped_long_running.h"
#ifdef ENABLE_DEBUGGER
#include "include/backend/debug/debugger/debugger.h"
#endif
#include "include/backend/debug/data_dump/dump_json_parser.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/ps/ps_context.h"
#endif

#include "runtime/device/device_address_utils.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "runtime/pipeline/pipeline.h"
#include "runtime/pipeline/task/run_graph_task.h"
#include "include/common/utils/stub_tensor.h"

namespace mindspore {
namespace backend {
namespace ms_backend {
MSBackend::~MSBackend() {
  GilReleaseWithCheck gil_release;
  runtime::Pipeline::Get().frontend_stage()->Wait();
}

runtime::ActorSet *MSBackend::RealCompileGraphBeforeRunActor(BackendGraphId graph_id,
                                                             const GraphCompilerInfo &graph_compiler_info,
                                                             const VectorRef &args, bool no_multi_graph) {
  WaitTaskFinish();
  WaitMultiStream(graph_compiler_info);
  CreateTensorArgs(args, graph_compiler_info);
  WaitTaskFinish();
  auto graphs = graph_compiler_info.graphs_;
  auto device_contexts = graph_compiler_info.device_contexts_;

  for (size_t i = 0; i < graphs.size(); ++i) {
    const auto &graph = graphs[i];
    MS_EXCEPTION_IF_NULL(graph);
    graph->set_flag(kFlagPyNativeRunInGraph, true);
    graph->set_flag(kFlagIsPynativeBpropGraph,
                    graph_compiler_info.root_func_graph_->has_flag(kFlagIsPynativeBpropGraph));
    if (graph->is_any_type_input()) {
      continue;
    }
    auto input_tensors = GetRunGraphInputs(graph_compiler_info, args);
    if (graph_compiler_info.enable_graph_pipeline_) {
      for (const auto &tensors : input_tensors) {
        for (const auto &tensor : tensors) {
          if (tensor) {
            tensor->set_need_pipeline_sync(true);
          }
        }
      }
    }

    if (no_multi_graph) {
      MS_LOG(INFO) << "Replace parameter format";
      // The input tensors of heterogeneous graphs or control flow graphs are null.
      // Need to get tensor after ParseControlNodes.
      pynative::GraphAdapter::ReplaceGraphParameterProperties(graph, input_tensors.at(i), device_contexts[i]);
    }
    (void)graph_compiler_->CompileGraphImpl(graph, device_contexts[i]);
    pynative::GraphAdapter::RemoveUnusedValueNodes(graph);
    // PyNative use kernel graph will result in front node and back node is the same; But in pynative task sink, backend
    // still create new kernel graph
    if (graph_compiler_info.root_func_graph_->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
      graph->CacheGraphOutputToFrontNodeWithIndex({graph->output()}, {graph->output()});
    } else {
      graph->CacheGraphOutputToFrontNodeWithIndex({graph->output()}, graph->front_outputs());
    }
    // Clear front outputs after the outputs is cached.
    graph->set_front_outputs({});
    AnfAlgo::UpdateGraphValidRefPair(graph);
    pynative::GraphAdapter::SensTensorToDevice(graph, device_contexts[i]);
  }

  ParseControlNodes(graph_compiler_info);
  UpdateGraphCompilerInfo(graph_compiler_info);
  auto actor_set = runtime::GraphScheduler::GetInstance().Transform(graph_compiler_info);
  MS_EXCEPTION_IF_NULL(actor_set);
  constexpr auto kKernelActorThreshold = 5000;
  // Turning off multithreading may cause stack overflow in control flow scenarios.
  if (no_multi_graph && actor_set->kernel_actors_.size() < kKernelActorThreshold &&
      graph_compiler_info.root_func_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    // Multithreading can cause spikes in memory usage and performance fluctuations.
    actor_set->is_multi_thread_execution_ = false;
    MS_LOG(INFO) << "Actor Multithreading is turned off!";
  }
  runtime::GraphScheduler::GetInstance().Schedule(actor_set);

  for (size_t i = 0; i < graphs.size(); ++i) {
    pynative::GraphAdapter::ClearForwardOutputValueNodeDeviceAddress(graphs[i], device_contexts[i]);
    pynative::GraphAdapter::GenerateRefCountForBpropValueNode(graphs[i]);
    graph_adapter_.GenerateBackoffValueNodeOwners(graphs[i]);
  }
  return actor_set;
}

void MSBackend::RunGraphByActors(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info,
                                 const VectorRef &args, VectorRef *outputs) {
  MS_LOG(INFO) << "Status record: begin run actor: " << graph_id;
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  auto graphs = graph_compiler_info.graphs_;
  auto &device_contexts = graph_compiler_info.device_contexts_;
  if (device_contexts.size() != graphs.size()) {
    MS_LOG(EXCEPTION) << "Graphs size " << graphs.size() << " is not equal to device_contexts size "
                      << device_contexts.size();
  }

  // KernelByKernel: The size of control_nodes is at least 1 since there is return node in the graph.
  // GraphMode: No control nodes.
  bool no_multi_graph = graph_compiler_info.control_nodes_.size() <= 1 && graphs.size() == 1;
  auto actor_set = runtime::GraphScheduler::GetInstance().Fetch(graph_id);
  if (actor_set == nullptr) {
    actor_set = RealCompileGraphBeforeRunActor(graph_id, graph_compiler_info, args, no_multi_graph);
    // Clear the temp members at the end of graph building.
    ClearGraphBuildMember();
  }
  MS_EXCEPTION_IF_NULL(actor_set);

  if (graph_compiler_info.enable_graph_pipeline_) {
    // 1. Construct stub output.
    auto output_node = graph_compiler_info.root_func_graph_->output();
    MS_EXCEPTION_IF_NULL(output_node);
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kOutputProcess,
                                       "MakeStubNode");
    auto stub_output_pair = stub::MakeStubNode(output_node->abstract());
    if (stub_output_pair.second) {
      MS_LOG(DEBUG) << "Enable pynative graph pipeline for actor set: " << graph_id;
      // 2. Async run graph.
      auto &stub_output = stub_output_pair.first;
      MS_EXCEPTION_IF_NULL(stub_output);
      outputs->push_back(stub_output);

      auto run_graph_task = std::make_shared<runtime::RunGraphTask>(
        [=, &graph_compiler_info]() {
          actor_set->output_actor_->SetStubOutput(stub_output);
          RunActorSet(graph_id, actor_set, graph_compiler_info, args, no_multi_graph, outputs);
        },
        stub_output);
      GilReleaseWithCheck release_gil;
      runtime::Pipeline::Get().frontend_stage()->Push(run_graph_task);
      return;
    }
    graph_compiler_info.enable_graph_pipeline_ = false;
    MS_LOG(INFO)
      << "Failed to create Stub output, encountered an unsupported output type for graph: " << graph_id
      << ". Currently, only output types that include: Tensor, Scalar, String, fixed-length Sequence, are "
         "supported. The single op and graph pipeline has been disabled, so the performance will not be improved.";
  }

  RunActorSet(graph_id, actor_set, graph_compiler_info, args, no_multi_graph, outputs);
}

void MSBackend::RunActorSet(BackendGraphId graph_id, runtime::ActorSet *actor_set,
                            const GraphCompilerInfo &graph_compiler_info, const VectorRef &args, bool no_multi_graph,
                            VectorRef *outputs) {
  WaitTaskFinish();
  WaitMultiStream(graph_compiler_info);
  WaitTaskFinish();

  auto graphs = graph_compiler_info.graphs_;
  auto &device_contexts = graph_compiler_info.device_contexts_;
  if (graph_compiler_info.root_func_graph_->has_flag(kFlagIsPynativeBpropGraph)) {
    for (size_t i = 0; i < graphs.size(); ++i) {
      graph_adapter_.UpdateForwardOutputInBpropGraph(graphs[i], device_contexts[i], no_multi_graph);
      pynative::GraphAdapter::UpdateDynamicValueNodeAbstract(graphs[i]);
    }
  }

  std::vector<std::vector<tensor::TensorPtr>> input_tensors;
  // make sure enable input optimize condition right.
  MS_LOG(INFO) << "Start to run graph, args size: " << args.size() << ", graph: " << actor_set->name_;
  runtime::ActorDispatcher::set_enable_sub_graph_execute_for_cur_actor_set(actor_set->enable_kbk_sub_graph_execute_);
  runtime::ActorDispatcher::set_enable_input_optimize_for_cur_actor_set(actor_set->enable_input_optimize_);
  if (!runtime::EnableInputOptimize() && !runtime::EnableParallelDispatchKernel()) {
    input_tensors = GetRunGraphInputs(graph_compiler_info, args);
    if (graphs.size() > input_tensors.size()) {
      MS_LOG(EXCEPTION) << "The actor_set " << actor_set->name_ << " graphs size " << graphs.size()
                        << " should less than or equal to inputs size " << input_tensors.size();
    }
    pynative::GraphAdapter::HandleHeterogeneousTensors(input_tensors, device_contexts, actor_set);
    // Release GIL and run actor DAG.
    GilReleaseWithCheck release_gil;
    VectorRef empty_args;
    runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors, empty_args);
  } else {
    GilReleaseWithCheck release_gil;
    runtime::GraphScheduler::GetInstance().Run(actor_set, input_tensors, args);
  }

  MS_EXCEPTION_IF_NULL(graph_compiler_);
  graph_compiler_->Summary(graph_compiler_info.graphs_);

  auto output = graph_compiler_info.root_func_graph_->output();
  if (output != nullptr) {
    MS_LOG(DEBUG) << "Current out " << output->DebugString();
  }
  if (graph_compiler_info.root_func_graph_->has_flag(kFlagIsPyNativeBpropKernelGraph)) {
    MS_EXCEPTION_IF_NULL(graph_compiler_info.origin_output_node_);
    MS_LOG(DEBUG) << "Origin out:" << graph_compiler_info.origin_output_node_->DebugString();
    graph_compiler_info.root_func_graph_->set_output(graph_compiler_info.origin_output_node_);
  }
  ConstructOutputs(actor_set, outputs, graph_compiler_info.root_func_graph_,
                   graph_compiler_info.enable_graph_pipeline_);
  actor_set->output_actor_->FreeSummaryNodeMem();
  runtime::GraphScheduler::GetInstance().ClearActorData(actor_set);
  // Close abstract_lock for dynamic_shape
  AnfUtils::CloseAbstractLock();
  MS_LOG(INFO) << "Status record: end run actor: " << graph_id;
}

void MSBackend::RunGraphByCondition(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info,
                                    const VectorRef &args, VectorRef *outputs) {
  RunGraphByActors(graph_id, graph_compiler_info, args, outputs);
}

void MSBackend::WaitTaskFinish() const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kWaitTaskFinish,
                                     runtime::kDefaultOpName);
  runtime::Pipeline::Get().WaitAll();
}

void MSBackend::SyncStream() {
  const auto &device_context =
    device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);

  auto ret = device_context->device_res_manager_->SyncAllStreams();
  if (!ret) {
    MS_LOG(EXCEPTION) << "Sync Stream failed";
  }
}

KernelGraphPtr MSBackend::GetGraphById(GraphId graph_id) {
  MS_EXCEPTION_IF_NULL(graph_compiler_);
  return graph_compiler_->Fetch(graph_id);
}

MS_REGISTER_BACKEND(kMSBackendName, MSBackend)
}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore
