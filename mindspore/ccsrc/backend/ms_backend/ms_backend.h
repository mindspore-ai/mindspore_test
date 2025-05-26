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
#ifndef MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MSBACKEND_H_
#define MINDSPORE_CCSRC_BACKEND_MS_BACKEND_MSBACKEND_H_

#include <list>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "include/common/utils/contract.h"
#include "ir/anf.h"
#include "base/base_ref.h"
#include "backend/graph_compiler/segment_runner.h"
#include "backend/graph_compiler/graph_partition.h"
#include "backend/graph_compiler/vm.h"
#include "backend/graph_compiler/op_backend.h"
#include "backend/common/session/session_basic.h"
#include "runtime/hardware/device_context.h"
#include "runtime/graph_scheduler/graph_scheduler.h"
#include "runtime/pynative/task/device_task.h"
#include "runtime/pynative/op_compiler.h"
#include "runtime/pynative/graph_adapter.h"
#include "include/backend/visible.h"
#include "backend/ms_backend/ms_backend_base.h"
#include "runtime/pynative/op_runner.h"
namespace mindspore {
namespace backend {
namespace ms_backend {
class BACKEND_EXPORT PyBoostAdapter {
 public:
  PyBoostAdapter() = default;
  ~PyBoostAdapter() = default;

  static bool IsPyBoostRegistered(const std::string &device_target, const std::string &op_name) {
    MS_EXCEPTION_IF_NULL(is_pyboost_registered_func_);
    return is_pyboost_registered_func_(device_target, op_name);
  }
  static void RunPyBoostCall(runtime::OpRunnerInfo *op_runner_info, VectorRef *op_outputs) {
    MS_EXCEPTION_IF_NULL(run_pyboost_call_func_);
    run_pyboost_call_func_(op_runner_info, op_outputs);
  }

  static void SetIsPyBoostRegistered(const IsPyBoostRegisteredFunc &func) { is_pyboost_registered_func_ = func; }
  static void SetRunPyBoostCallFunc(const RunPyBoostCallFunc &func) { run_pyboost_call_func_ = func; }

 private:
  inline static IsPyBoostRegisteredFunc is_pyboost_registered_func_;
  inline static RunPyBoostCallFunc run_pyboost_call_func_;
};

class BACKEND_EXPORT MSBackend : public MSBackendBase {
 public:
  MSBackend() : MSBackendBase() {}
  ~MSBackend() override;

  void SetPyBoostRegistered(const IsPyBoostRegisteredFunc &func, const RunPyBoostCallFunc &call_func) override {
    PyBoostAdapter::SetIsPyBoostRegistered(func);
    PyBoostAdapter::SetRunPyBoostCallFunc(call_func);
  }

  // Execute all tasks in queue when lazy build is enabled in PyNative mode.
  void WaitTaskFinish() const override;
  // Clear resource when python exit.
  void ClearOpExecutorResource() const;

  // Sync default stream in PyNative mode.
  void SyncStream();

  KernelGraphPtr GetGraphById(GraphId graph_id);

 private:
  void RunGraphByCondition(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                           VectorRef *outputs) override;
  // Split complete kernel graph to single op graph in PyNative back
  // propagation, then compile and run single op graph or pyboost op(if op registered).
  void RunGraphBySingleOp(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args, VectorRef *outputs);

  runtime::ActorSet *RealCompileGraphBeforeRunActor(BackendGraphId graph_id,
                                                    const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                                                    bool no_multi_graph);
  void RunGraphByActors(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                        VectorRef *outputs);

  void RunMsGradGraph(const CNodePtr &kernel, const VectorRef &args, VectorRef *outputs) const;

  void RunActorSet(BackendGraphId graph_id, runtime::ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                   const VectorRef &args, bool no_multi_graph, VectorRef *outputs);

  // Cache output tensor ref count of kernels for back propagation graph in PyNative mode.
  std::map<GraphId, std::map<KernelWithIndex, size_t>> cnode_ref_counts_;

  mindspore::compile::OpBackend op_backend_;
  pynative::GraphAdapter graph_adapter_;
};

using MsBackendPtr = std::shared_ptr<MSBackend>;
}  // namespace ms_backend
}  // namespace backend
using BackendOpRunInfoPtr = std::shared_ptr<session::BackendOpRunInfo>;
}  // namespace mindspore
#endif
