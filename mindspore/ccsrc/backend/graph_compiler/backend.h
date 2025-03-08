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
#ifndef MINDSPORE_CCSRC_VM_BACKEND_H_
#define MINDSPORE_CCSRC_VM_BACKEND_H_

#include <list>
#include <memory>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "include/common/utils/contract.h"
#include "base/base_ref.h"
#include "ir/anf.h"
#include "backend/graph_compiler/backend_base.h"
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
#include "runtime/pynative/op_runner.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace compile {
class BACKEND_EXPORT MindRTBackend : public MindRTBackendBase {
 public:
  MindRTBackend(const std::string &backend_name, const std::string &device_name, uint32_t device_id)
      : MindRTBackendBase(backend_name, device_name, device_id) {}
  ~MindRTBackend() override;

  // Execute all tasks in queue when lazy build is enabled in PyNative mode.
  void WaitTaskFinish() const override;
  // Clear resource when python exit.
  void ClearOpExecutorResource() const;

  // Sync default stream in PyNative mode.
  void SyncStream();

  KernelGraphPtr GetGraphById(GraphId graph_id);

 private:
  void RunGraphByCondition(const ActorInfo &actor_info, const GraphCompilerInfo &graph_compiler_info,
                           const VectorRef &args, VectorRef *outputs) override;
  // Split complete kernel graph to single op graph in PyNative back
  // propagation, then compile and run single op graph or pyboost op(if op registered).
  void RunGraphBySingleOp(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args, VectorRef *outputs);

  runtime::ActorSet *RealCompileGraphBeforeRunActor(const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                                                    bool no_multi_graph);
  void RunGraphByActors(const ActorInfo &actor_info, const GraphCompilerInfo &graph_compiler_info,
                        const VectorRef &args, VectorRef *outputs);

  void RunMsGradGraph(const CNodePtr &kernel, const VectorRef &args, VectorRef *outputs) const;

  // Clean the compilation cache to avoid memory leakage in dynamic shape scenarios.
  void ClearResource();

  void RunActorSet(const ActorInfo &actor_info, runtime::ActorSet *actor_set,
                   const GraphCompilerInfo &graph_compiler_info, const VectorRef &args, bool no_multi_graph,
                   VectorRef *outputs);

  // Cache output tensor ref count of kernels for back propagation graph in PyNative mode.
  std::map<GraphId, std::map<KernelWithIndex, size_t>> cnode_ref_counts_;

  // Cache forward op output value node tensor ref count of kernels for back propagation graph in PyNative mode.
  std::map<std::string, size_t> forward_op_output_tensor_id_;

  OpBackend op_backend_;
  pynative::GraphAdapter graph_adapter_;

  bool first_step_{true};
};

class BACKEND_EXPORT PyBoostAdapter {
 public:
  PyBoostAdapter() = default;
  ~PyBoostAdapter() = default;

  using IsPyBoostRegisteredFunc = std::function<bool(const std::string &device_target, const std::string &op_name)>;
  using RunPyBoostCallFunc = std::function<void(runtime::OpRunnerInfo *, VectorRef *)>;

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
using MindRTBackendPtr = std::shared_ptr<compile::MindRTBackend>;
}  // namespace compile
using BackendOpRunInfoPtr = std::shared_ptr<session::BackendOpRunInfo>;
}  // namespace mindspore
#endif
