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
class BACKEND_EXPORT MSBackend : public MSBackendBase {
 public:
  MSBackend() : MSBackendBase() {}
  ~MSBackend() override;

  // Execute all tasks in queue when lazy build is enabled in PyNative mode.
  void WaitTaskFinish() const override;

  // Sync default stream in PyNative mode.
  void SyncStream();

  KernelGraphPtr GetGraphById(GraphId graph_id);

 private:
  void RunGraphByCondition(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                           VectorRef *outputs) override;

  runtime::ActorSet *RealCompileGraphBeforeRunActor(BackendGraphId graph_id,
                                                    const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                                                    bool no_multi_graph);
  void RunGraphByActors(BackendGraphId graph_id, const GraphCompilerInfo &graph_compiler_info, const VectorRef &args,
                        VectorRef *outputs);

  void RunActorSet(BackendGraphId graph_id, runtime::ActorSet *actor_set, const GraphCompilerInfo &graph_compiler_info,
                   const VectorRef &args, bool no_multi_graph, VectorRef *outputs);

  mindspore::compile::OpBackend op_backend_;
  pynative::GraphAdapter graph_adapter_;
};
}  // namespace ms_backend
}  // namespace backend
using BackendOpRunInfoPtr = std::shared_ptr<session::BackendOpRunInfo>;
}  // namespace mindspore
#endif
