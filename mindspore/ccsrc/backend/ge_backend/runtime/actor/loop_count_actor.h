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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_LOOP_COUNT_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_LOOP_COUNT_ACTOR_H_

#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <utility>
#include "utils/hash_map.h"
#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "backend/ge_backend/runtime/actor/debug_aware_actor.h"
#include "backend/ge_backend/runtime/device_tensor_store.h"
#include "backend/ge_backend/runtime/control_node_parser.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
// The loop count actor is used to receive the control of tail kernel actor to represent the end of one step
// and decide whether to loop execution by loop count.
class LoopCountActor : public DebugAwareActor {
 public:
  LoopCountActor(const std::string &name, const std::string &graph_name, size_t loop_count, size_t sink_size,
                 const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid, const AID *profiler_aid,
                 GraphExecutionStrategy strategy, const bool is_need_sync_stream)
      : DebugAwareActor(name, KernelTransformType::kLoopCountActor, recorder_aid, memory_manager_aid, debug_aid,
                        profiler_aid),
        graph_name_(graph_name),
        loop_count_(loop_count),
        sink_size_(sink_size),
        current_count_(0),
        total_running_count_(0),
        strategy_(strategy),
        is_need_sync_stream_(is_need_sync_stream) {}

  ~LoopCountActor() override = default;

  // The callback waits for the memory manager actor to finish all the message processing.
  void OnMemoryAllocFinish(OpContext<KernelTensor> *const context) override;

  // The debug related operation interface.
  void SendDebugReq(OpContext<KernelTensor> *const context) override;
  void SendProfilerReq(OpContext<KernelTensor> *const context);

  // Get the member.
  size_t loop_count() const { return loop_count_; }
  size_t sink_size() const { return sink_size_; }
  const AID &data_prepare_aid() const { return data_prepare_aid_; }
  const std::vector<AID> &entrance_aids() const { return entrance_aids_; }

 protected:
  void Run(OpContext<KernelTensor> *const context) override;
  void SendOutput(OpContext<KernelTensor> *const context) override;

 private:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;

  void IncreaseLoopCount(OpContext<KernelTensor> *const context);

  // Graph name of GraphCompilerInfo. For example, kernel_graph_0-3.
  std::string graph_name_;

  // The loop count is constant, the current count is increased after each step running finished.
  size_t loop_count_;
  size_t sink_size_;
  size_t current_count_;
  // The total running count represents the toal step running count.
  size_t total_running_count_;

  // The actors which need be handled separately by loop count actor.
  AID data_prepare_aid_;
  std::vector<AID> entrance_aids_;

  // The execution strategy for executing actor.
  // In pipeline mode,  sync stream for every step.
  GraphExecutionStrategy strategy_{GraphExecutionStrategy::kPipeline};

  // Only need sync stream in DR scenarios.
  bool is_need_sync_stream_{true};
};

using LoopCountActorPtr = std::shared_ptr<LoopCountActor>;
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_LOOP_COUNT_ACTOR_H_
