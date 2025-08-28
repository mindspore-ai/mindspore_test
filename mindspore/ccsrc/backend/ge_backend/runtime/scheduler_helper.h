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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_SCHEDULER_HELPER_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_SCHEDULER_HELPER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include <set>
#include <algorithm>

#include "backend/ge_backend/runtime/actor/actor_set.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
class SchedulerHelper {
 public:
  // Convert the actors vector by the actor set.
  static std::vector<AbstractActorPtr> CollectActors(const ActorSet *actor_set);

  // Judge the input node whether need the control arrow.
  static bool HasMonadControl(const AnfNodePtr &anf_node, const KernelGraphPtr &graph);

  static void AddDeviceTensorStore(const AnfNodePtr &anf_node, const KernelTensorPtr &kernel_tensor);

  static void AddMonadDeviceTensorStore(AbstractActor *const to_actor, const CNodePtr &kernel,
                                        const KernelGraphPtr &graph);

  // Add the arrow between from actor and to actor.
  static void AddDataArrow(AbstractActor *const from_actor, AbstractActor *const to_actor, size_t from_output_index,
                           size_t to_input_index, const AnfNodePtr &from_kernel = nullptr);
  static void AddResultArrow(AbstractActor *const from_actor, OutputActor *const to_actor,
                             const AnfNodePtr &from_kernel, size_t from_output_index, size_t output_position);
  static void AddControlArrow(AbstractActor *const from_actor, AbstractActor *const to_actor);

  // Add the arrow for control actor.
  static void AddPartialArrow(ControlActor *const from_actor, ControlActor *const to_actor, size_t from_index,
                              size_t to_index);
  static void AddBranchIDArrow(ControlActor *const from_actor, ControlActor *const to_actor);
  // Body control arrow is only exists to entrance actor..
  static void AddLoopBodyControlArrow(AbstractActor *from_actor, EntranceActor *to_actor);
  // Data arrow with branch id is only exists from gather actor to entrance actor.
  static void AddDataWithBranchIDArrow(GatherActor *const gather_actor, const EntranceActor *entrance_actor,
                                       const FuncGraphPtr &func_graph);
  // Since the output of exit actor has branches, it needs to be based on a dedicated interface.
  static void AddDataArrowForExitActor(ExitActor *const exit_actor, AbstractActor *const to_actor, size_t from_index,
                                       size_t to_index, int branch_id);
  static void AddPartialArrowForExitActor(ExitActor *const exit_actor, ControlActor *const to_actor, size_t from_index,
                                          size_t to_index, int branch_id);
  static void AddControlArrowForExitActor(ExitActor *from_actor, AbstractActor *to_actor, int branch_id);

  static void DumpActorSet(const ActorSet *actor_set, std::ofstream &ofs);
  static void DumpFormatActorSet(const ActorSet *actor_set, std::ofstream &ofs);
};
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_GRAPH_SCHEDULER_H_
