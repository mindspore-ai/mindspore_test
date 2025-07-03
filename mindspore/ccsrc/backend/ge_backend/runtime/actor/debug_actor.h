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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_DEBUG_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_DEBUG_ACTOR_H_

#include <string>
#include <vector>
#include "backend/ge_backend/runtime/actor/actor_common.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
// The debug actor is used to debug and dump kernel info, it gets the kernel real time execution info in the device, so
// it is synchronous and blocked.
class DebugActor : public ActorBase {
 public:
  DebugActor() : ActorBase("GEDebugActor") {}
  ~DebugActor() override = default;

  // The debug of each node.
  void DebugPreLaunch(const AnfNodePtr &node, const std::vector<KernelTensorPtr> &op_input_kernel_tensors,
                      const std::vector<KernelTensorPtr> &op_output_kernel_tensors,
                      OpContext<KernelTensor> *const op_context, const AID *from_aid);
  void DebugPostLaunch(const AnfNodePtr &node, const std::vector<KernelTensorPtr> &op_input_kernel_tensors,
                       const std::vector<KernelTensorPtr> &op_output_kernel_tensors,
                       OpContext<KernelTensor> *const op_context, const AID *from_aid);

  // The debug on step begin.
  void DebugOnStepBegin(const std::vector<KernelGraphPtr> &graphs,
                        const std::vector<AnfNodePtr> &origin_parameters_order,
                        OpContext<KernelTensor> *const op_context, const AID *from_aid);

  // The debug on step end.
  void DebugOnStepEnd(OpContext<KernelTensor> *const op_context, const AID *from_aid, int total_running_count,
                      int sink_size);

 private:
  void Finalize() override;
  int step_count_ = 0;
};
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_DEBUG_ACTOR_H_
