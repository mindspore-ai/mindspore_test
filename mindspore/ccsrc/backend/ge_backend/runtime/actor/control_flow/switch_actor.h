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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_CONTROLFLOW_SWITCH_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_CONTROLFLOW_SWITCH_ACTOR_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "backend/ge_backend/runtime/actor/control_flow/control_actor.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
using mindspore::session::KernelWithIndex;

// Switch actor is used to execute the branch according to the input condition.
// Switch and SwitchLayer node will be converted to switch actor.
class SwitchActor : public ControlActor {
 public:
  SwitchActor(const std::string &name, const AID &memory_manager_aid, const std::vector<KernelWithIndex> &parameters,
              const AnfNodePtr &node);
  ~SwitchActor() override = default;

 protected:
  void FetchInput(OpContext<KernelTensor> *const context) override;

 private:
  friend class ControlNodeScheduler;
  // Get the output branch index of the switch actor.
  size_t GetIndex(const OpContext<KernelTensor> *const context) const;
  size_t index_{0};
};

using SwitchActorPtr = std::shared_ptr<SwitchActor>;
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_CONTROLFLOW_SWITCH_ACTOR_H_
