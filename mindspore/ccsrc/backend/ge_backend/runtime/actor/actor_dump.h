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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_ACTOR_DUMP_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_ACTOR_DUMP_H_

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <tuple>

#include "backend/ge_backend/runtime/actor/abstract_actor.h"
#include "backend/ge_backend/runtime/actor/actor_set.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
void DumpDataPrepareActor(const DataPrepareActorPtr &actor, std::ofstream &ofs);
void DumpLoopCountActor(const LoopCountActorPtr &actor, std::ofstream &ofs);
void DumpOutputActor(const OutputActorPtr &actor, std::ofstream &ofs);
void DumpDSActors(const std::vector<DataSourceActorPtr> &actors, std::ofstream &ofs);
void DumpSuperKernelActors(const std::vector<SuperKernelActorPtr> &actors, std::ofstream &ofs);
void DumpNoInputKernelActors(const std::vector<AbstractActorPtr> &actors, std::ofstream &ofs);
void DumpControlActors(const ControlActorSetPtr &control_actor_set, std::ofstream &ofs);
using DeviceAddressPtr = device::DeviceAddressPtr;
using KernelTensorPtr = kernel::KernelTensorPtr;
using ActorInfoMap = mindspore::HashMap<AbstractActor *, std::tuple<size_t, std::vector<KernelTensorPtr>>>;
using GetInputAidFunc = std::function<std::vector<std::string>(AbstractActor *const)>;
std::vector<AbstractActor *> TopoSortForActor(AbstractActor *root, const GetInputAidFunc &get_input_func = nullptr);
void DumpActorInfo(AbstractActor *actor, size_t index, ActorInfoMap *actor_info, std::ofstream &ofs);
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_ACTOR_DUMP_H_
