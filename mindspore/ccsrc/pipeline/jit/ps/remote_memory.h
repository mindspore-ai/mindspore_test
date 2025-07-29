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

#ifndef MINDSPORE_CCSRC_PIPELINE_REMOTE_MEMORY_H_
#define MINDSPORE_CCSRC_PIPELINE_REMOTE_MEMORY_H_

#include "ir/anf.h"
#include "ir/manager.h"
#include "ir/func_graph.h"
#include "ops/op_def.h"
#include "frontend/ir/primitive_py.h"

namespace mindspore {
namespace remote_memory {
constexpr auto kRemoteActivationAttr = "remote_activation";

template <typename T>
bool NeedActivationToRemote(const T &primal) {
  static const bool enable_remote_memory = (common::GetEnv("MS_DEV_ENABLE_REMOTE_MEMORY") == "1");
  if (!enable_remote_memory) {
    return false;
  }
  if constexpr (std::is_same<T, PrimitivePtr>::value) {
    PrimitivePtr primitive = primal;
    return primitive->HasAttr(kRemoteActivationAttr);
  }
  return false;
}

CNodePtr ActivationToRemote(const FuncGraphPtr &fprop, const AnfNodePtr &activaction);
FuncGraphPtr GenerateMultitypeFGWithRemoteOps(const FuncGraphPtr &func_graph, const TypePtrList &prefetch_type);
void InsertPrefetchForLoad(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph);
void AddRemoteOpsToGraphs(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph);
bool InsertActivactionRemoteOpsForGraph(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph);
}  // namespace remote_memory
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_REMOTE_MEMORY_H_
