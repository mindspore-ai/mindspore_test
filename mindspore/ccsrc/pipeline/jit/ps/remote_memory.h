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

namespace mindspore {
namespace remote_memory {
void AddDetachToGraph(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph);

template <typename T>
bool NeedActivationToRemote(const T &primal) {
  static const auto enable_remote = (common::GetCompileConfig("ENABLE_REMOTE") == "1");
  if (!enable_remote) {
    return false;
  }
  if constexpr (std::is_same<T, PrimitivePtr>::value) {
    // todo: do we only convert PrimitiveFunction?
    return ops::IsPrimitiveFunction(primal->name());
  }
  return false;
}

CNodePtr ActivationToRemote(const FuncGraphManagerPtr &mng, const FuncGraphPtr &fprop, const FuncGraphPtr &bprop,
                            const AnfNodePtr &out, const AnfNodePtr &dout, const AnfNodePtr &out_param);
void InsertPrefetchForLoad(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph);
}  // namespace remote_memory
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_REMOTE_MEMORY_H_
