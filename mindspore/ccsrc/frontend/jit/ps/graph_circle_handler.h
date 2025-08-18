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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_GRAPH_CIRCLE_HANDLER_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_GRAPH_CIRCLE_HANDLER_H_

#include <string>
#include "ir/func_graph.h"
#include "ir/manager.h"

namespace mindspore {
namespace circle_handler {
constexpr auto kCircleDetect = "circle_detect";
AnfNodePtrList FindGraphCircle(const FuncGraphPtr &fg);
void SetAttrToDepend(const FuncGraphPtr &fg);
bool RevertDependNode(const FuncGraphPtr &fg, const FuncGraphManagerPtr &mng);
void DetectAndRevertGraphCircle(const FuncGraphPtr &fg, const FuncGraphManagerPtr &mng, const std::string &pass_name,
                                const std::string &switch_name = "");
}  // namespace circle_handler
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_GRAPH_CIRCLE_HANDLER_H_
