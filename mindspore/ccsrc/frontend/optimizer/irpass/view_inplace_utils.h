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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VIEW_INPLACE_OP_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VIEW_INPLACE_OP_UTILS_H_

#include <utility>
#include <vector>
#include <string>
#include <unordered_map>

#include "ir/anf.h"
#include "mindspore/core/include/ir/manager.h"

namespace mindspore {
namespace opt {
namespace irpass {
constexpr auto kOriginalViewOp = "view_op";
constexpr auto kIsVirtualViewOp = "is_virtual_view_op";

enum ViewInplacePassType {
  CommonInline = 0,
  VirtualOpsInsert,
  DoInplaceAndVirtualOpsRemove,
  OnlyDoInplace,
  EliminateVirtualView
};

bool IsViewOutput(const AnfNodePtr &node);
std::pair<CNodePtr, bool> IsCreatedByViewOp(const AnfNodePtr &node);
bool IsInplaceNode(const AnfNodePtr &node);
bool IsViewNode(const AnfNodePtr &node);
bool IsVirtualViewCNode(const AnfNodePtr &node);
AnfNodePtr CheckUMonad(const AnfNodePtr &node);
std::string GetRefKey(const AnfNodePtr &node);
void ReplaceInplaceNodeForCNode(const CNodePtr &cnode, const std::unordered_map<AnfNodePtr, AnfNodePtr> &inplace_input,
                                const FuncGraphManagerPtr &manager, const FuncGraphPtr &func_graph,
                                bool need_ignore_fv = false);
std::vector<bool> GetInplaceChangedParamIndex(const FuncGraphPtr &fg);
int IsFuncOutputSameWithParamNode(const FuncGraphPtr &fg);
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_VIEW_INPLACE_OP_UTILS_H_
