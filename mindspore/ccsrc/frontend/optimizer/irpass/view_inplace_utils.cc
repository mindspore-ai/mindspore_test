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
#include "frontend/optimizer/irpass/view_inplace_utils.h"

#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {
bool IsViewOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abs = node->abstract();
  if (abs != nullptr && abs->isa<abstract::AbstractRefTensor>()) {
    const auto ref = abs->cast<abstract::AbstractRefPtr>();
    if (ref->is_view_output()) {
      return true;
    }
  }
  return false;
}

bool IsViewNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
  return prim != nullptr && prim->graph_view_prim();
}

bool IsInplaceNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
  return prim != nullptr && prim->inplace_prim();
}

std::pair<CNodePtr, bool> IsCreatedByViewOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsViewNode(node)) {
    auto cnode = node->cast<CNodePtr>();
    return {cnode, true};
  }
  const auto &abs = node->abstract();
  if (abs != nullptr && abs->isa<abstract::AbstractRefTensor>()) {
    auto ref = abs->cast<abstract::AbstractRefPtr>();
    if (ref->is_view_output()) {
      auto view_op = abs->user_data<CNode>(kOriginalViewOp);
      if (view_op != nullptr) {
        return {view_op, true};
      }
    }
  }
  return {nullptr, IsViewOutput(node)};
}

bool IsVirtualViewCNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  return cnode != nullptr && cnode->HasAttr(kIsVirtualViewOp);
}

AnfNodePtr CheckUMonad(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!HasAbstractUMonad(node)) {
    MS_LOG(EXCEPTION) << "Need to be umonad, but got: " << node->DebugString();
  }
  return node;
}

std::string GetRefKey(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto abs = node->abstract();
  if (abs == nullptr || !abs->isa<abstract::AbstractRefTensor>()) {
    return "";
  }
  auto abs_ref = abs->cast<abstract::AbstractRefPtr>();
  auto ref_key_value = abs_ref->ref_key_value()->cast<StringImmPtr>();
  return ref_key_value == nullptr ? "" : ref_key_value->value();
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
