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
#include "frontend/optimizer/irpass/minmax_grad.h"

#include <vector>
#include <memory>

#include "frontend/optimizer/optimizer.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "frontend/optimizer/anf_visitor.h"
#include "frontend/operator/ops.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
AnfNodePtr MinMaximumGrad::operator()(const OptimizerPtr &optimizer, const AnfNodePtr &node) {
  Reset();
  AnfVisitor::Match(prim::kPrimTupleGetItem, {MinMaximumGrad::IsOriginMaxMinGrad, IsValueNode<Int64Imm>})(node);
  if (grad_ == nullptr || idx_ < 0 || idx_ > 1 || node->func_graph() == nullptr) {
    return nullptr;
  }

  // check single use
  auto mng = optimizer->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto &users = mng->node_users();
  if (users.find(grad_) == users.end() || users[grad_].size() != 1) {
    return nullptr;
  }

  // {target_grad, Xs}
  auto &inputs = grad_->inputs();
  auto prim = GetValueNode<PrimitivePtr>(inputs[0]);

  auto new_prim = std::make_shared<Primitive>(prim->name());
  new_prim->set_attr("grad_x", MakeValue(true));
  new_prim->set_attr("grad_y", MakeValue(true));

  if (idx_ == 0) {
    new_prim->set_attr("grad_y", MakeValue(false));
  }
  if (idx_ == 1) {
    new_prim->set_attr("grad_x", MakeValue(false));
  }

  std::vector<AnfNodePtr> args;
  args.push_back(NewValueNode(new_prim));
  (void)args.insert(args.cend(), inputs.cbegin() + 1, inputs.cend());

  auto fg = node->func_graph();
  auto new_code = fg->NewCNode(args);
  if (AnfUtils::GetDumpFlag(grad_)) {
    AnfUtils::SetDumpFlag(new_code);
  }

  return fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), new_code, NewValueNode(MakeValue(idx_))});
}

void MinMaximumGrad::Visit(const CNodePtr &cnode) { grad_ = cnode; }

void MinMaximumGrad::Visit(const ValueNodePtr &vnode) { idx_ = GetValue<int64_t>(vnode->value()); }

void MinMaximumGrad::Reset() {
  idx_ = -1;
  grad_ = nullptr;
}

// Check if node is MinimumGrad() or MaximumGrad()
bool MinMaximumGrad::IsOriginMaxMinGrad(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimMaximumGrad) && !IsPrimitiveCNode(node, prim::kPrimMinimumGrad)) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto x_v = prim->GetAttr("grad_x");
  auto y_v = prim->GetAttr("grad_y");
  if (x_v == nullptr || y_v == nullptr || !x_v->isa<BoolImm>() || !y_v->isa<BoolImm>()) {
    return false;
  }

  bool x = GetValue<bool>(x_v);
  bool y = GetValue<bool>(y_v);
  return x && y;
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
