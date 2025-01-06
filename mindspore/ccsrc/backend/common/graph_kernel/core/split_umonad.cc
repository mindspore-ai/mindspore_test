/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/core/split_umonad.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "include/backend/kernel_info.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "include/common/utils/anfalgo.h"
#include "ops/op_def.h"

namespace mindspore::graphkernel {
namespace {
bool IsUMonad(const BaseRef &n) {
  MS_EXCEPTION_IF_NULL(n);
  if (utils::isa<AnfNodePtr>(n)) {
    auto node = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(node);
    return HasAbstractUMonad(node);
  }
  return false;
}

bool UsedByInplace(const AnfNodePtr &node, const FuncGraphManagerPtr &mng) {
  auto &users = mng->node_users()[node];
  for (auto &user : users) {
    if (IsPrimitiveCNode(user.first, prim::kPrimDepend)) {
      if (user.second == kRealInputIndexInDepend && UsedByInplace(user.first, mng)) {
        return true;
      }
    } else if (user.first->isa<CNode>() &&
               common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, user.first->cast<CNodePtr>())) {
      return true;
    }
  }
  return false;
}
}  // namespace

const BaseRef SplitUMonad::DefinePattern() const {
  VarPtr v = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  VarPtr U = std::make_shared<CondVar>(IsUMonad);
  return VectorRef({v, Xs, U});
}

const bool SplitUMonad::CanSplit(const AnfNodePtr &node) const {
  if (IsPrimitiveCNode(node, prim::kPrimAssign)) {
    return true;
  }
  if (!node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    return false;
  }
  auto prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(prim);
  const auto &op_name = prim->name();
  auto op_def = ops::GetOpDef(op_name);
  if (op_def == nullptr || !op_def->is_view_) {
    return false;
  }
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  // view op is used by inplace op, which may modify data of view device address
  if (UsedByInplace(node, mng)) {
    MS_LOG(DEBUG) << "node [" << node->fullname_with_scope() << "] is used by inplace op";
    // keep basic op can not be expanded or cluster
    cnode->AddAttr("keep_basic", MakeValue(true));
    return false;
  }
  return true;
}

AnfNodePtr ProcessNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  // Get original op's abstract and inputs
  AbstractBasePtr original_abstract = cnode->abstract()->Clone();
  auto original_inputs = cnode->inputs();

  // Create depend node
  AnfNodePtrList depend_inputs = {NewValueNode(prim::kPrimDepend), original_inputs[input_idx], original_inputs.back()};
  auto depend_cnode = func_graph->NewCNode(depend_inputs);
  depend_cnode->set_abstract(original_inputs[input_idx]->abstract());
  depend_cnode->set_kernel_info(std::make_shared<device::KernelInfo>());
  // Create new node, delete U from inputs.
  AnfNodePtrList new_inputs = {cnode->input(0)};
  for (size_t i = 1; i + 1 < cnode->size(); i++) {
    if (i == input_idx) {
      new_inputs.push_back(depend_cnode);
    } else {
      new_inputs.push_back(cnode->input(i));
    }
  }
  auto new_cnode = func_graph->NewCNode(new_inputs);
  new_cnode->set_abstract(original_abstract);
  new_cnode->set_kernel_info(cnode->kernel_info_ptr());
  return new_cnode;
}

const AnfNodePtr SplitNode::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(node);
  if (!CanSplit(node)) {
    return node;
  }
  return ProcessNode(node->func_graph(), node, 1);
}

AnfNodePtr OpUMonadExpanderDeco::Run(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto new_node = node;
  // assume the UMonad node is the last input
  if (cnode->size() > 1 && HasAbstractUMonad(cnode->inputs().back())) {
    new_node = ProcessNode(node->func_graph(), node, input_idx_);
  }
  return decorated_->Run(new_node);
}
}  // namespace mindspore::graphkernel
