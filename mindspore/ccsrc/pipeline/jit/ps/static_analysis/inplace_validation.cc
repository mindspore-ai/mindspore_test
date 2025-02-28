/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/static_analysis/inplace_validation.h"
#include <map>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/core/include/ir/core_ops_primitive.h"
#include "frontend/optimizer/utils.h"
#include "utils/trace_base.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace pipeline {
namespace {
static std::map<FuncGraphPtr, bool> check_flag;
bool IsInplaceOps(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto prim_node = node->cast<CNodePtr>()->input(0);
  auto prim = GetValueNode<PrimitivePtr>(prim_node);
  if (prim != nullptr) {
    return prim->inplace_prim();
  }
  return false;
}

bool CheckRequiresGradRefkeysInInplaceOps(const std::string &ref_key_str,
                                          const std::vector<std::string> &requires_grad_refkeys) {
  auto iter = std::find(requires_grad_refkeys.begin(), requires_grad_refkeys.end(), ref_key_str);
  return iter != requires_grad_refkeys.end();
}

void CheckInplaceOpValidate(const FuncGraphPtr &func_graph, const std::vector<std::string> &requires_grad_refkeys) {
  MS_EXCEPTION_IF_NULL(func_graph);
  const auto &nodes = TopoSort(func_graph->get_return());
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto node = nodes[i];
    MS_EXCEPTION_IF_NULL(node);
    if (!IsInplaceOps(node)) {
      continue;
    }
    auto abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    if (!abs->isa<abstract::AbstractRefTensor>()) {
      continue;
    }
    TraceGuard guard(MakeTraceInfo<TraceInplace>(node->debug_info()));
    auto abs_ref = abs->cast<abstract::AbstractRefPtr>();
    MS_EXCEPTION_IF_NULL(abs_ref->ref_key_value());
    auto ref_key = abs_ref->ref_key_value()->cast<StringImmPtr>();
    MS_EXCEPTION_IF_NULL(ref_key);
    const std::string &ref_key_str = ref_key->value();
    if (CheckRequiresGradRefkeysInInplaceOps(ref_key_str, requires_grad_refkeys)) {
      MS_LOG(EXCEPTION) << "A leaf Variable that requires grad is being used in an in-place operation.";
    }
  }
}

// %0 = J(@forward_fun)
// %1 = %0(%args0, %args1, ...)
// %2 = TupleGetItem(%1, I64(1))
// %3 = %2(xxx)
// %4 = TupleGetItem(%3, I64(1))  input 0
// %5 = TupleGetItem(%3, I64(2))  input 1
// %6 = TupleGetItem(%3, I64(0))  parameters
// %7 = RefToEmbed(%para3_net.param1)
// ...
// %10 = Load(%param1, u)
// %11 = PrimFunc_ZerosLike(%10)
// %12 = EnvironGet(%6, %7, %11)
std::vector<std::string> GetRequiresGradRefKeys(const AnfNodePtr &j_node, const FuncGraphPtr &forward_func) {
  auto func = j_node->func_graph();
  MS_EXCEPTION_IF_NULL(func);
  auto mgr = func->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  auto j_node_caller = opt::GetBpropCaller(mgr, j_node);                      // %1
  auto j_getter = opt::GetBpropGetter(mgr, j_node_caller->cast<CNodePtr>());  // %2
  auto j_getter_caller = opt::GetBpropCaller(mgr, j_getter);                  // %3
  std::vector<std::string> ref_keys;
  auto arg_inputs = forward_func->parameters();
  constexpr size_t ref_to_embed_index = 2;
  constexpr size_t index_in_tuple_getitem = 2;
  const AnfNodeIndexSet &user_nodes = mgr->node_users()[j_getter_caller];
  for (const auto &iter : user_nodes) {
    if (!IsPrimitiveCNode(iter.first, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto idx = GetValueNode<Int64ImmPtr>(iter.first->cast<CNodePtr>()->input(index_in_tuple_getitem));
    if (idx == nullptr) {
      continue;
    }
    // Process parameters
    if (idx->value() != 0) {
      // %5 or %6
      if (arg_inputs.size() < LongToSize(idx->value())) {
        MS_LOG(EXCEPTION) << "The number of args must be greater than index, but got he number of args: "
                          << arg_inputs.size() << ", index: " << idx->value();
      }
      auto arg_input = arg_inputs[idx->value() - 1];
      MS_EXCEPTION_IF_NULL(arg_input);
      auto arg_input_abs = arg_input->abstract();
      if (arg_input_abs && arg_input_abs->isa<abstract::AbstractRefTensor>()) {
        auto arg_input_ref_key = abstract::GetRefKeyFromAbstract(arg_input_abs);
        MS_LOG(DEBUG) << "The ref_key of arg is: " << arg_input_ref_key;
        ref_keys.push_back(arg_input_ref_key);
      }
      continue;
    }
    const AnfNodeIndexSet &grad_param_user_nodes = mgr->node_users()[iter.first];  // %6
    // EnvironGet is a user node
    for (const auto &grad_param_iter : grad_param_user_nodes) {
      if (IsPrimitiveCNode(grad_param_iter.first, prim::kPrimEnvironGet)) {
        auto ref_to_embed = grad_param_iter.first->cast<CNodePtr>()->input(ref_to_embed_index);
        MS_EXCEPTION_IF_NULL(ref_to_embed);
        if (ref_to_embed->isa<CNode>()) {
          MS_EXCEPTION_IF_NULL(ref_to_embed->cast<CNodePtr>()->input(1));
          auto abs = ref_to_embed->cast<CNodePtr>()->input(1)->abstract();
          if (abs == nullptr) {
            continue;
          }
          auto ref_key = abstract::GetRefKeyFromAbstract(abs);
          MS_LOG(DEBUG) << "The ref_key of parameter is: " << ref_key;
          ref_keys.push_back(ref_key);
        }
      }
    }
  }
  return ref_keys;
}

std::vector<std::pair<FuncGraphPtr, std::vector<std::string>>> GetForwardFuncGraphRefKeys(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  // Record forward func_graph
  std::vector<std::pair<FuncGraphPtr, std::vector<std::string>>> forward_func_refkeys;
  const auto &all_nodes = TopoSort(root->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimJ)) {
      continue;
    }

    auto cnode = node->cast<CNodePtr>();
    auto forward_fg = GetValueNode<FuncGraphPtr>(cnode->input(1));
    if (forward_fg == nullptr) {
      continue;
    }
    auto ref_keys = GetRequiresGradRefKeys(node, forward_fg);
    forward_func_refkeys.push_back(std::make_pair(forward_fg, ref_keys));
  }
  return forward_func_refkeys;
}

std::set<std::string> GetAllRefKeys(const AnfNodePtrList &nodes) {
  std::set<std::string> ref_keys;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto abs = nodes[i]->abstract();
    if (abs == nullptr || !abs->isa<abstract::AbstractRefTensor>()) {
      continue;
    }
    auto ref_key = abstract::GetRefKeyFromAbstract(abs);
    if (ref_key == "") {
      MS_LOG(EXCEPTION) << "The AbstractRefTensor has no ref_key: " << abs->ToString();
    }
    MS_LOG(DEBUG) << "The ref_key is: " << ref_key;
    ref_keys.insert(ref_key);
  }
  return ref_keys;
}

bool HasRefKeyInput(const AnfNodePtr &node, const std::string &ref_key) {
  auto inputs = node->cast<CNodePtr>()->inputs();
  for (auto input : inputs) {
    auto abs = input->abstract();
    bool ref_key_input =
      abs && abs->isa<abstract::AbstractRefTensor>() && (abstract::GetRefKeyFromAbstract(abs) == ref_key);
    if (ref_key_input) {
      return true;
    }
  }
  return false;
}

std::map<AnfNodePtr, int64_t> InitNodeVersion(const std::vector<AnfNodePtr> &nodes, const std::string &refkey) {
  std::map<AnfNodePtr, int64_t> node_version;
  for (size_t i = 0; i < nodes.size(); ++i) {
    MS_LOG(DEBUG) << "i: " << i << "  nodes[i]: " << nodes[i]->DebugString();
    auto node = nodes[i];
    auto node_abs = node->abstract();
    bool is_ref_key =
      node_abs && node_abs->isa<abstract::AbstractRefTensor>() && (abstract::GetRefKeyFromAbstract(node_abs) == refkey);
    if (is_ref_key) {
      node_version[node] = 0;
    } else {
      node_version[node] = -1;
    }
  }
  return node_version;
}

void CheckAndRaiseException(const std::vector<int64_t> &inputs_version, const AnfNodePtr &node) {
  for (size_t index = 0; index < inputs_version.size() - 1; ++index) {
    if (inputs_version[index] != inputs_version[index + 1]) {
      MS_LOG(EXCEPTION)
        << "One of the variables needed for gradient computation has been modified by an inplace operation."
        << trace::GetDebugInfoStr(node->debug_info());
    }
  }
}

void CheckInplaceValidationInGrad(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (check_flag[func_graph] == true) {
    MS_LOG(DEBUG) << "The func has checked: " << func_graph->ToString();
    return;
  }
  MS_LOG(DEBUG) << "Check inplace validation of func in grad: " << func_graph->ToString();
  check_flag[func_graph] = true;
  // If the func_graph is from grad
  const auto &nodes = TopoSort(func_graph->get_return());
  auto all_refkeys = GetAllRefKeys(nodes);
  std::map<AnfNodePtr, int64_t> node_version;
  constexpr size_t first_index = 1;
  constexpr size_t second_index = 2;
  for (auto refkey : all_refkeys) {
    // init
    node_version = InitNodeVersion(nodes, refkey);
    for (size_t i = 0; i < nodes.size(); ++i) {
      auto node = nodes[i];
      // If the node is J node, means the backward graph has not been expanded.
      if (!node->isa<CNode>() || IsPrimitiveCNode(node, prim::kPrimJ)) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      int64_t version = node_version[node];
      MS_LOG(DEBUG) << "Check cnode:" << cnode->DebugString();
      // Has ref_key input: Load, InplaceOps, UpdateState
      auto inputs = cnode->inputs();
      if (HasRefKeyInput(node, refkey)) {
        if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
          auto first_input_version = node_version[cnode->input(first_index)];
          auto second_input_version = node_version[cnode->input(second_index)];
          if (first_input_version >= second_input_version) {
            node_version[node] = first_input_version;
            continue;
          }
          node_version[node] = second_input_version;
          continue;
        } else if (IsInplaceOps(node)) {
          auto u_monad = inputs[inputs.size() - 1];
          node_version[node] = node_version[u_monad] + 1;
          continue;
        } else if (IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
          node_version[node] = node_version[cnode->input(second_index)];
          continue;
        }
      }

      std::vector<int64_t> inputs_version;
      for (auto input : inputs) {
        // Process sub func_graph
        auto func = GetValueNode<FuncGraphPtr>(input);
        if (func != nullptr) {
          MS_LOG(DEBUG) << "Check the inplace validation of sub func:" << func->ToString();
          CheckInplaceValidationInGrad(func);
        }
        // other value node
        if (input->isa<ValueNode>()) {
          continue;
        }
        if (node_version[input] != -1) {
          MS_LOG(DEBUG) << "input:" << input->DebugString() << " version:" << node_version[input];
          (void)inputs_version.push_back(node_version[input]);
        }
      }
      if (inputs_version.size() >= 1) {
        CheckAndRaiseException(inputs_version, node);
        version = inputs_version[0];
      }
      node_version[node] = version;
      // Check until the results of the backward graph.
      auto need_check = node->user_data<bool>(NODE_FLAG_CHECK_INPLACE_GRAD);
      if (need_check != nullptr && (*need_check)) {
        MS_LOG(DEBUG) << "node need check:" << node->DebugString();
        break;
      }
    }
  }
}

bool ExistInplaceOps(const FuncGraphPtr &func) {
  MS_EXCEPTION_IF_NULL(func);
  const auto &all_nodes = TopoSort(func->return_node(), SuccDeeperSimple);
  return std::any_of(all_nodes.begin(), all_nodes.end(), [](const AnfNodePtr &node) { return IsInplaceOps(node); });
}
}  // namespace

bool InplaceValidation(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto forward_func_refkeys = GetForwardFuncGraphRefKeys(root);
  for (auto fg_refkey_pair : forward_func_refkeys) {
    auto forword_func = fg_refkey_pair.first;
    auto ref_keys = fg_refkey_pair.second;
    CheckInplaceOpValidate(forword_func, ref_keys);
  }
  return false;
}

bool InplaceValidationAfterExpand(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  const auto &all_nodes = TopoSort(root->return_node(), SuccDeeperSimple);
  for (auto node : all_nodes) {
    MS_LOG(DEBUG) << "node:" << node->DebugString();
    auto need_check = node->user_data<bool>(NODE_FLAG_CHECK_INPLACE_GRAD);
    if (need_check != nullptr && (*need_check)) {
      auto func = node->func_graph();
      if (func != nullptr && ExistInplaceOps(func)) {
        MS_LOG(DEBUG) << "Check the inplace validation of func_graph in grad: " << func->ToString();
        CheckInplaceValidationInGrad(func);
      }
    }
  }
  check_flag.clear();
  return false;
}
}  // namespace pipeline
}  // namespace mindspore
