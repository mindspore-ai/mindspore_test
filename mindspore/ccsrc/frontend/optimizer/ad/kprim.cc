/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#ifndef _WIN32
#include <dirent.h>
#endif
#include <memory>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "frontend/ir/primitive_py.h"
#include "ir/meta_func_graph.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "pipeline/jit/ps/resource.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/operator/meta_dsl/common/meta_impl.h"
#include "frontend/expander/bprop/bprop.h"
#include "frontend/expander/bprop/bprop_meta_func_graph.h"
#include "include/common/utils/primitive_utils.h"
#include "include/common/utils/utils.h"
#include "utils/symbolic.h"
#include "utils/ms_context.h"
#include "utils/info.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "utils/anf_utils.h"
#include "frontend/expander/utils.h"
#include "frontend/expander/bprop/bprop_irbuilder.h"
#include "include/common/pynative/grad_state.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_h.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace ad {
KPrim g_k_prims;

namespace {
constexpr char kLiftedUserDataKey[] = "lifted_from_fv";
constexpr char kUMonadInOutput[] = "u_monad_in_output";
constexpr char kIOMonadInOutput[] = "io_monad_in_output";

FuncGraphPtr GetBprop(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resources, const CNodePtr &cnode) {
  // Set a child scope named "grad'PrimitiveName'" for the bprop function,
  // and add "Gradients" to the front.
  static const std::string gradients_scope = "Gradients/";
  static const std::string grad_op_child_scope_prefix = "/Grad_";
  MS_EXCEPTION_IF_NULL(prim);
  const auto &prim_name = prim->name();
  auto scope = std::make_shared<Scope>(gradients_scope + ScopeManager::GetInstance().GetCurrentScope()->name() +
                                       grad_op_child_scope_prefix + prim_name);
  ScopeGuard scope_guard(scope);

  // Get bprop implemented by Meta Dsl.
  FuncGraphPtr bprop_meta_impl_graph = prim::RegMetaImplFactory::GetInstance().GetBprop(prim);
  if (bprop_meta_impl_graph != nullptr) {
    return bprop_meta_impl_graph;
  }
  // Firstly we get bprop from expander. If failed, try mindir. If still failed, try the python bprop function.
  FuncGraphPtr func_graph = expander::bprop::GetBpropMetaFuncGraph(prim, cnode);
  if (func_graph != nullptr) {
    return func_graph;
  }

  py::function fn;
  if (prim->is_base()) {
    fn = GetBpropFunction(prim_name);
  } else if (mindspore::ops::IsPrimitiveFunction(prim_name)) {
    fn = GetBpropFunction(prim_name);
  } else if (prim->isa<PrimitivePy>()) {
    fn = prim->cast_ptr<PrimitivePy>()->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim_name);
    }
  } else {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode) << "Unexpected prim: " << prim->ToString();
  }
  if (!fn || py::isinstance<py::none>(fn)) {
    MS_LOG(DEBUG) << "Fail to find bprop function for " << prim_name << ". fn: " << py::str(fn);
    return nullptr;
  }
  {
    MS_LOG_TRY_CATCH_SCOPE;
    func_graph = parse::ParsePythonCode(fn);
  }
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Fail to parse bprop function for " << prim_name << ".";
    return nullptr;
  }
  auto bprop_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP);
  if (bprop_flag) {
    func_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  }
  pipeline::ResourceBasePtr res = (resources != nullptr) ? resources : std::make_shared<pipeline::Resource>();
  (void)parse::ResolveFuncGraph(func_graph, res, false);
  return func_graph;
}

std::vector<size_t> GetNeedCloneInputIndex(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  if (!prim->inplace_prim()) {
    return {};
  }
  auto handle = mindspore::expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder(prim->name());
  if (handle != nullptr && handle->clone_inplace_input_func != nullptr) {
    return prim->rw_write_input_indexes();
  }
  return {};
}

AnfNodePtr InplaceArgsClone(const FuncGraphPtr &fprop, const FuncGraphPtr &bprop, const PrimitivePtr &prim,
                            const AnfNodePtr &umonad_arg) {
  MS_EXCEPTION_IF_NULL(prim);
  const auto &need_clone_input_index = GetNeedCloneInputIndex(prim);
  if (need_clone_input_index.empty()) {
    return umonad_arg;
  } else {
    // Need do input args clone for inplace ops
    // Change From
    // ==> primal_cnode: {kPrimInplace, args0, args1, ..., umonad}
    // To:
    // ==> new_load_cnode = Load(args0, umonad)
    // ==> new_umonad_cnode = UpdateState(umonad, new_load_cnode)
    // ==> primal_cnode: {kPrimInplace, args0, args1, ..., new_umoad_cnode}
    MS_EXCEPTION_IF_NULL(fprop);
    MS_EXCEPTION_IF_NULL(bprop);
    const auto &params = fprop->parameters();
    AnfNodePtr umonad_param = umonad_arg;
    CNodePtr bprop_meta_funcgraph_caller = nullptr;
    for (auto node : TopoSort(bprop->output())) {
      if (!node->isa<CNode>()) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();
      if (IsValueNode<expander::bprop::BpropMetaFuncGraph>(cnode->input(0))) {
        bprop_meta_funcgraph_caller = cnode;
        break;
      }
    }
    if (bprop_meta_funcgraph_caller == nullptr) {
      MS_LOG(WARNING) << "No bprop meta funcgraph found for prim: " << prim->ToString();
      return umonad_arg;
    }
    MS_EXCEPTION_IF_NULL(bprop_meta_funcgraph_caller);
    for (size_t index : need_clone_input_index) {
      const auto original_inplace_param = params[index];
      auto new_load_cnode = fprop->NewCNode({NewValueNode(prim::kPrimLoad), original_inplace_param, umonad_param});
      auto new_umonad_cnode = fprop->NewCNode({NewValueNode(prim::kPrimUpdateState), umonad_param, new_load_cnode});
      bprop_meta_funcgraph_caller->set_input(index + 1, new_load_cnode);
      MS_LOG(INFO) << "Clone bprop argument with index: " << std::to_string(index)
                   << " for inplace op: " << prim->ToString();
      umonad_param = new_umonad_cnode;
    }
    return umonad_param;
  }
}

AnfNodePtr CalDoutWithMask(const FuncGraphPtr &fg, const AnfNodePtr &dout_node, bool is_view_inplace) {
  if (!is_view_inplace) {
    return dout_node;
  }
  auto get_dout_tuple = std::make_shared<prim::GenerateBpropOutTuple>("get_dout_tuple");
  return fg->NewCNodeInOrder({NewValueNode(get_dout_tuple), dout_node});
}
}  // namespace

FuncGraphPtr KPrim::GetPrimBprop(const PrimitivePtr &prim, const ValueNodePtr &value_node,
                                 const pipeline::ResourceBasePtr &resources, const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(value_node);
  auto iter = bprop_registry_.find(prim);
  if (iter != bprop_registry_.end() && !iter->second->dropped() && !prim->HasAttr("side_effect_backprop_mem")) {
    return iter->second;
  }

  FuncGraphPtr bprop_fg = GetBprop(prim, resources, cnode);
  if (bprop_fg != nullptr) {
    // Set bprop_g graph cache
    bprop_registry_[prim] = bprop_fg;
  } else {
    bprop_fg = FakeBprop(value_node, resources);
  }

  return bprop_fg;
}

FuncGraphPtr KPrim::GetFprop(const PrimitivePtr &prim) const {
  static const std::string ad_module = "mindspore.ops._grad_experimental.grad_implementations";
  std::string func_name = "_fprop_" + prim->name();
  py::function fn = python_adapter::GetPyFn(ad_module, func_name);
  auto func_graph = parse::ParsePythonCode(fn);
  MS_EXCEPTION_IF_NULL(func_graph);
  return BasicClone(func_graph);
}

MetaFuncGraphPtr KPrim::KMetaFuncGraph(const PrimitivePtr &prim, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(prim);

  auto iter = bprop_registry_meta_.find(prim);
  if (iter != bprop_registry_meta_.end()) {
    return iter->second;
  }

  if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple)) {
    MetaFuncGraphPtr meta = std::make_shared<prim::MakeTupleGradient>("make_tuple_gradient");
    bprop_registry_meta_[prim::kPrimMakeTuple] = meta;
    return meta;
  }

  if (IsPrimitiveEquals(prim, prim::kPrimMakeList)) {
    MetaFuncGraphPtr meta = std::make_shared<prim::MakeListGradient>("make_list_gradient");
    bprop_registry_meta_[prim::kPrimMakeList] = meta;
    return meta;
  }

  if (IsPrimitiveEquals(prim, prim::kPrimMakeDict)) {
    MetaFuncGraphPtr meta = std::make_shared<prim::MakeDictGradient>("make_dict_gradient");
    bprop_registry_meta_[prim::kPrimMakeDict] = meta;
    return meta;
  }

  if (IsPrimitiveEquals(prim, prim::kPrimMutable)) {
    MetaFuncGraphPtr meta = std::make_shared<prim::MutableGradient>("MutableGradient");
    bprop_registry_meta_[prim::kPrimMutable] = meta;
    return meta;
  }

  MS_LOG_WITH_NODE(EXCEPTION, node) << "Fail to find bprop function for " << prim->name() << ".";
}

static void AddMonad(const FuncGraphPtr &bprop_fg, const CNodePtr &output, const AnfNodePtr &monad) {
  if (!IsPrimitiveCNode(output, prim::kPrimMakeTuple)) {
    constexpr char model_name[] = "mindspore.ops.composite.multitype_ops.add_impl";
    constexpr char python_ops[] = "_tuple_add";
    auto tuple_add_ops = NewValueNode(prim::GetPythonOps(python_ops, model_name));
    auto maketuple_monad = bprop_fg->NewCNode({NewValueNode(prim::kPrimMakeTuple), monad});
    auto tuple_add_monad = bprop_fg->NewCNode({tuple_add_ops, output, maketuple_monad});
    bprop_fg->set_output(tuple_add_monad);
  } else {
    output->add_input(monad);
  }
}

static void AppendMonadOutput(const FuncGraphPtr &bprop_fg, const AnfNodePtr &monad) {
  const auto &output = bprop_fg->output();
  MS_EXCEPTION_IF_NULL(output);
  auto output_cnode = output->cast<CNodePtr>();
  if (output_cnode != nullptr) {
    if (HasAbstractUMonad(monad) && !bprop_fg->has_flag(kUMonadInOutput)) {
      AddMonad(bprop_fg, output_cnode, monad);
      bprop_fg->set_flag(kUMonadInOutput, true);
    } else if (HasAbstractIOMonad(monad) && !bprop_fg->has_flag(kIOMonadInOutput)) {
      AddMonad(bprop_fg, output_cnode, monad);
      bprop_fg->set_flag(kIOMonadInOutput, true);
    }
    return;
  }
  // If output is an empty tuple, create a (make_tuple, monad) as the new output.
  auto make_tuple = NewValueNode(prim::kPrimMakeTuple);
  output_cnode = bprop_fg->NewCNode({make_tuple, monad});
  if (HasAbstractUMonad(monad)) {
    bprop_fg->set_flag(kUMonadInOutput, true);
  } else if (HasAbstractIOMonad(monad)) {
    bprop_fg->set_flag(kIOMonadInOutput, true);
  }
  bprop_fg->set_output(output_cnode);
}

// Append U or/and IO monad to output of Bprop funcgraph.
static void AdjustForAutoMonad(const PrimitivePtr &prim, const FuncGraphPtr &bprop_fg) {
  auto effect_info = GetPrimEffectInfo(prim);
  if (effect_info.memory) {
    MS_LOG(DEBUG) << "Append U monad for Bprop FuncGraph of Primitive " << prim->ToString();
    auto u = NewValueNode(kUMonad);
    u->set_abstract(kUMonad->ToAbstract());
    AppendMonadOutput(bprop_fg, u);
  }
  if (effect_info.io) {
    MS_LOG(DEBUG) << "Append IO monad for Bprop FuncGraph of Primitive " << prim->ToString();
    auto io = NewValueNode(kIOMonad);
    io->set_abstract(kIOMonad->ToAbstract());
    AppendMonadOutput(bprop_fg, io);
  }
}

std::vector<NodeDebugInfoPtr> GeneratePrimalDebugInfo(const ValueNodePtr &value_node,
                                                      const pipeline::ResourceBasePtr &resources) {
  std::vector<NodeDebugInfoPtr> primal_debug_infos;
  if (resources != nullptr) {
    auto manager = resources->manager();
    auto &users = manager->node_users()[value_node];
    for (auto user_iter = users.begin(); user_iter != users.end(); ++user_iter) {
      primal_debug_infos.push_back(user_iter->first->debug_info());
    }
  }
  return primal_debug_infos;
}

void SetDumpFlag(const PrimitivePtr &prim, const FuncGraphPtr &bprop_fg) {
  if (prim == nullptr || bprop_fg == nullptr) {
    return;
  }
  auto attr = prim->GetAttr(kAttrDump);
  if (attr != nullptr) {
    if (attr->isa<StringImm>()) {
      auto str_attr = attr->cast_ptr<StringImm>();
      MS_EXCEPTION_IF_NULL(str_attr);
      if (str_attr->value() == kValueTrue) {
        bprop_fg->set_flag(FUNC_GRAPH_FLAG_DUMP, true);
      }
    }
  }
}

FuncGraphPtr AdaptBpropInput(const FuncGraphPtr &bprop_fg) {
  auto fg = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> res_input = {NewValueNode(bprop_fg)};
  size_t len = bprop_fg->parameters().size();
  for (size_t i = 0; i < len; ++i) {
    auto input = fg->add_parameter();
    if (i != len - 1) {
      (void)res_input.emplace_back(input);
      continue;
    }
    auto get_dout = std::make_shared<prim::GetRealBpropOut>("get_dout_from_tuple");
    (void)res_input.emplace_back(fg->NewCNodeInOrder({NewValueNode(get_dout), input}));
  }
  auto res_node = fg->NewCNodeInOrder(res_input);
  fg->set_output(res_node);
  return fg;
}

FuncGraphPtr KPrim::KPrimitive(const CNodePtr &cnode, const ValueNodePtr &value_node,
                               const pipeline::ResourceBasePtr &resources, bool is_view_inplace) {
  if (!IsValueNode<Primitive>(value_node)) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode) << "Primitive node is not valid.";
  }

  auto prim = GetValueNode<PrimitivePtr>(value_node);
  if (IsPrimitiveEquals(prim, prim::kPrimSwitchLayer)) {
    auto fprop = GetFprop(prim);
    fprop->transforms().emplace("primal", FuncGraphTransform(prim::kPrimSwitchLayer));
    return fprop;
  } else if (IsPrimitiveEquals(prim, prim::kPrimMakeTuple) || IsPrimitiveEquals(prim, prim::kPrimMakeList) ||
             IsPrimitiveEquals(prim, prim::kPrimMakeDict) || IsPrimitiveEquals(prim, prim::kPrimMutable)) {
    // Return null to use Meta bprop.
    return nullptr;
  }

  FuncGraphPtr bprop_fg = nullptr;
  if (IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook)) {
    if (!pynative::GradState::Get().RequiresGrad()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode)
        << "The Hook operation is not supported in graph mode, which is only supported in pynative mode.\n"
        << trace::GetDebugInfoStr(cnode->debug_info());
    }
    bprop_fg = BpropCut(value_node, resources);
  } else {
    bprop_fg = GetPrimBprop(prim, value_node, resources, cnode);
  }
  MS_EXCEPTION_IF_NULL(bprop_fg);
  MS_EXCEPTION_IF_NULL(bprop_fg->return_node());

  SetDumpFlag(prim, bprop_fg);
  AdjustForAutoMonad(prim, bprop_fg);
  mindspore::HashMap<std::string, ValuePtr> primal_attrs;
  std::vector<NodeDebugInfoPtr> primal_debug_infos = GeneratePrimalDebugInfo(value_node, resources);
  if (cnode != nullptr) {
    primal_attrs = cnode->primal_attrs();
    cnode->AddPrimalAttr(kPrimalAttrUniqueId, MakeValue(cnode->UniqueId()));
    const auto forward_node_primal_attr = prim->name() + "_" + cnode->UniqueId();
    primal_attrs[kPrimalAttrForwardNodeName] = MakeValue(forward_node_primal_attr);
    primal_attrs[kPrimalAttrForwardUniqueId] = MakeValue(cnode->UniqueId());
  }
  if (is_view_inplace) {
    bprop_fg = AdaptBpropInput(bprop_fg);
  }
  auto expanded_fg = BpropToK(prim, bprop_fg, nullptr, cnode, primal_attrs, primal_debug_infos, is_view_inplace);
  if (expanded_fg == nullptr) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode)
      << "Failed convert " << prim->name()
      << " prim bprop function to J expanded func graph. NodeInfo: " << trace::GetDebugInfoStr(bprop_fg->debug_info());
  }
  if (lift_fv_before_grad && IsPrimitiveEquals(prim, prim::kPrimSwitch)) {
    // Inline fprop_switch before renormalize;
    expanded_fg->set_flag(FUNC_GRAPH_FLAG_FORCE_INLINE, true);
    MS_LOG(DEBUG) << "set force_inline for fg: " << expanded_fg->ToString();
  }
  return expanded_fg;
}

AnfNodePtr KPrim::BuildOutput(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg,
                              bool is_view_inplace) const {
  // The primal fg may have extra parameters from lifted fv or u_monad and io_monad.
  std::vector<AnfNodePtr> extra_lifted_args;
  std::vector<AnfNodePtr> extra_monad_args;
  // caller had checked size() - 2 is greater than 0.
  auto bprop_fg_param_size = bprop_fg->parameters().size() - 2;
  if (current_primal_fg != nullptr && bprop_fg_param_size < current_primal_fg->parameters().size()) {
    auto current_primal_fg_param_size = current_primal_fg->parameters().size();
    MS_LOG(DEBUG) << "Current Primal FuncGraph may have extra parameters(U or IO monad) which bprop don't define, so "
                     "Insert it. Extra parameters size: "
                  << current_primal_fg_param_size - bprop_fg_param_size;
    // The lifted parameters are put in front: {lifted parameters, origin parameters, u/io monad}.
    for (size_t i = 0; i < current_primal_fg_param_size; ++i) {
      auto primal_parameter = dyn_cast<Parameter>(current_primal_fg->parameters()[i]);
      MS_EXCEPTION_IF_NULL(primal_parameter);
      auto lifted = primal_parameter->user_data<bool>(kLiftedUserDataKey);
      if (lifted == nullptr || !*lifted) {
        break;
      }
      extra_lifted_args.push_back(
        bprop_fg->NewCNodeInOrder({NewValueNode(prim::GetPythonOps("zeros_like")), primal_parameter}));
      ++bprop_fg_param_size;
    }
    for (auto i = bprop_fg_param_size; i < current_primal_fg_param_size; ++i) {
      const auto &primal_node = current_primal_fg->parameters()[i];
      AnfNodePtr extra_node;
      // Simplify zeros_like(primal_node) to U or IO, so extra_node in bprop_fg will not refer to primal_node
      // as a free variable of primal_graph.
      // Notes: if the implementation of zeros_like changes, here too.
      if (HasAbstractUMonad(primal_node)) {
        extra_node = NewValueNode(kUMonad);
      } else if (HasAbstractIOMonad(primal_node)) {
        extra_node = NewValueNode(kIOMonad);
      } else {
        MS_EXCEPTION_WITH_NODE(TypeError, bprop_fg->return_node())
          << "The params of function 'bprop' of Primitive or Cell requires the forward inputs as well "
             "as the 'out' and 'dout'.\n"
          << trace::GetDebugInfoStr(bprop_fg->debug_info());
      }
      extra_monad_args.push_back(extra_node);
      MS_LOG(DEBUG) << "Insert to bprop_fg for node: " << primal_node->DebugString();
    }
  }
  // bprop_fg has been checked in caller
  if (IsPrimitiveCNode(bprop_fg->output(), prim::kPrimMakeTuple)) {
    // Set bprop output as (env, dx, dy, dz, ...)
    auto cbprop = bprop_fg->output()->cast_ptr<CNode>();
    auto &inputs = cbprop->inputs();

    std::vector<AnfNodePtr> args;
    args.push_back(NewValueNode(prim::kPrimMakeTuple));
    args.push_back(NewEnviron(bprop_fg));
    // The lifted parameters are put in front.
    if (!extra_lifted_args.empty()) {
      std::transform(
        extra_lifted_args.cbegin(), extra_lifted_args.cend(), std::back_inserter(args),
        [bprop_fg, is_view_inplace](const AnfNodePtr &arg) { return CalDoutWithMask(bprop_fg, arg, is_view_inplace); });
    }
    std::transform(
      inputs.cbegin() + 1, inputs.cend(), std::back_inserter(args),
      [bprop_fg, is_view_inplace](const AnfNodePtr &arg) { return CalDoutWithMask(bprop_fg, arg, is_view_inplace); });
    if (!extra_monad_args.empty()) {
      (void)args.insert(args.cend(), extra_monad_args.cbegin(), extra_monad_args.cend());
    }
    return NewCNode(args, bprop_fg);
  }

  // Set bprop output as (env, dx)
  constexpr char model_name[] = "mindspore.ops.composite.multitype_ops.add_impl";
  constexpr char python_ops[] = "_tuple_add";
  auto bprop_tuple_add_check_func = std::make_shared<std::function<bool(const std::vector<AbstractBasePtr> &args)>>(
    [](const std::vector<AbstractBasePtr> &args) {
      for (const auto &arg : args) {
        if (!arg->isa<abstract::AbstractTuple>()) {
          MS_EXCEPTION(TypeError) << "For bprop function, output should be a tuple, but got " << arg->ToString();
        }
      }
      return true;
    });
  auto tuple_env = NewCNode({NewValueNode(prim::kPrimMakeTuple), NewEnviron(bprop_fg)}, bprop_fg);
  auto tuple_add_ops = NewValueNode(prim::GetPythonOps(python_ops, model_name));
  if (IsValueNode<FuncGraphBase>(tuple_add_ops)) {
    auto tuple_add_func_graph = GetValueNode<FuncGraphBasePtr>(tuple_add_ops);
    MS_LOG(DEBUG) << "Get tuple add func successful. Tuple add fg: " << tuple_add_func_graph->ToString();
    auto checker = std::make_shared<FuncGraphChecker>();
    checker->AddCheckFunc<const std::vector<AbstractBasePtr> &>(bprop_tuple_add_check_func);
    tuple_add_func_graph->AddChecker("check_infer_inputs", checker);
  }

  std::vector<AnfNodePtr> res_args{NewValueNode(prim::kPrimMakeTuple)};
  if (!extra_lifted_args.empty()) {
    std::transform(
      extra_lifted_args.cbegin(), extra_lifted_args.cend(), std::back_inserter(res_args),
      [bprop_fg, is_view_inplace](const AnfNodePtr &arg) { return CalDoutWithMask(bprop_fg, arg, is_view_inplace); });
    auto extra_tuple = NewCNode(res_args, bprop_fg);
    tuple_env = NewCNode({tuple_add_ops, tuple_env, extra_tuple}, bprop_fg);
  }
  auto bprop_fg_output = bprop_fg->output();
  if (is_view_inplace) {
    auto generate_dout_tuple = std::make_shared<prim::GenerateBpropOutTuple>("generate_bprop_out_tuple");
    bprop_fg_output = bprop_fg->NewCNodeInOrder({NewValueNode(generate_dout_tuple), bprop_fg_output});
  }
  if (!extra_monad_args.empty()) {
    (void)extra_monad_args.insert(extra_monad_args.cbegin(), NewValueNode(prim::kPrimMakeTuple));
    auto extra_tuple = NewCNode(extra_monad_args, bprop_fg);
    auto old_output_extra = NewCNode({tuple_add_ops, bprop_fg_output, extra_tuple}, bprop_fg);
    return NewCNode({tuple_add_ops, tuple_env, old_output_extra}, bprop_fg);
  }

  return NewCNode({tuple_add_ops, tuple_env, bprop_fg_output}, bprop_fg);
}

static void TransformNormalArgs(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg, const FuncGraphPtr &outer,
                                std::vector<AnfNodePtr> *transf_args) {
  MS_EXCEPTION_IF_NULL(mng);
  // bprop_fg has been checked in caller
  // transform except the last 2 parameters: out, dout.
  const size_t last_parameter_sizes = 2;
  auto bprop_fg_param_size = bprop_fg->parameters().size() - last_parameter_sizes;
  for (size_t i = 0; i < bprop_fg_param_size; ++i) {
    auto p = bprop_fg->parameters()[i];
    MS_EXCEPTION_IF_NULL(p);

    TraceGuard trace_guard(MakeTraceInfo<TraceGradFprop>(p->debug_info()));
    auto transf_p = outer->add_parameter();

    (void)mng->Replace(p, transf_p);
    transf_args->push_back(transf_p);
  }
}

void KPrim::TransformArgsForPrimitive(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                      const PrimitivePtr &primitive, const FuncGraphPtr &outer,
                                      std::vector<AnfNodePtr> *const transf_args) const {
  TransformNormalArgs(mng, bprop_fg, outer, transf_args);
  // Fprop_fg for Primitive with side effect should append extra U or IO monad parameter.
  auto effect_info = GetPrimEffectInfo(primitive);
  if (effect_info.memory) {
    MS_LOG(DEBUG) << "Append U monad to Fprop FuncGraph for Primitive " << primitive->ToString();
    auto transf_p = outer->add_parameter();
    transf_args->push_back(InplaceArgsClone(outer, bprop_fg, primitive, transf_p));
  }
  if (effect_info.io) {
    MS_LOG(DEBUG) << "Append IO monad to Fprop FuncGraph for Primitive " << primitive->ToString();
    auto transf_p = outer->add_parameter();
    transf_args->push_back(transf_p);
  }
}

template <typename T>
void KPrim::TransformArgsForFuncGraph(const FuncGraphManagerPtr &mng, const FuncGraphPtr &bprop_fg,
                                      const T &current_primal_fg, const FuncGraphPtr &outer,
                                      std::vector<AnfNodePtr> *const transf_args) const {
  constexpr size_t need_filter_size = 2;
  auto bprop_fg_param_size = bprop_fg->parameters().size() - need_filter_size;
  const auto &current_primal_fg_params = current_primal_fg->parameters();
  // The lifted parameters are put in front: {lifted parameters, origin parameters, u/io monad}.
  for (size_t i = 0; i < current_primal_fg_params.size(); ++i) {
    auto primal_parameter = dyn_cast_ptr<Parameter>(current_primal_fg_params[i]);
    MS_EXCEPTION_IF_NULL(primal_parameter);
    auto lifted = primal_parameter->template user_data<bool>(kLiftedUserDataKey);
    if (lifted == nullptr || !*lifted) {
      break;
    }
    TraceGuard trace_guard(MakeTraceInfo<TraceGradFprop>(primal_parameter->debug_info()));
    auto transf_p = outer->add_parameter();
    transf_args->push_back(transf_p);
    ++bprop_fg_param_size;
  }
  TransformNormalArgs(mng, bprop_fg, outer, transf_args);
  // Current primal fg may have extra parameters after AutoMonad
  if (bprop_fg_param_size < current_primal_fg_params.size()) {
    for (auto i = bprop_fg_param_size; i < current_primal_fg_params.size(); ++i) {
      auto p = current_primal_fg_params[i];
      MS_EXCEPTION_IF_NULL(p);
      // extra parameters should be Monad.
      if (!HasAbstractMonad(p)) {
        continue;
      }
      MS_LOG(DEBUG) << "Function " << current_primal_fg->ToString()
                    << ", has extra monad parameter: " << p->DebugString()
                    << ", abstract: " << p->abstract()->ToString();

      TraceGuard trace_guard(MakeTraceInfo<TraceGradFprop>(p->debug_info()));
      auto transf_p = outer->add_parameter();
      // See also Notes on extra_node of BuildOutput.
      // Notes: No need to replace p with transf_p as the only use of p is here.
      // If extra_node in bprop_fg use p as free variable, a replacement of p is required here.
      // This replacement will make the usage of p in current_primal_fg got replaced with transf_p
      // of outer. outer will be released after it is being cloned to fprop_fg, so the func_graph_
      // in transf_p will be nullptr.
      // So the RULE is DONT tamper the current_primal_fg;
      transf_args->push_back(transf_p);
    }
  }
  if (transf_args->size() != current_primal_fg_params.size()) {
    MS_EXCEPTION_WITH_NODE(TypeError, bprop_fg->return_node())
      << "Function " << current_primal_fg->ToString() << ", The number of parameter of this primal function is "
      << current_primal_fg_params.size() << ", but the number of parameters of bprop is " << bprop_fg_param_size;
  }
}

void KPrim::CheckBprop(const FuncGraphPtr &bprop_fg, const string &prim_to_check) const {
  TraceGuard guard(MakeTraceInfo<TraceCopy>(bprop_fg->return_node()->debug_info()));
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool check_bprop_flag =
    (context->get_param<bool>(MS_CTX_CHECK_BPROP_FLAG) || common::GetCompileConfig("CHECK_BPROP") == "1");
  // Skip checking if check_bprop not set
  if (!check_bprop_flag) {
    return;
  }

  // bprop_fg has been checked in caller
  auto check_bprop_class = prim::GetPythonOps("CheckBprop", "mindspore.ops.operations._inner_ops");
  MS_EXCEPTION_IF_NULL(check_bprop_class);
  auto check_bprop =
    bprop_fg->NewCNode({NewValueNode(check_bprop_class), NewValueNode(std::make_shared<StringImm>(prim_to_check))});

  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  constexpr int primitive_size = 1;
  constexpr int brprop_offset_size = 2;
  (void)inputs.insert(inputs.cbegin() + primitive_size, bprop_fg->parameters().cbegin(),
                      bprop_fg->parameters().cend() - brprop_offset_size);
  AnfNodePtr params = bprop_fg->NewCNode(inputs);

  inputs.clear();
  inputs.push_back(check_bprop);
  inputs.push_back(bprop_fg->output());
  inputs.push_back(params);
  AnfNodePtr bprop_out = bprop_fg->NewCNode(inputs);
  bprop_fg->set_output(bprop_out);
}

FuncGraphPtr KPrim::KUserDefinedCellBprop(const FuncGraphPtr &bprop_fg, const FuncGraphPtr &current_primal_fg,
                                          bool is_view_inplace) {
  MS_EXCEPTION_IF_NULL(bprop_fg);
  // primal_fg is FuncGraph just after convert. Refer ConvertCellObjToFuncGraph.
  // current_primal_fg is specalized and AutoMoaded primal_fg;
  auto primal_fg = bprop_fg->transforms().find("primal")->second.func_graph();
  auto expanded_fg = BpropToK(primal_fg, bprop_fg, current_primal_fg, nullptr, {}, {}, is_view_inplace);
  if (expanded_fg == nullptr) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, bprop_fg->return_node())
      << "Failed convert " << primal_fg->ToString()
      << " Cell bprop function to K expanded func graph. NodeInfo: " << trace::GetDebugInfoStr(primal_fg->debug_info());
  }
  return expanded_fg;
}

FuncGraphPtr KPrim::BpropCut(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) const {
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  MS_EXCEPTION_IF_NULL(prim);
  auto &node_users = resources->manager()->node_users();

  auto &users = node_users[value_node];
  auto cnode = std::find_if(users.begin(), users.end(), [&prim](const std::pair<AnfNodePtr, int64_t> &user) -> bool {
    return IsPrimitiveCNode(user.first, prim);
  });
  if (cnode == users.end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, value_node) << "Fail to find cnode.";
  }
  auto cnode_first = cnode->first->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(cnode_first);
  auto inputs_num = cnode_first->size() - 1;

  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut");
  bprop_cut->CopyHookFunction(prim_py);

  auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
  if (!cell_id.empty()) {
    (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
  }

  outputs.push_back(NewValueNode(bprop_cut));
  for (size_t i = 0; i < inputs_num; ++i) {
    auto param = func_graph->add_parameter();
    outputs.push_back(param);
  }
  auto p1 = func_graph->add_parameter();
  auto p2 = func_graph->add_parameter();
  outputs.push_back(p1);
  outputs.push_back(p2);

  func_graph->set_output(func_graph->NewCNode(outputs));
  return func_graph;
}

FuncGraphPtr KPrim::FakeBprop(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) const {
  auto prim = value_node->value()->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(prim);
  auto &node_users = resources->manager()->node_users();

  auto &users = node_users[value_node];
  auto cnode = std::find_if(users.begin(), users.end(), [&prim](const std::pair<AnfNodePtr, int64_t> &user) -> bool {
    return IsPrimitiveCNode(user.first, prim);
  });
  if (cnode == users.end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, value_node) << "Fail to find user for " << prim->ToString();
  }
  auto cnode_first = cnode->first->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(cnode_first);
  auto inputs_num = cnode_first->size() - 1;
  auto effect_info = GetPrimEffectInfo(prim);
  // Don't add U or IO monad parameters as it will be added later.
  size_t monad_params_size = 0;
  if (effect_info.memory) {
    monad_params_size++;
  }
  if (effect_info.io) {
    monad_params_size++;
  }
  if (inputs_num < monad_params_size) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode->first)
      << "Arguments number should be greater than or equal to " << monad_params_size
      << ", but the CNode is: " << cnode->first->DebugString();
  }
  inputs_num -= monad_params_size;

  auto func_graph = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> outputs;
  outputs.push_back(NewValueNode(prim::kPrimMakeTuple));

  auto fake_bprop = std::make_shared<Primitive>("fake_bprop");
  (void)fake_bprop->AddAttr("info", MakeValue("Primitive " + prim->name() + "'s bprop not defined."));

  for (size_t i = 0; i < inputs_num; ++i) {
    // Mock params for inputs
    auto param = func_graph->add_parameter();
    // Mock derivatives for each inputs
    if (IsPrimitiveEquals(prim, prim::kPrimUpdateState)) {
      outputs.push_back(func_graph->NewCNode({NewValueNode(prim::GetPythonOps("zeros_like")), param}));
    } else {
      outputs.push_back(func_graph->NewCNode({NewValueNode(fake_bprop), param}));
    }
  }
  // mock params for out and dout
  (void)func_graph->add_parameter();
  (void)func_graph->add_parameter();
  func_graph->set_output(func_graph->NewCNode(outputs));
  return func_graph;
}

bool KPrim::CheckCustomVjp(const FuncGraphPtr &bprop_fg) const {
  MS_EXCEPTION_IF_NULL(bprop_fg);
  auto parameters_size = bprop_fg->parameters().size();
  if (bprop_fg->has_flag("custom_vjp") && parameters_size == 1) {
    return true;
  }
  return false;
}

FuncGraphPtr KPrim::GetCustomVjpBprop(const FuncGraphPtr &bprop_fg) const {
  MS_EXCEPTION_IF_NULL(bprop_fg);
  auto bprop_fg_output = dyn_cast<CNode>(bprop_fg->output());
  MS_EXCEPTION_IF_NULL(bprop_fg_output);
  // Check the definition of the bprop function
  if (IsValueNode<None>(bprop_fg_output->input(1))) {
    MS_EXCEPTION_WITH_NODE(TypeError, bprop_fg_output)
      << "The bprop function of @custom_vjp is undefined. Please use 'defbwd(bprop)' to define the 'bprop' function.";
  }

  auto custom_vjp_bprop_fg = GetValueNode<FuncGraphPtr>(bprop_fg_output->input(1));
  if (custom_vjp_bprop_fg != nullptr) {
    custom_vjp_bprop_fg->set_transforms(bprop_fg->transforms());
    return custom_vjp_bprop_fg;
  } else {
    MS_EXCEPTION_WITH_NODE(TypeError, bprop_fg_output)
      << "The 'bprop' function defined by @custom_vjp defbwd(bprop) is illegal.";
  }
}
}  // namespace ad
}  // namespace mindspore
