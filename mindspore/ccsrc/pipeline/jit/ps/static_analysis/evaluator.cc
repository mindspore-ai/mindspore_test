/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/static_analysis/evaluator.h"

#include <algorithm>
#include <ostream>
#include <utility>

#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "utils/hash_set.h"
#include "ir/func_graph_cloner.h"
#include "abstract/utils.h"
#include "include/common/fallback.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "utils/ms_context.h"
#include "utils/compile_config.h"
#include "pipeline/jit/ps/static_analysis/stack_frame.h"
#include "pipeline/jit/ps/static_analysis/async_eval_result.h"
#include "pipeline/jit/ps/executor/graph_executor_py.h"
#include "pipeline/jit/ps/pipeline.h"
#include "frontend/expander/bprop/bprop_meta_func_graph.h"
#include "frontend/operator/composite/unpack_call.h"
#include "frontend/optimizer/fallback_rewriter.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"

namespace mindspore {
namespace abstract {
namespace {
string EvalEntryLogging(const EvaluatorPtr &evaluator, const AbstractBasePtrList &arg_abs_list,
                        const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(evaluator);
  std::stringstream ss;
  if (out_conf != nullptr) {
    MS_EXCEPTION_IF_NULL(out_conf->node());
    MS_EXCEPTION_IF_NULL(out_conf->node()->scope());
    ss << "Evaluator " << evaluator->ToString() << " run for " << out_conf->node()->scope()->name();
  }
  for (size_t i = 0; i < arg_abs_list.size(); i++) {
    ss << evaluator->ToString() << " input[" << i
       << "] abstract value: " << (arg_abs_list[i] ? arg_abs_list[i]->ToString() : "null abstract.");
  }
  return ss.str();
}

void EvalFailLogging(const EvaluatorPtr &evaluator, const AbstractBasePtrList &, const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(evaluator);
  if (out_conf != nullptr) {
    auto node = out_conf->node();
    MS_EXCEPTION_IF_NULL(node);
    if (IsValueNode<Primitive>(node)) {
      MS_LOG(ERROR) << "Evaluator " << evaluator->ToString() << " run failed for node " << node->fullname_with_scope()
                    << ", with debug info: " << trace::GetDebugInfoStr(node->debug_info());
    } else {
      MS_LOG(ERROR) << "Evaluator " << evaluator->ToString() << " run failed for node " << node->DebugString()
                    << ", with debug info: " << trace::GetDebugInfoStr(node->debug_info());
    }
  }
}

bool ContainsAbstractAnyInner(const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<AbstractSequence>()) {
    auto abs_list = abs->cast<AbstractSequencePtr>();
    const auto &elements = abs_list->elements();
    return std::any_of(elements.begin(), elements.end(), [](const AbstractBasePtr &e) {
      MS_EXCEPTION_IF_NULL(e);
      return ContainsAbstractAnyInner(e);
    });
  }
  if (abs->isa<AbstractKeywordArg>()) {
    auto abs_value = abs->cast<AbstractKeywordArgPtr>()->get_arg();
    return ContainsAbstractAnyInner(abs_value);
  }
  return abs->isa<AbstractAny>();
}

TypePtr GetArgsUniqueDtype(const AbstractBasePtrList &args_abs_list) {
  TypePtr res = nullptr;
  for (const auto &arg : args_abs_list) {
    MS_EXCEPTION_IF_NULL(arg);
    if (!arg->isa<AbstractTensor>()) {
      continue;
    }
    // Check default dtype if it's AbstractAny(AbstractTensor)
    if (arg->isa<abstract::AbstractAny>()) {
      auto any_arg = arg->cast_ptr<abstract::AbstractAny>();
      MS_EXCEPTION_IF_NULL(any_arg);
      if (!any_arg->supposed_tensor_dtype()) {
        continue;
      }
    }
    // Fetch the dtype from item of tensor.
    auto tensor_abs = arg->cast_ptr<AbstractTensor>();
    MS_EXCEPTION_IF_NULL(tensor_abs);
    MS_EXCEPTION_IF_NULL(tensor_abs->element());
    const auto dtype = tensor_abs->element()->BuildType();
    MS_EXCEPTION_IF_NULL(dtype);
    if (res == nullptr) {
      res = dtype;
      continue;
    }
    if (dtype != res) {
      return nullptr;
    }
  }
  return res;
}

FuncGraphPtr GetCloneBpropGraph(const MetaFuncGraphPtr &meta_func_graph, const FuncGraphPtr &generated_func_graph,
                                const AnfNodePtr &bound_node, const ScopePtr &scope) {
  MS_EXCEPTION_IF_NULL(meta_func_graph);
  auto bound_cnode = dyn_cast_ptr<CNode>(bound_node);
  if (bound_cnode == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "For BpropMetaFuncGraph '" << meta_func_graph->ToString()
                               << "', the evaluator should have the bound cnode.";
  }
  PrimalAttrGuard primal_attr_guard(bound_cnode->primal_attrs());
  const auto &primal_debug_infos = bound_cnode->primal_debug_infos();
  std::vector<NodeDebugInfoPtr> primal_debug_infos_vec;
  (void)std::copy(primal_debug_infos.begin(), primal_debug_infos.end(), std::back_inserter(primal_debug_infos_vec));
  PrimalDebugInfoGuard primal_debug_info_guard(primal_debug_infos_vec);
  FuncGraphPtr cloned_func_graph =
    BasicClone(generated_func_graph, false, std::make_shared<UpdateInfo>(scope, bound_cnode->debug_info()));
  return cloned_func_graph;
}

uint32_t GetMaxCallDepth() {
  int32_t max_call_depth = pipeline::GraphExecutorPy::GetInstance()->max_call_depth();
  if (max_call_depth != -1) {
    return IntToUint(max_call_depth);
  }
  return MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_MAX_CALL_DEPTH);
}
}  // namespace

bool ContainsAbstractAny(const AbstractBasePtrList &args_abs_list) {
  return std::any_of(args_abs_list.cbegin(), args_abs_list.cend(), [](const AbstractBasePtr &item) {
    MS_EXCEPTION_IF_NULL(item);
    return ContainsAbstractAnyInner(item);
  });
}

// MakeTuple and MakeList will handle AbstractAny in ops infer.
static const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> ignore_any_type_checking_prims{
  prim::kPrimReturn,         prim::kPrimDepend,       prim::kPrimSwitch,      prim::kPrimSwitchLayer,
  prim::kPrimUpdateState,    prim::kPrimLoad,         prim::kPrimIsConstant,  prim::kPrimMakeKeywordArg,
  prim::kPrimIsShapeUnknown, prim::kPrimIsDimUnknown, prim::kPrimListGetItem, prim::kPrimTupleGetItem,
  prim::kPrimSequenceLen,    prim::kPrimMakeDict,     prim::kPrimMutable};

AbstractBasePtrList EvaluateArguments(const ConfigPtrList &args_conf_list) {
  AbstractBasePtrList args_abs_list;
  args_abs_list.reserve(args_conf_list.size());
  for (auto &config : args_conf_list) {
    MS_EXCEPTION_IF_NULL(config);
    auto result = config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(result);
    const auto &abs = result->abstract();
    // Check if there's an inplace abstract and use it.
    MS_EXCEPTION_IF_NULL(abs);
    (void)args_abs_list.emplace_back(abs);
  }
  return args_abs_list;
}

bool CheckIfAlwaysEval(const AnfNodeConfigPtr &conf, const AbstractBasePtr &arg) {
  MS_EXCEPTION_IF_NULL(arg);
  auto new_sequence = dyn_cast_ptr<AbstractSequence>(arg);
  if (new_sequence != nullptr && !new_sequence->dynamic_len() && new_sequence->sequence_nodes() != nullptr &&
      new_sequence->size() != 0) {
    const auto &prev_result = ObtainEvalResultFromCache(conf);
    if (prev_result == nullptr) {
      return false;
    }
    auto prev_abs = prev_result->abstract();
    auto old_sequence = dyn_cast_ptr<AbstractSequence>(prev_abs);
    if (old_sequence != nullptr &&
        (old_sequence->sequence_nodes() == nullptr || old_sequence->sequence_nodes()->empty()) && *arg == *prev_abs) {
      MS_LOG(DEBUG) << "Always eval";
      return true;
    }
  }
  return false;
}

void BaseFuncGraphEvaluator::EnterStackFrame(const AnalysisEnginePtr &engine, const StackFramePtr &current_stack_frame,
                                             const StackFramePtr &new_stack_frame) {
  MS_EXCEPTION_IF_NULL(current_stack_frame);
  MS_EXCEPTION_IF_NULL(new_stack_frame);
  MS_EXCEPTION_IF_NULL(engine);
  // Enter new func graph.
  auto &current_node = current_stack_frame->CurrentNode();
  auto current_context = current_stack_frame->current_context();
  AnfNodeConfigPtr call_conf = engine->MakeConfig(current_node, current_context, current_context->func_graph());
  auto evaluator = new_stack_frame->evaluator();
  MS_EXCEPTION_IF_NULL(evaluator);
  auto new_context = new_stack_frame->current_context();
  trace::TraceGraphEvalEnter(new_context, call_conf);
  engine->PushGraphAmpStrategy(new_stack_frame->func_graph());

  // Increase & Check the func graph call depth.
  // Don't check it if the user set no_recursive flag.
  IncreaseFunctionCallDepth();
  IncreaseStackFrameDepth();
  const auto &top_graph = engine->top_func_graph();
  bool no_recursive = (top_graph == nullptr ? false : top_graph->has_flag(FUNC_GRAPH_FLAG_NO_RECURSIVE));
  const uint32_t max_depth = GetMaxCallDepth();
  if (!no_recursive && FunctionCallDepth() > max_depth) {
    MS_LOG(EXCEPTION) << "Exceed function call depth limit " << max_depth
                      << ", (function call depth: " << FunctionCallDepth()
                      << ", simulate call depth: " << StackFrameDepth() << ").\n"
                      << "It's always happened with complex construction of code or infinite recursion or loop.\n"
                      << "Please check the code if it's has the infinite recursion "
                      << "or call 'mindspore.set_recursion_limit(recursion_limit=value)' to adjust this value.\n"
                      << "If recursion_limit is set larger, the system max stack depth should be set larger too "
                      << "to avoid stack overflow.\n"
                      << "For more details, please refer to the FAQ at https://www.mindspore.cn.";
  }
  MS_LOG(DEBUG) << evaluator << "(" << evaluator->type_name() << "/" << evaluator->ToString()
                << "), enter, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();
}

void BaseFuncGraphEvaluator::LeaveStackFrame(const AnalysisEnginePtr &engine,
                                             const StackFramePtr &current_stack_frame) {
  MS_EXCEPTION_IF_NULL(current_stack_frame);
  // Leave current func graph.
  auto current_context = current_stack_frame->current_context();
  trace::TraceGraphEvalLeave(current_context);
  engine->PopGraphAmpStrategy();

  // Decrease the func graph call depth.
  DecreaseFunctionCallDepth();
  DecreaseStackFrameDepth();

  auto evaluator = current_stack_frame->evaluator();
  MS_EXCEPTION_IF_NULL(evaluator);
  MS_LOG(DEBUG) << evaluator << "(" << evaluator->type_name() << "/" << evaluator->ToString()
                << "), leave, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();
}

// Start running stack frames in a Evaluator.
AbstractBasePtr BaseFuncGraphEvaluator::LaunchStackFrame(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                                         const AnalysisContextPtr &context) {
  EvalResultPtr eval_result = nullptr;
  AbstractBasePtr abstract = nullptr;
  std::stack<StackFramePtr> stack_frames;
  auto current_stack_frame = std::make_shared<StackFrame>(shared_from_base<Evaluator>(), fg, context, parent_context_);
  MS_LOG(DEBUG) << "[" << this << "/StackFrame] Start at func graph, " << current_stack_frame;
  stack_frames.push(current_stack_frame);
  while (true) {
    current_stack_frame = stack_frames.top();
    MS_EXCEPTION_IF_NULL(current_stack_frame);
    if (current_stack_frame->Done()) {
      MS_EXCEPTION_IF_NULL(abstract);
      MS_EXCEPTION_IF_NULL(current_stack_frame->func_graph());
      if (current_stack_frame->func_graph()->has_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP)) {
        // Set all fprop outputs as used.
        SetSequenceElementsUseFlagsRecursively(abstract, true);
      }
      MS_LOG(DEBUG) << "[" << this << "/StackFrame] Leave from func graph, " << current_stack_frame;
      stack_frames.pop();
      if (stack_frames.empty()) {
        MS_LOG(DEBUG) << "[" << this << "/StackFrame] Finish at func graph, " << current_stack_frame
                      << ", abstract: " << abstract->ToString();
        break;
      }
      // Leave current func graph.
      LeaveStackFrame(engine, current_stack_frame);
      // Switch the stack frame.
      auto last_stack_frame = current_stack_frame;
      current_stack_frame = stack_frames.top();
      MS_LOG(DEBUG) << "[" << this << "/StackFrame] Back to func graph, " << current_stack_frame;
      current_stack_frame->Back(engine, last_stack_frame, eval_result);
      continue;
    }

    auto new_stack_frame = current_stack_frame->Jump(engine);
    if (new_stack_frame != nullptr) {
      // Enter new func graph.
      EnterStackFrame(engine, current_stack_frame, new_stack_frame);
      // Update current stack frame.
      stack_frames.push(new_stack_frame);
      MS_LOG(DEBUG) << "[" << this << "/StackFrame] Jump to new func graph, " << new_stack_frame;
      continue;
    }

    eval_result = current_stack_frame->Step(engine);
    MS_EXCEPTION_IF_NULL(eval_result);
    abstract = eval_result->abstract();
  }
  return abstract;
}

AbstractBasePtr BaseFuncGraphEvaluator::LaunchRecursiveEval(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                                            const AnalysisContextPtr &context) const {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(engine);
  const AnfNodePtr &func_node = fg->get_return();
  const auto &all_nodes = TopoSort(func_node, SuccIncoming, [](const AnfNodePtr &node) -> IncludeType {
    MS_EXCEPTION_IF_NULL(node);
    static const bool enable_pre_lift = (common::GetCompileConfig("PRE_LIFT") == "1");
    if (node->isa<ValueNode>() || node->isa<Parameter>() ||
        (enable_pre_lift && IsPrimitiveCNode(node, prim::kPrimPartial))) {
      return EXCLUDE;
    }
    return FOLLOW;
  });
  AbstractBasePtr abstract = nullptr;
  for (const auto &node : all_nodes) {
    MS_EXCEPTION_IF_NULL(node);
    AnfNodeConfigPtr node_conf = engine->MakeConfig(node, context, fg);
    MS_LOG(DEBUG) << "Analysis node begin, func graph: " << fg << "/" << fg->ToString()
                  << ", node: " << node->DebugString() << ", node_conf: " << node_conf->ToString();
    EvalResultPtr node_eval_result = nullptr;
    if (always_eval_flag()) {
      MS_LOG(DEBUG) << "Always eval node";
      node_eval_result = engine->ObtainEvalResultWithoutCache(node_conf);
    } else {
      node_eval_result = ObtainEvalResultFromCache(node_conf);
      if (node_eval_result != nullptr) {
        const auto &cnode = node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode);
        const auto &maybe_func = engine->GetCNodeOperatorAbstract(cnode, context, fg);
        if (maybe_func->isa<MetaFuncGraphAbstractClosure>() || maybe_func->isa<FuncGraphAbstractClosure>()) {
          const auto &abs_func_graph = maybe_func->cast<AbstractFunctionPtr>();
          SynchronizeSequenceElementsUseFlagsForFuncGraphArgs(engine, fg, cnode, abs_func_graph, context);
        }

        if (engine->check_side_effect() && node_eval_result->has_side_effect_node()) {
          MS_LOG(DEBUG) << "Found side-effect, cnode: " << cnode->DebugString() << ", func_graph: " << fg->ToString();
          cnode->set_has_side_effect_node(true);
          fg->set_has_side_effect_node(true);
        }
        MS_LOG(DEBUG) << "No need to jump as found result from cache for node_config";
      } else {
        node_eval_result = engine->ObtainEvalResultWithoutCache(node_conf);
      }
    }
    MS_EXCEPTION_IF_NULL(node_eval_result);
    abstract = node_eval_result->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    MS_LOG(DEBUG) << GetInferThread() << "Eval ( " << node_conf->ToString() << ") = " << abstract->ToString();
  }
  MS_EXCEPTION_IF_NULL(abstract);
  if (fg->has_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP)) {
    // Set all fprop outputs as used.
    SetSequenceElementsUseFlagsRecursively(abstract, true);
  }
  return abstract;
}

EvalResultPtr BaseFuncGraphEvaluator::Eval(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list,
                                           const AnfNodeConfigPtr &out_conf) {
  auto eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
  if (eval_result != nullptr) {
    MS_LOG(ERROR) << ToString() << ArgsToString(args_abs_list) << " entered again. There is something wrong.";
    return eval_result;
  }
  MS_LOG(DEBUG) << ToString() << " entered first.";
  MS_EXCEPTION_IF_NULL(engine);
  // Increase & Check the func graph call depth.
  // Don't check it if the user set no_recursive flag.
  IncreaseFunctionCallDepth();
  const auto &top_graph = engine->top_func_graph();
  bool no_recursive = (top_graph == nullptr ? false : top_graph->has_flag(FUNC_GRAPH_FLAG_NO_RECURSIVE));
  const uint32_t max_depth = GetMaxCallDepth();
  if (!no_recursive && FunctionCallDepth() > max_depth) {
    MS_LOG(EXCEPTION) << "Exceed function call depth limit " << max_depth
                      << ", (function call depth: " << FunctionCallDepth()
                      << ", simulate call depth: " << StackFrameDepth() << ").\n"
                      << "It's always happened with complex construction of code or infinite recursion or loop.\n"
                      << "Please check the code if it's has the infinite recursion "
                      << "or call 'mindspore.set_recursion_limit(recursion_limit=value)' to adjust this value.\n"
                      << "If recursion_limit is set larger, the system max stack depth should be set larger too "
                      << "to avoid stack overflow.\n"
                      << "For more details, please refer to the FAQ at https://www.mindspore.cn.";
  }
  MS_LOG(DEBUG) << this << "(" << type_name() << "/" << ToString()
                << "), enter, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();

  FuncGraphPtr fg = GetFuncGraph(engine, args_abs_list);
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(parent_context_);
  auto context = NewContext(parent_context_, fg, args_abs_list);
  trace::TraceGraphEvalEnter(context, out_conf);
  engine->PushGraphAmpStrategy(fg);

  std::size_t nargs = fg->parameters().size();
  if (args_abs_list.size() != nargs) {
    MS_EXCEPTION(TypeError) << "The parameters number of the function is " << fg->parameters().size()
                            << ", but the number of provided arguments is " << args_abs_list.size() << ".\n"
                            << "FunctionGraph : " << fg->ToString()
                            << "\nNodeInfo: " << trace::GetDebugInfoStr(fg->debug_info());
  }
  MS_LOG(DEBUG) << GetInferThread() << "@" << fg->ToString() << ArgsToString(args_abs_list) << " { ";
  if (parent_context_->func_graph() != nullptr) {
    MS_LOG(DEBUG) << GetInferThread() << "graph_: " << AnalysisSchedule::thread_id() << ":"
                  << parent_context_->func_graph()->ToString() << "()->" << AnalysisSchedule::thread_id() << ":"
                  << fg->ToString() << "();";
  }

  auto func_graph_evaluator = mindspore::cast<FuncGraphEvaluator>(this);
  if (func_graph_evaluator != nullptr) {
    MS_EXCEPTION_IF_NULL(engine->root_func_graph());
    if (engine->root_func_graph() == func_graph_evaluator->func_graph()) {
      engine->set_root_context(context);
      if (common::GetCompileConfig("STRICT_CHECK_PARENT_CONTEXT") != "1") {
        MS_LOG(INFO) << "Add root_func_graph_backup";
        engine->set_root_func_graph_backup(fg);
      }
    }
  }
  bool always_eval_flag = false;
  const auto &parameters = fg->parameters();
  for (size_t i = 0; i < nargs; i++) {
    const auto &arg = args_abs_list[i];
    const auto &node = parameters[i];
    AnfNodeConfigPtr conf = engine->MakeConfig(node, context, fg);
    always_eval_flag = always_eval_flag || CheckIfAlwaysEval(conf, arg);
    auto result = std::make_shared<EvalResult>(arg, nullptr);
    engine->SaveEvalResultInCache(conf, result);
    MS_EXCEPTION_IF_NULL(arg);
    MS_LOG(DEBUG) << GetInferThread() << ", Save argument[" << i << "] result for " << fg->ToString()
                  << ", NodeConfig: " << conf->ToString() << ", result: " << arg << "/" << arg->ToString();
  }
  PushAlwaysEvalFlag(always_eval_flag);
  if (fg->get_return() == nullptr) {
    MS_LOG(EXCEPTION) << "The func graph " << fg << "/" << fg->ToString() << " has no return node.";
  }
  MS_LOG(DEBUG) << "Analysis FuncGraph begin, func graph: " << fg << "/" << fg->ToString()
                << ", context: " << context->ToString() << ", return node: " << fg->get_return()->DebugString()
                << ", parent: " << (parent_context_->func_graph() ? parent_context_->func_graph()->ToString() : "NULL")
                << ", current function call depth: " << FunctionCallDepth();
  AbstractBasePtr abstract = nullptr;
  if (engine->enable_recursive_eval()) {
    abstract = LaunchRecursiveEval(engine, fg, context);
  } else {
    abstract = LaunchStackFrame(engine, fg, context);
  }
  PopAlwaysEvalFlag();

  MS_EXCEPTION_IF_NULL(abstract);
  MS_LOG(DEBUG) << "Analysis FuncGraph end, " << fg << "/" << fg->ToString()
                << ", evaluated abstract: " << abstract->ToString() << ", is stub: " << fg->stub();
  if (fg->stub()) {
    abstract = std::make_shared<AbstractUndetermined>();
  }
  MS_LOG(DEBUG) << GetInferThread() << "} //" << fg->ToString() << " = " << abstract->ToString();

  SyncFuncGraphSideEffectFlag(fg);

  trace::TraceGraphEvalLeave(context);
  engine->PopGraphAmpStrategy();
  // Decrease the func graph call depth.
  DecreaseFunctionCallDepth();
  MS_LOG(DEBUG) << this << "(" << type_name() << "/" << ToString()
                << "), leave, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();
  auto res = std::make_shared<EvalResult>(abstract, nullptr);
  return res;
}

void BroadenArgs(const AbstractBasePtrList &args_abs_list, AbstractBasePtrList *broaded_args, bool broaden_scalar) {
  MS_EXCEPTION_IF_NULL(broaded_args);
  (void)std::transform(
    args_abs_list.begin(), args_abs_list.end(), std::back_inserter(*broaded_args),
    [&broaden_scalar](const AbstractBasePtr &arg) -> AbstractBasePtr {
      auto arg_sequence = arg->cast<AbstractSequencePtr>();
      if (arg_sequence != nullptr && !arg_sequence->dynamic_len() && !arg->isa<AbstractSparseTensor>()) {
        MS_LOG(DEBUG) << "set as arg of dyn len param, arg:" << arg->ToString();
        auto dyn_len_arg = arg_sequence->BroadenToDynamicLenSequence();
        return broaden_scalar ? AbstractBroaden(dyn_len_arg) : dyn_len_arg->Broaden();
      }
      if (arg->GetValueTrack() != kValueAny) {
        return broaden_scalar ? AbstractBroaden(arg) : arg->Broaden();
      }
      return arg;
    });
}

AbstractBasePtrList FuncGraphEvaluator::NormalizeArgs(const AbstractBasePtrList &args_abs_list) const {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (func_graph_->has_flag(FUNC_GRAPH_FLAG_ROLLED_HEADER)) {
    AbstractBasePtrList new_args_abs_list;
    bool contain_constant = false;
    for (size_t i = 0; i < args_abs_list.size(); ++i) {
      if (i == 1) {
        new_args_abs_list.emplace_back(AbstractBroaden(args_abs_list[i]));
      } else {
        // Do not do broaden for the value may be used.
        if (i != 0 &&
            (!args_abs_list[i]->BuildValue()->ContainsValueAny() || args_abs_list[i]->isa<AbstractSequence>())) {
          contain_constant = true;
          break;
        }
        new_args_abs_list.emplace_back(args_abs_list[i]);
      }
    }
    if (contain_constant) {
      return args_abs_list;
    }
    MS_LOG(DEBUG) << "new_args_abs_list: " << mindspore::ToString(new_args_abs_list);
    return new_args_abs_list;
  }
  if (func_graph_->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
    AbstractBasePtrList broadened_list;
    auto broaden_scalar = !func_graph_->has_flag(FUNC_GRAPH_FLAG_VMAP_TRANSFORMED);
    BroadenArgs(args_abs_list, &broadened_list, broaden_scalar);
    MS_LOG(DEBUG) << func_graph_->ToString() << ", original: " << mindspore::ToString(args_abs_list)
                  << ", broadened: " << mindspore::ToString(broadened_list);
    return broadened_list;
  }
  return args_abs_list;
}

AbstractBasePtrList FuncGraphEvaluator::BroadenUndeterminedArgs(const AbstractBasePtrList &args_abs_list,
                                                                const AnalysisEnginePtr &engine) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (func_graph_->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
    return args_abs_list;
  }
  // Set ignore flag for mutlithread eval.
  engine->SetIgnoreValueFlag(AnalysisSchedule::thread_id(), func_graph_.get());
  // Set ignore flag for recursive eval.
  if (func_graph_->has_flag(kFuncGraphFlagUndetermined)) {
    func_graph_->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
    MS_LOG(DEBUG) << "Set " << func_graph_->ToString() << " with IGNORE_VALUES flag in recursive eval.";
  }
  if (func_graph_->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
    auto normalized_args_abs_list = NormalizeArgs(args_abs_list);
    MS_LOG(DEBUG) << "Normalized args " << mindspore::ToString(normalized_args_abs_list);
    return normalized_args_abs_list;
  }
  return args_abs_list;
}

FuncGraphPtr FuncGraphEvaluator::GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list) {
  auto iter = func_graph_cache_.find(args_abs_list);
  FuncGraphPtr res;
  if (iter == func_graph_cache_.end()) {
    auto fg = func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    FuncGraphPtr generated_graph = fg->GenerateFuncGraph(args_abs_list);
    func_graph_cache_[args_abs_list] = generated_graph;
    MS_LOG(DEBUG) << "Generate special instance of function graph: " << ToString()
                  << ", special function: " << generated_graph->ToString() << ", args: " << ArgsToString(args_abs_list);

    MS_EXCEPTION_IF_NULL(engine);
    MS_EXCEPTION_IF_NULL(engine->func_graph_manager());
    engine->func_graph_manager()->AddFuncGraph(generated_graph);
    if (engine->check_side_effect()) {
      generated_graph->PresetCertainSideEffect();
    }
    res = generated_graph;
  } else {
    res = iter->second;
  }

  // For the top graph, if it is replaced by generated graph, update the top graph to the new one.
  if (engine->top_func_graph() == func_graph()) {
    if (res != func_graph()) {
      engine->set_top_func_graph(res);
      parse::Parser::UpdateTopFuncGraph(res);
    }
  }
  return res;
}

FuncGraphPtr MetaFuncGraphEvaluator::GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list) {
  auto iter = func_graph_cache_.find(args_abs_list);
  if (iter != func_graph_cache_.end()) {
    return iter->second;
  }
  MS_EXCEPTION_IF_NULL(meta_func_graph_);
  (void)meta_func_graph_->GetChecker("check_infer_inputs").Execute(args_abs_list);

  if (scope_ != nullptr) {
    meta_func_graph_->set_scope_name(scope_->name());
  }
  if (this->bound_node() != nullptr) {
    auto node_debug_info = bound_node()->debug_info();
    TraceGuard trace_guard(MakeTraceInfo<TraceGenMetaFuncGraph>(node_debug_info));  // Allow null debug info.
    if (node_debug_info != nullptr) {
      auto node_location = trace::GetSourceCodeDebugInfo(node_debug_info)->location();
      if (node_location != nullptr) {
        meta_func_graph_->set_node_expr_src(node_location->expr_src());
      }
    }
    generated_func_graph_ = meta_func_graph_->GenerateFuncGraph(args_abs_list);
  } else {
    generated_func_graph_ = meta_func_graph_->GenerateFuncGraph(args_abs_list);
  }

  FuncGraphPtr cloned_func_graph;
  NodeDebugInfoPtr debug_info;
  if (this->bound_node() != nullptr) {
    debug_info = this->bound_node()->debug_info();
  }
  if (meta_func_graph_->isa<expander::bprop::BpropMetaFuncGraph>()) {
    cloned_func_graph = GetCloneBpropGraph(meta_func_graph_, generated_func_graph_, this->bound_node(), scope_);
  } else {
    cloned_func_graph = BasicClone(generated_func_graph_, false, std::make_shared<UpdateInfo>(scope_, debug_info));
  }
  func_graph_cache_[args_abs_list] = cloned_func_graph;
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(engine->func_graph_manager());
  engine->func_graph_manager()->AddFuncGraph(cloned_func_graph);
  if (engine->check_side_effect()) {
    cloned_func_graph->PresetCertainSideEffect();
  }
  return cloned_func_graph;
}

EvalResultPtr Evaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                             const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  args_abs_list = NormalizeArgs(args_abs_list);
  args_abs_list = BroadenUndeterminedArgs(args_abs_list, engine);
  MS_LOG(DEBUG) << EvalEntryLogging(shared_from_base<Evaluator>(), args_abs_list, out_conf);
  EvalResultPtr eval_result = nullptr;
  const std::string &evaluator_name = ToString();
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto &cache = evaluator_cache_mgr_->GetCache();
  auto iter = cache.find(args_abs_list);
  if (iter == cache.end()) {
    MS_LOG(DEBUG) << "[" << this << "/" << evaluator_name << "] cache miss, call Eval(), args: " << args_abs_list;
    eval_result = Eval(engine, args_abs_list, out_conf);
    MS_EXCEPTION_IF_NULL(eval_result);
    if (eval_result->abstract() == nullptr) {
      EvalFailLogging(shared_from_base<Evaluator>(), args_abs_list, out_conf);
      MS_LOG(INTERNAL_EXCEPTION) << "Evaluator " << evaluator_name << " result is nullptr.";
    }
    MS_LOG(DEBUG) << "[" << this << "/" << evaluator_name
                  << "] set cache. result: " << eval_result->abstract()->ToString()
                  << ", args_abs_list hash: " << AbstractBasePtrListHash(args_abs_list)
                  << ", args_abs_list: " << args_abs_list;
    evaluator_cache_mgr_->SetValue(args_abs_list, eval_result);
  } else {
    eval_result = iter->second;
    MS_EXCEPTION_IF_NULL(eval_result->abstract());
    MS_LOG(DEBUG) << "[" << this << "/" << evaluator_name
                  << "] cache hit. result: " << eval_result->abstract()->ToString() << ", args: " << args_abs_list;
    for (size_t i = 0; i < args_abs_list.size(); ++i) {
      const auto &old_arg = iter->first[i];
      const auto &new_arg = args_abs_list[i];
      // Update inputs abstract, if matched in cache.
      SynchronizeSuccessiveInputs(old_arg, new_arg);
    }
  }
  return eval_result;
}

EvalResultPtr Evaluator::EvalUndeterminedArgs(const AbstractBasePtrList &args_abs_list) {
  auto is_undetermined = std::any_of(args_abs_list.begin(), args_abs_list.end(), [](auto &arg) -> bool {
    return arg->IsSameTypeId(AbstractUndetermined::kTypeId);
  });
  if (is_undetermined) {
    MS_LOG(DEBUG) << "Eval " << identifier_ << " return undetermined abstract result";
    return std::make_shared<EvalResult>(std::make_shared<AbstractUndetermined>(), std::make_shared<AttrValueMap>());
  }
  return nullptr;
}

EvalResultPtr TrivialPrimEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);

  EvalResultPtr res;
  // If the arguments contain Any, return Any directly.
  // Only check in TrivialPrimEvaluator, not in TransitionPrimEvaluator.
  const auto standard_prim_eval = dyn_cast_ptr<StandardPrimEvaluator>(shared_from_this());
  bool ignore_any_type_checking =
    (standard_prim_eval != nullptr &&
     ignore_any_type_checking_prims.find(standard_prim_eval->prim()) != ignore_any_type_checking_prims.end());
  if (!ignore_any_type_checking && ContainsAbstractAny(args_abs_list)) {
    MS_LOG(INFO) << ToString() << " receives arguments that contain Any.";
    auto any_abstract = std::make_shared<AbstractAny>();
    const auto &dtype = GetArgsUniqueDtype(args_abs_list);
    if (dtype != nullptr) {
      MS_EXCEPTION_IF_NULL(any_abstract->element());
      any_abstract->element()->set_type(dtype);
      any_abstract->set_supposed_tensor_dtype(true);
    }
    for (const auto &abs : args_abs_list) {
      MS_EXCEPTION_IF_NULL(abs);
      if (abs->isa<abstract::AbstractSequence>()) {
        SetSequenceElementsUseFlagsRecursively(abs, true);
      }
    }
    res = std::make_shared<EvalResult>(any_abstract, std::make_shared<AttrValueMap>());
  } else {
    try {
      res = EvalPrim(engine, args_abs_list);
    } catch (std::exception &e) {
      MS_LOG(ERROR) << "Primitive: <" << ToString() << "> infer failed, failed info: " << e.what();
      std::rethrow_exception(std::current_exception());
    }
    auto prim = standard_prim_eval != nullptr ? standard_prim_eval->prim() : nullptr;
    if (prim != nullptr && fallback::GetJitSyntaxLevel() == kStrict) {
      auto output_abs = res != nullptr ? res->abstract() : nullptr;
      if (opt::ShouldRunWithJitFallback(prim, args_abs_list, output_abs)) {
        std::ostringstream oss;
        for (size_t i = 0; i < args_abs_list.size(); ++i) {
          oss << "Arg[" << i << "]: " << (args_abs_list[i] != nullptr ? args_abs_list[i]->ToString() : "NULL") << "\n";
        }
        MS_EXCEPTION(TypeError)
          << "In JIT strict mode, the primitive '" << prim->name()
          << "' does not support the following argument types. You can use os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2' "
             "to enable the JIT lax mode to support the current syntax. Arguments are:\n"
          << oss.str();
      }
    }
  }
  return res;
}

EvalResultPtr TransitionPrimEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                           const AnfNodeConfigPtr &out_conf) {
  if (args_conf_list.empty() && identifier_ != "MakeTupleEvaluator" && identifier_ != "MakeListEvaluator" &&
      identifier_ != "RaiseEvaluator" && identifier_ != "ConstexprEvaluator") {
    MS_LOG(INTERNAL_EXCEPTION) << "Size should be greater than 0, during running " << identifier_;
  }
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  return EvalPrim(engine, args_abs_list, args_conf_list[0], out_conf);
}

EvalResultPtr SymbolicPrimEvaluator::Run(AnalysisEnginePtr, const ConfigPtrList &args_conf_list,
                                         const AnfNodeConfigPtr &) {
  return EvalPrim(args_conf_list);
}

EvalResultPtr TrackedEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                    const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  EvalResultPtr res = sub_evaluator_->Run(engine, args_conf_list, out_conf);
  // Don't lookup from cache, as different out_conf with same node but different context
  // may add different entry to anfnode_config_map_, like getattr primitive.
  evaluator_cache_mgr_->SetValue(args_abs_list, res);
  return res;
}

EvalResultPtr PartialAppEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                       const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  ConfigPtrList partial_args_conf_list;
  // Join arguments in partial and the rest arguments from args_conf_list.
  (void)std::transform(args_abs_list_.begin(), args_abs_list_.end(), std::back_inserter(partial_args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });

  (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(partial_args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });
  EvalResultPtr res = evaluator_->Run(engine, partial_args_conf_list, out_conf);
  evaluator_cache_mgr_->SetValue(args_abs_list, res);
  return res;
}

EvalResultPtr JEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                              const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, args_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);
  // If the primal func graph's output is sequence, set its elements use flags all true.
  SetSequenceElementsUseFlagsRecursively(result->abstract(), true);
  // Build a virtual function: bprop_f which use sense of y as input, return sense of function free variable and input
  // parameters. (sense_f, sense_x, ...)(*bpro_f) (sense_y)
  AbstractBasePtrList bparams;
  bparams.push_back(SensitivityTransform(primal_func_));
  // Check if primal func graph has the primitive returned sparse result in its bprop().
  auto real_primal_func = dyn_cast_ptr<FuncGraphAbstractClosure>(primal_func_);
  MS_EXCEPTION_IF_NULL(real_primal_func);
  FuncGraphPtr primal_func_graph = real_primal_func->func_graph();
  MS_EXCEPTION_IF_NULL(primal_func_graph);
  bool has_sparse_bprop_prim = primal_func_graph->has_flag(FUNC_GRAPH_FLAG_SPARSE_BPROP);
  (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(bparams),
                       [&has_sparse_bprop_prim](const AbstractBasePtr &arg_abs) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg_abs);
                         if (has_sparse_bprop_prim && arg_abs->isa<AbstractTensor>()) {
                           return std::make_shared<AbstractUndetermined>();
                         }
                         return SensitivityTransform(arg_abs);
                       });
  AbstractBasePtr bparams_final = std::make_shared<AbstractTuple>(bparams);
  AbstractFunctionPtr bprop;
  MS_EXCEPTION_IF_NULL(out_conf);
  auto current_node = out_conf->node();
  MS_EXCEPTION_IF_NULL(current_node);
  if (current_node->isa<CNode>()) {
    bprop = std::make_shared<VirtualAbstractClosure>(SensitivityTransform(result->abstract()), bparams_final);
  } else {
    bprop = std::make_shared<VirtualAbstractClosure>(SensitivityTransform(result->abstract()), bparams_final);
  }

  // J(f)(J(x)) return a tuple (y, bprop_f)
  AbstractBasePtrList jargs = {result->abstract(), bprop};
  AbstractBasePtr jtuple = std::make_shared<AbstractTuple>(jargs);
  auto res = std::make_shared<EvalResult>(jtuple, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_abs_list, res);
  return res;
}

EvalResultPtr TaylorEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                   const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, args_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);
  evaluator_cache_mgr_->SetValue(args_abs_list, result);
  return result;
}

EvalResultPtr ShardEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                  const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, args_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);
  auto res = std::make_shared<EvalResult>(result->abstract(), std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_abs_list, res);
  return res;
}

EvalResultPtr AddAttrEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                    const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_abs_list = EvaluateArguments(args_conf_list);
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, args_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);
  auto res = std::make_shared<EvalResult>(result->abstract(), std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_abs_list, res);
  return res;
}

namespace {
AbstractBasePtr ReduceDim(int *axis, const AbstractBasePtr &orig_abs, int *axis_size) {
  MS_EXCEPTION_IF_NULL(axis);
  MS_EXCEPTION_IF_NULL(orig_abs);
  MS_EXCEPTION_IF_NULL(axis_size);
  if (!orig_abs->isa<AbstractTensor>()) {
    MS_LOG(EXCEPTION) << "The orig_abs should be AbstractTensor when corresponding axis is " << *axis << ", but got a "
                      << orig_abs->ToString() << ". Tip: Please check the correspondence between "
                      << "vmap's 'in_axes' and inputs. You may want to explicitly specify the 'in_axes' "
                      << "corresponding to " << orig_abs->ToString() << " as 'None' to solve this problem.";
  }
  auto orig_abs_shape = dyn_cast_ptr<Shape>(orig_abs->BuildShape());
  MS_EXCEPTION_IF_NULL(orig_abs_shape);
  ShapeVector orig_shape = orig_abs_shape->shape();
  int shape_len = SizeToInt(orig_shape.size());
  if (*axis < -shape_len || *axis >= shape_len) {
    MS_LOG(EXCEPTION) << "The axis: " << *axis << " in 'in_axes' is out of bounds for array of dimension ["
                      << -shape_len << "," << shape_len << ").";
  }
  *axis = *axis < 0 ? shape_len + *axis : *axis;
  auto temp_axes_size = orig_shape[IntToSize(*axis)];
  if (*axis_size == -1) {
    *axis_size = LongToInt(temp_axes_size);
  } else if (*axis_size != temp_axes_size) {
    MS_LOG(EXCEPTION) << "The 'axis_size' of each argument in the scope of 'vmap' should be equal, but got "
                      << *axis_size << " and " << temp_axes_size << ".";
  }
  (void)orig_shape.erase(orig_shape.begin() + *axis);
  BaseShapePtr new_shape = std::make_shared<Shape>(orig_shape);
  MS_EXCEPTION_IF_NULL(orig_abs->Clone());
  AbstractBasePtr abs_clone = orig_abs->Clone()->Broaden();
  abs_clone->set_shape(new_shape);
  return abs_clone;
}

AbstractBasePtr GetLogicalViewAbs(const AbstractBasePtr &physical_view_abs, const ValuePtr &in_axes, int *axis_size) {
  MS_EXCEPTION_IF_NULL(physical_view_abs);
  MS_EXCEPTION_IF_NULL(in_axes);
  auto physical_view_abs_sequence = dyn_cast_ptr<AbstractSequence>(physical_view_abs);
  if (physical_view_abs_sequence != nullptr) {
    AbstractBasePtrList abs_list = physical_view_abs_sequence->elements();
    AbstractBasePtrList logical_view_abs_list;
    auto in_axes_seq = dyn_cast_ptr<ValueSequeue>(in_axes);
    int index = 0;
    (void)std::transform(abs_list.begin(), abs_list.end(), std::back_inserter(logical_view_abs_list),
                         [&axis_size, &index, in_axes_seq, in_axes](const AbstractBasePtr &sub_abs) -> AbstractBasePtr {
                           ValuePtr sub_in_axes = in_axes;
                           if (in_axes->isa<ValueSequeue>()) {
                             sub_in_axes = (*in_axes_seq)[index];
                             index++;
                           }
                           return GetLogicalViewAbs(sub_abs, sub_in_axes, axis_size);
                         });
    if (physical_view_abs->isa<AbstractList>()) {
      return std::make_shared<AbstractList>(logical_view_abs_list, physical_view_abs_sequence->sequence_nodes());
    }
    return std::make_shared<AbstractTuple>(logical_view_abs_list, physical_view_abs_sequence->sequence_nodes());
  }
  ValuePtr in_axis = in_axes;
  if (in_axis->isa<Int64Imm>()) {
    int axis = dyn_cast_ptr<Int64Imm>(in_axis)->value();
    auto logical_view_abs = ReduceDim(&axis, physical_view_abs, axis_size);
    return logical_view_abs;
  }
  if (!in_axis->isa<None>()) {
    MS_LOG(EXCEPTION) << "The axis in vmap's 'in_axes' should be a None or a scalar of type Int64Imm, but got a "
                      << in_axis->ToString() << ".";
  }
  // in_axis is None.
  return physical_view_abs;
}

AbstractBasePtr ExtendDim(int *axis, const AbstractBasePtr &orig_abs, int axis_size) {
  MS_EXCEPTION_IF_NULL(orig_abs);
  MS_EXCEPTION_IF_NULL(axis);
  AbstractBasePtr out_abs = nullptr;
  ShapeVector orig_shape;
  if (orig_abs->isa<AbstractTensor>()) {
    auto shape = dyn_cast_ptr<Shape>(orig_abs->BuildShape());
    if (shape != nullptr) {
      orig_shape = shape->shape();
    }
    if (std::any_of(orig_shape.begin(), orig_shape.end(),
                    [](ShapeValueDType s) { return s == Shape::kShapeRankAny; })) {
      return orig_abs;
    }
  }
  int shape_len = SizeToInt(orig_shape.size() + 1);
  if (*axis < -shape_len || *axis >= shape_len) {
    MS_LOG(EXCEPTION) << "The axis: " << *axis << " in 'out_axes' is out of bounds for array of dimension ["
                      << -shape_len << "," << shape_len << ").";
  }
  *axis = *axis < 0 ? shape_len + *axis : *axis;
  (void)orig_shape.insert(orig_shape.begin() + *axis, axis_size);
  BaseShapePtr new_shape = std::make_shared<Shape>(orig_shape);
  if (orig_abs->isa<AbstractTensor>()) {
    auto tmp_abs = orig_abs->Clone();
    MS_EXCEPTION_IF_NULL(tmp_abs);
    out_abs = tmp_abs->Broaden();
    MS_EXCEPTION_IF_NULL(out_abs);
    out_abs->set_shape(new_shape);
  } else if (orig_abs->isa<AbstractScalar>()) {
    out_abs = std::make_shared<AbstractTensor>(orig_abs, new_shape);
  } else {
    MS_LOG(EXCEPTION) << "The outputs of vmap's 'fn' should be consisting of tensors or constants, but got "
                      << orig_abs->ToString() << ".";
  }
  return out_abs;
}

AbstractBasePtr GetPhysicalViewAbs(const AbstractBasePtr &logical_view_abs, const ValuePtr &out_axes, int axis_size) {
  MS_EXCEPTION_IF_NULL(logical_view_abs);
  MS_EXCEPTION_IF_NULL(out_axes);
  auto logical_view_abs_sequence = dyn_cast_ptr<AbstractSequence>(logical_view_abs);
  if (logical_view_abs_sequence != nullptr) {
    AbstractBasePtrList logical_view_abs_list = logical_view_abs_sequence->elements();
    AbstractBasePtrList physical_view_abs_list;
    auto out_axes_seq = dyn_cast_ptr<ValueSequeue>(out_axes);
    if (out_axes_seq != nullptr) {
      if (logical_view_abs_list.size() != out_axes_seq->size()) {
        MS_LOG(EXCEPTION) << "The size of vmap's 'out_axes' should be equal to the number of results of 'fn': "
                          << logical_view_abs_list.size() << ", but got size: " << out_axes_seq->size() << ".";
      }
    }
    int index = 0;
    (void)std::transform(
      logical_view_abs_list.begin(), logical_view_abs_list.end(), std::back_inserter(physical_view_abs_list),
      [&axis_size, &index, out_axes_seq, out_axes](const AbstractBasePtr &arg_abs) -> AbstractBasePtr {
        ValuePtr sub_out_axes = out_axes;
        if (out_axes->isa<ValueSequeue>()) {
          sub_out_axes = (*out_axes_seq)[index];
          index++;
        }
        if (arg_abs->isa<AbstractSequence>()) {
          return GetPhysicalViewAbs(arg_abs, sub_out_axes, axis_size);
        }
        if (sub_out_axes->isa<Int64Imm>()) {
          int axis = static_cast<int>(dyn_cast_ptr<Int64Imm>(sub_out_axes)->value());
          return ExtendDim(&axis, arg_abs, axis_size);
        } else if (sub_out_axes->isa<None>()) {
          return arg_abs;
        }
        MS_LOG(EXCEPTION) << "The axis in vmap's 'out_axes' should be a None or a scalar of type Int64Imm, but got a "
                          << sub_out_axes->ToString() << ".";
      });
    auto ret = logical_view_abs->Clone();
    dyn_cast_ptr<AbstractSequence>(ret)->set_elements(physical_view_abs_list);
    return ret;
  }

  // for the single output case, outputs: A, and out_axes: 1 or (1,).
  ValuePtr sub_out_axes = out_axes;
  ValueSequeuePtr out_axes_seq = dyn_cast<ValueSequeue>(out_axes);
  if (out_axes_seq != nullptr) {
    if (out_axes_seq->size() != 1) {
      MS_LOG(EXCEPTION) << "The size of vmap's 'out_axes' should be equal to the result size: 1, but got size: "
                        << out_axes_seq->size() << ".";
    }
    sub_out_axes = (*out_axes_seq)[0];
  }

  int axis = 0;
  auto axis_int_ptr = dyn_cast_ptr<Int64Imm>(sub_out_axes);
  if (axis_int_ptr != nullptr) {
    axis = LongToInt(axis_int_ptr->value());
  } else {
    MS_LOG(EXCEPTION) << "The axis in vmap's 'out_axes' should be a None or a scalar of type Int64Imm, but got a "
                      << sub_out_axes->ToString() << ".";
  }
  return ExtendDim(&axis, logical_view_abs, axis_size);
}
}  // namespace

// According to the in_axes (e.g. (1,(None,3))), the abstraction of input parameters with the
// physical view (e.g. (A,(B,C))) are converted into that with the logical view (e.g.(a,(b,c))),
// more specific, the input `A` with shape (32, 16, 8) fitting the axis index `1` is converted in to
// `a` with shape (32, 8). And then leverage the original graph to perform the evaluation.
// Finally, the outputs with the logical view are converted back into the physical view in
// combination with the out_axes. The inferring result is consistent with that after eliminating
// the VmapOperator.
EvalResultPtr VmapEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                 const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_abs_list;
  int axis_size = -1;
  int index = 0;
  auto in_axes = in_axes_;
  auto in_axes_seq = dyn_cast_ptr<ValueSequeue>(in_axes);
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_abs_list),
                       [&axis_size, &index, in_axes_seq, in_axes](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         AbstractBasePtr abs = conf->ObtainEvalResult()->abstract();
                         MS_EXCEPTION_IF_NULL(abs);
                         // Drop the side effect tag parameters, because it has no mapping axis.
                         // e.g. args=(A,(B,C),U), in_axes=(1,(None,3))
                         if (abs->isa<AbstractMonad>()) {
                           return abs;
                         }
                         ValuePtr sub_in_axes = in_axes;
                         MS_EXCEPTION_IF_NULL(in_axes);
                         if (in_axes->isa<ValueSequeue>()) {
                           sub_in_axes = (*in_axes_seq)[index];
                           index++;
                         }
                         auto arg_abs = GetLogicalViewAbs(abs, sub_in_axes, &axis_size);
                         return arg_abs;
                       });
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
  if (eval_result != nullptr) {
    return eval_result;
  }
  ConfigPtrList virtual_conf_list;
  (void)std::transform(args_abs_list.begin(), args_abs_list.end(), std::back_inserter(virtual_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, virtual_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);

  // If the primal func graph's output is sequence, set its elements use flags all true.
  SetSequenceElementsUseFlagsRecursively(result->abstract(), true);

  if (axis_size == -1 && cell_size_ != 0) {
    axis_size = SizeToInt(cell_size_);
  } else if (axis_size != -1 && cell_size_ != 0 && axis_size != SizeToInt(cell_size_)) {
    MS_EXCEPTION(ValueError) << "If you want to execute the model ensembling parallel training, please make sure "
                             << "the 'axis_size' in the scope of vmap consistent with the cell size of the input "
                             << "'CellList', otherwise, please do not enter 'CellList' as the first argument, "
                             << "but we get axis_size: " << axis_size << " and the cell size: " << cell_size_ << ".";
  }

  AbstractBasePtr result_abs = result->abstract();
  AbstractBasePtr after_vmap = GetPhysicalViewAbs(result_abs, out_axes_, axis_size);

  auto res = std::make_shared<EvalResult>(after_vmap, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_abs_list, res);
  return res;
}

EvalResultPtr VirtualEvaluator::Eval(AnalysisEnginePtr, const AbstractBasePtrList &args_abs_list,
                                     const AnfNodeConfigPtr &out_conf) {
  if (args_abs_list.size() != args_abs_list_.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Arguments mismatch, parameters no: " << args_abs_list_.size()
                               << ", arguments no: " << args_abs_list.size();
  }
  const auto sense_param_index = args_abs_list.size() - 1;
  bool sense_param_flag = false;
  MS_EXCEPTION_IF_NULL(this->bound_node());
  if (this->bound_node()->isa<CNode>()) {
    sense_param_flag = this->bound_node()->cast<CNodePtr>()->HasAttr("sens_param_");
  }
  static const auto enable_eliminate_unused_element = (common::GetCompileConfig("ENABLE_DDE") != "0");
  // Check each parameter and argument match;
  for (std::size_t i = 0; i < args_abs_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(args_abs_list[i]);
    // For VirtualAbstractClosure, likely J's bprop, we just set its tuple arguments as used before really grad.
    if (enable_eliminate_unused_element && args_abs_list[i]->isa<AbstractSequence>()) {
      MS_LOG(INFO) << "Notice: For VirtualAbstractClosure, update all use flags as true for arguments[" << i
                   << "]: " << args_abs_list[i]->ToString();
      SetSequenceElementsUseFlagsRecursively(args_abs_list[i], true);
    }
    if (i == sense_param_index && sense_param_flag) {
      const auto &sense_shape = args_abs_list[i]->BuildShape();
      MS_EXCEPTION_IF_NULL(sense_shape);
      if (sense_shape->IsDynamic()) {
        MS_EXCEPTION(ValueError) << "The shape of sense must not be dynamic shape."
                                 << "\nFor more details with 'sense', please refer to "
                                 << "https://www.mindspore.cn/docs/zh-CN/master/faq/network_compilation.html.";
      }
    }
    (void)args_abs_list[i]->Join(args_abs_list_[i]);
  }
  return std::make_shared<EvalResult>(output_, std::make_shared<AttrValueMap>());
}

EvalResultPtr Evaluator::SingleRun(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                   const AnfNodeConfigPtr &out_conf) {
  EvalResultPtr result;
  try {
    result = this->Run(engine, args_conf_list, out_conf);
  } catch (const std::exception &ex) {
    MS_LOG(INFO) << "Eval " << ToString() << " throw exception.";
    AnalysisSchedule::GetInstance().HandleException(ex);
  }
  AnalysisSchedule::GetInstance().Wait();
  return result;
}
}  // namespace abstract
}  // namespace mindspore
