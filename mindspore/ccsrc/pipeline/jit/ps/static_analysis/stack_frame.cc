/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/static_analysis/stack_frame.h"
#include "pipeline/jit/ps/debug/trace.h"
#include "pipeline/jit/ps/static_analysis/async_eval_result.h"
#include "utils/compile_config.h"

namespace mindspore {
namespace abstract {
AbstractBasePtrList StackFrame::GenerateArgsAbsList(const AnalysisEnginePtr &engine, const EvaluatorPtr &evaluator,
                                                    const CNodePtr &current_cnode) {
  MS_EXCEPTION_IF_NULL(current_cnode);
  MS_EXCEPTION_IF_NULL(evaluator);
  AbstractBasePtrList args_abs_list;
  for (std::size_t i = 1; i < current_cnode->size(); i++) {
    auto config = engine->MakeConfig(current_cnode->input(i), current_context_, current_context_->func_graph());
    auto result = config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(result);
    auto abs = result->abstract();
    args_abs_list.push_back(abs);
  }
  args_abs_list = evaluator->NormalizeArgs(args_abs_list);
  args_abs_list = evaluator->BroadenUndeterminedArgs(args_abs_list, engine);
  return args_abs_list;
}

AnalysisContextPtr StackFrame::GetParentContext(const BaseFuncGraphEvaluatorPtr &fg_evaluator,
                                                const AbstractFunctionPtr &graph_func) const {
  MS_EXCEPTION_IF_NULL(graph_func);
  MS_EXCEPTION_IF_NULL(fg_evaluator);
  AnalysisContextPtr parent_context = nullptr;
  auto func_graph_abs = dyn_cast_ptr<FuncGraphAbstractClosure>(graph_func);
  if (func_graph_abs != nullptr) {  // Set parent context for FuncGraphAbstractClosure.
    parent_context = func_graph_abs->context();
  } else if (graph_func->isa<MetaFuncGraphAbstractClosure>()) {  // Or DummyContext for MetaFuncGraphAbstractClosure.
    parent_context = fg_evaluator->parent_context();
    if (parent_context == nullptr) {
      parent_context = AnalysisContext::DummyContext();
      fg_evaluator->set_parent_context(parent_context);
    }
  } else {  // Not call FuncGraph or MetaFuncGraph.
    MS_LOG(INTERNAL_EXCEPTION) << "Should be FuncGraphAbstractClosure or MetaFuncGraphAbstractClosure.";
  }
  return parent_context;
}

// Inner jump implementation.
StackFramePtr StackFrame::DoJump(const AnalysisEnginePtr &engine, const CNodePtr &current_cnode,
                                 const AbstractFunctionPtr &graph_func) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(current_cnode);
  // Get the evaluator for func graph.
  auto evaluator = engine->GetEvaluatorFor(graph_func);
  auto fg_evaluator = dyn_cast<BaseFuncGraphEvaluator>(evaluator);
  if (fg_evaluator == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Evaluator should be a BaseGraphEvaluator, but got " << evaluator->ToString();
  }
  fg_evaluator->set_bound_node(current_cnode);

  // Evaluate the inputs firstly. Build arguments for the func graph.
  AbstractBasePtrList args_abs_list = GenerateArgsAbsList(engine, evaluator, current_cnode);
  // Check if already evaluated before.
  MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
  auto &cache = evaluator->evaluator_cache_mgr()->GetCache();
  auto iter = cache.find(args_abs_list);
  if (iter != cache.end()) {
    MS_EXCEPTION_IF_NULL(current_context_);
    MS_LOG(DEBUG) << "Eval before, current_node: " << current_cnode->DebugString()
                  << ", current_context_: " << current_context_->ToString() << ", args: " << args_abs_list;
    for (size_t i = 0; i < args_abs_list.size(); ++i) {
      const auto &old_arg = iter->first[i];
      const auto &new_arg = args_abs_list[i];
      // Update inputs abstract, if matched in cache.
      SynchronizeSuccessiveInputs(old_arg, new_arg);
    }
    return nullptr;
  }

  // Generate func graph with arguments.
  auto fg = fg_evaluator->GetFuncGraph(engine, args_abs_list);
  MS_EXCEPTION_IF_NULL(fg);
  std::size_t nargs = fg->parameters().size();
  if (args_abs_list.size() != nargs) {
    MS_EXCEPTION(TypeError) << "The parameters number of the function is " << fg->parameters().size()
                            << ", but the number of provided arguments is " << args_abs_list.size() << ".\n"
                            << "FunctionGraph ID : " << fg->ToString()
                            << "\nNodeInfo: " << trace::GetDebugInfoStr(fg->debug_info());
  }
  MS_LOG(DEBUG) << "Not eval before, current_node: " << current_cnode->DebugString() << ", fg: " << fg->ToString()
                << ", current_context_: " << current_context_->ToString() << ", args: " << args_abs_list;

  // Find parent context and create new context.
  AnalysisContextPtr parent_context = GetParentContext(fg_evaluator, graph_func);
  MS_EXCEPTION_IF_NULL(parent_context);
  auto new_context = NewContext(parent_context, fg, args_abs_list);

  bool always_eval_flag = false;
  // Evaluate the parameters with new context.
  for (size_t i = 0; i < nargs; i++) {
    const auto &arg_abs = args_abs_list[i];
    const auto &node = fg->parameters()[i];
    AnfNodeConfigPtr conf = engine->MakeConfig(node, new_context, new_context->func_graph());
    always_eval_flag = always_eval_flag || CheckIfAlwaysEval(conf, arg_abs);
    auto result = std::make_shared<EvalResult>(arg_abs, nullptr);
    MS_LOG(DEBUG) << "Save argument[" << i << "] result, NodeConfig: " << conf->ToString()
                  << ", result: " << result->abstract().get() << "/" << result->abstract()->ToString();
    engine->SaveEvalResultInCache(conf, result);
  }
  fg_evaluator->PushAlwaysEvalFlag(always_eval_flag);
  fg_evaluator->SyncFuncGraphSideEffectFlag(fg);
  // Create a new stack frame and set arguments for it.
  auto new_stack_frame = std::make_shared<StackFrame>(fg_evaluator, fg, new_context, parent_context);
  new_stack_frame->set_args_abs_list(std::move(args_abs_list));
  return new_stack_frame;
}

// Check if we need branch to another func graph.
StackFramePtr StackFrame::Jump(const AnalysisEnginePtr &engine) {
  MS_EXCEPTION_IF_NULL(engine);
  auto &current_node = CurrentNode();
  if (!current_node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = current_node->cast<CNodePtr>();
  auto maybe_func = engine->GetCNodeOperatorAbstract(cnode, current_context_, current_context_->func_graph());
  if (!maybe_func->isa<abstract::MetaFuncGraphAbstractClosure>() &&
      !maybe_func->isa<abstract::FuncGraphAbstractClosure>()) {
    return nullptr;  // Not call FuncGraph or MetaFuncGraph.
  }

  // It's FuncGraph Call or MetaFuncGraph Call. `maybe_func` is definitely a AbstractFunction.
  AnfNodeConfigPtr call_node_conf = engine->MakeConfig(cnode, current_context_, current_context_->func_graph());
  MS_EXCEPTION_IF_NULL(call_node_conf);
  // Note: Because of kFuncGraphFlagUndetermined flag, call to the same funcgraph with same arguments may not
  // be idempotent as those arguments may be broadened in the second call, so just do the jump when necessary.
  auto fg_evaluator = dyn_cast_ptr<BaseFuncGraphEvaluator>(evaluator());
  if (fg_evaluator == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Evaluator should be a BaseGraphEvaluator, but got " << evaluator()->ToString();
  }
  if (!fg_evaluator->always_eval_flag()) {
    MS_LOG(DEBUG) << "Check if CNode had been evaluated, cnode: " << cnode->DebugString();
    const auto &node_eval_result = ObtainEvalResultFromCache(call_node_conf);
    if (node_eval_result != nullptr) {
      const auto &abstract = node_eval_result->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      MS_LOG(DEBUG) << "No need to jump as found result from cache for node_config: " << call_node_conf->ToString()
                    << ", result: " << abstract->ToString();
      const auto &abs_func_graph = maybe_func->cast<AbstractFunctionPtr>();
      SynchronizeSequenceElementsUseFlagsForFuncGraphArgs(engine, current_context_->func_graph(), cnode, abs_func_graph,
                                                          current_context_);
      return nullptr;
    }
  }
  // Enter the call CNode.
  trace::TraceEvalCNodeEnter(call_node_conf);
  auto res = DoJump(engine, cnode, dyn_cast<AbstractFunction>(maybe_func));
  if (res == nullptr) {
    trace::TraceEvalCNodeLeave();
  }
  return res;
}

// Run one step in current func graph.
EvalResultPtr StackFrame::Step(const AnalysisEnginePtr &engine) {
  MS_EXCEPTION_IF_NULL(engine);
  auto &current_node = NextNode();
  MS_LOG(DEBUG) << "current_node: " << current_node->DebugString()
                << ", current_context_: " << current_context_->ToString();
  auto current_context_fg = current_context_->func_graph();
  if (current_context_fg == nullptr && common::GetCompileConfig("STRICT_CHECK_PARENT_CONTEXT") != "1") {
    current_context_fg = engine->root_func_graph_backup();
  }
  AnfNodeConfigPtr node_conf = engine->MakeConfig(current_node, current_context_, current_context_fg);
  EvalResultPtr node_eval_result = nullptr;
  auto fg_evaluator = dyn_cast_ptr<BaseFuncGraphEvaluator>(evaluator());
  if (fg_evaluator == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Evaluator should be a BaseGraphEvaluator, but got " << evaluator()->ToString();
  }
  if (fg_evaluator->always_eval_flag()) {
    MS_LOG(DEBUG) << "Always eval node";
    node_eval_result = engine->ObtainEvalResultWithoutCache(node_conf);
  } else {
    node_eval_result = engine->ObtainEvalResultWithCache(node_conf);
    if (engine->check_side_effect()) {
      MS_EXCEPTION_IF_NULL(node_eval_result);
      if (node_eval_result->has_side_effect_node()) {
        auto cnode = dyn_cast_ptr<CNode>(current_node);
        MS_EXCEPTION_IF_NULL(cnode);
        MS_LOG(DEBUG) << "Found side-effect, cnode: " << cnode->DebugString()
                      << ", func_graph: " << node_conf->func_graph()->ToString();
        cnode->set_has_side_effect_node(true);
        current_context_->func_graph()->set_has_side_effect_node(true);
      }
    }
  }
  MS_LOG(DEBUG) << GetInferThread() << "Eval(" << node_conf->ToString() << ") = "
                << (node_eval_result->abstract() ? node_eval_result->abstract()->ToString() : "Abstract null");
  return node_eval_result;
}

// Return back from child func graph.
void StackFrame::Back(const AnalysisEnginePtr &engine, const StackFramePtr &last_stack_frame,
                      const EvalResultPtr &eval_result) {
  MS_EXCEPTION_IF_NULL(engine);
  MS_EXCEPTION_IF_NULL(last_stack_frame);
  MS_EXCEPTION_IF_NULL(eval_result);
  // Overwrite the result if func graph is stub.
  EvalResultPtr result = eval_result;
  MS_EXCEPTION_IF_NULL(last_stack_frame->func_graph());
  if (last_stack_frame->func_graph()->stub()) {
    result = std::make_shared<EvalResult>(std::make_shared<AbstractUndetermined>(), nullptr);
  }

  // Check if callee func graph contains side-effect.
  if (engine->check_side_effect()) {
    auto cnode = dyn_cast_ptr<CNode>(CurrentNode());
    MS_EXCEPTION_IF_NULL(cnode);
    MS_EXCEPTION_IF_NULL(func_graph());
    if (last_stack_frame->func_graph()->has_side_effect_node()) {
      MS_LOG(DEBUG) << "Found side-effect, cnode: " << cnode->DebugString()
                    << ", func_graph: " << func_graph()->ToString();
      cnode->set_has_side_effect_node(true);
      func_graph()->set_has_side_effect_node(true);
    } else {
      // Check inputs side-effect.
      for (std::size_t i = 1; i < cnode->size(); ++i) {
        auto input_cnode = dyn_cast_ptr<CNode>(cnode->input(i));
        if (input_cnode != nullptr && input_cnode->has_side_effect_node()) {
          MS_LOG(DEBUG) << "Found side-effect, cnode: " << cnode->DebugString()
                        << ", func_graph: " << func_graph()->ToString();
          cnode->set_has_side_effect_node(true);
          func_graph()->set_has_side_effect_node(true);
          break;
        }
      }
    }
  }

  // Save func graph eval result for specialize.
  auto evaluator = last_stack_frame->evaluator();
  MS_EXCEPTION_IF_NULL(evaluator);
  MS_EXCEPTION_IF_NULL(evaluator->evaluator_cache_mgr());
  evaluator->evaluator_cache_mgr()->SetValue(last_stack_frame->args_abs_list(), result);
  auto fg_evaluator = dyn_cast_ptr<BaseFuncGraphEvaluator>(evaluator);
  if (fg_evaluator == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Evaluator should be a BaseGraphEvaluator, but got " << evaluator->ToString();
  }
  fg_evaluator->PopAlwaysEvalFlag();

  // Continue saving node's result for parent func graph.
  auto &current_node = NextNode();
  AnfNodeConfigPtr node_conf = engine->MakeConfig(current_node, current_context_, current_context_->func_graph());
  MS_LOG(DEBUG) << "current_node: " << current_node->DebugString()
                << ", current_context_: " << current_context_->ToString()
                << ", Save result, NodeConfig: " << node_conf->ToString() << ", result: " << result->abstract().get()
                << "/" << result->abstract()->ToString();
  engine->SaveEvalResultInCache(node_conf, result);
  // Leave the call CNode.
  trace::TraceEvalCNodeLeave();
}
}  // namespace abstract
}  // namespace mindspore
