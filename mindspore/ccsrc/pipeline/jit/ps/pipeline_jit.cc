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

#include "pipeline/jit/ps/pipeline_jit.h"
#include <vector>
#include <utility>
#include "pipeline/jit/ps/pass.h"
#include "pipeline/jit/ps/event_message_print.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "utils/phase.h"
#include "include/common/fallback.h"
#include "include/common/utils/compile_cache_context.h"
#include "include/backend/distributed/recovery/recovery_context.h"

namespace mindspore {
namespace pipeline {

JitExecutorPyPtr JitExecutorPy::executor_ = nullptr;
std::mutex JitExecutorPy::instance_lock_;

namespace {
void CacheFuncGraph(const ResourcePtr &resource) {
  if (!resource->EnableCompileCache()) {
    return;
  }
  {
    MsProfileStatGuard stat_guard("SaveCacheFuncGraph", "compile_cache", true);
    resource->CacheFuncGraph();
  }
}

void Optimize(const ResourcePtr &resource, const std::vector<PassItem> &passes) {
  MS_EXCEPTION_IF_NULL(resource);
  bool already_print_profile = false;
  ProfileExecute(MsProfile::GetProfile(), [&resource, &passes, &already_print_profile]() {
    static const std::string last_compile_action = kValidate;
    static const auto compile_profile_finish_action = common::GetCompileConfig("COMPILE_PROFILE_FINISH_ACTION");
    size_t counter = 0;
    for (auto &pass : passes) {
      std::string pass_name = pass.first;
      MsProfileStatGuard stat_guard(std::move(pass_name), "compile_irpass", true);
      ProcessStatus::GetInstance().RecordStart(pass.first);
      auto profile_context = MsProfile::GetProfile()->Step(pass.first);
      auto pass_func = [&pass, &resource, &counter]() {
        MS_LOG(INFO) << "Pass " << pass.first << " start ...";
        auto result = pass.second(resource);
        if (!result) {
          MS_LOG(INTERNAL_EXCEPTION) << "Pass running to end, failed in pass:" << pass.first;
        }
#ifdef ENABLE_DUMP_IR
        auto context = MsContext::GetInstance();
        MS_EXCEPTION_IF_NULL(context);
        if (context->CanDump(kIntroductory) && resource->func_graph() != nullptr) {
          std::string base_name = GetBaseNameForIR(SizeToLong(counter), pass.first);
          auto func_graph = resource->func_graph();
          MS_EXCEPTION_IF_NULL(func_graph);
          DumpIR(base_name + ".ir", func_graph, true, kWholeStack);
          MS_LOG(DEBUG) << "Dump " << base_name << " func graph.";
        }
#endif
        counter++;
        MS_LOG(INFO) << "Pass " << pass.first << " end.";
      };
      ProfileExecute(profile_context, pass_func);
      ProcessStatus::GetInstance().RecordEnd();
      if (pass.first == kTaskEmit) {
        SetLoopCount(resource);
      } else if (pass.first == last_compile_action) {
        CheckInterpretNodeLineInfos();
        CacheFuncGraph(resource);
        ResetId(resource);
      } else if (pass.first == kAutoMonadReorder) {
        resource->set_optimize_graph(resource->func_graph());
      }

      if (EnabledProfile() && compile_profile_finish_action == pass.first) {
        ProfileExecuteBreak(MsProfile::GetProfile());
        MsProfile::Print();
        already_print_profile = true;
      }
    }
  });

  if (EnabledProfile()) {
    if (!already_print_profile) {
      MsProfile::Print();
    }
    MsProfile::Reset();
  }
}

std::vector<PassItem> GetJitPasses(const ResourcePtr &resource, bool build_top_graph = true) {
  std::vector<PassItem> jit_passes;
  if (!resource->EnableCompileCache() || resource->func_graph() == nullptr) {
    // Compile the frontend graph.
    if (build_top_graph) {
      (void)jit_passes.emplace_back(kBootstrap, BootstrapAction);
    }
    (void)jit_passes.emplace_back(kTypeInference, TypeInferenceAction);
    (void)jit_passes.emplace_back(kAutoMonad, AutoMonadAction);
    (void)jit_passes.emplace_back(kGraphReusing, GraphReusingAction);
    (void)jit_passes.emplace_back(kPreAutoParallel, SetTrainingFlagPass);
    (void)jit_passes.emplace_back(kPyInterpretToExecute, PyInterpretToExecutePass);
    (void)jit_passes.emplace_back(kRewriterBeforeOptA, RewriterBeforeOptAPass);
    (void)jit_passes.emplace_back(kExpandDumpFlag, ExpandDumpFlagPass);
    (void)jit_passes.emplace_back(kJitOptA, JitOptPassAGroup);
    (void)jit_passes.emplace_back(kPyInterpretToExecuteAfterOptA, PyInterpretToExecutePass);
    (void)jit_passes.emplace_back(kRewriterAfterOptA, RewriterAfterOptAPass);
    (void)jit_passes.emplace_back(kConvertAfterRewriter, ConvertAfterRewriterPass);
    (void)jit_passes.emplace_back(kOrderPyExecuteAfterRewriter, OrderPyExecuteAfterRewriterPass);
    (void)jit_passes.emplace_back(kJitOptB, JitOptPassBGroup);
    (void)jit_passes.emplace_back(kCconv, CconvPass);
    (void)jit_passes.emplace_back(kLoopUnroll, LoopUnrollPass);
    (void)jit_passes.emplace_back(kJitOptPassAfterCconv, JitOptPassAfterCconvGroup);
    (void)jit_passes.emplace_back(kRemoveDupValue, RemoveValueNodeDuplicationsPassForJit);
    (void)jit_passes.emplace_back(kPartialUnusedArgsEliminate, PartialUnusedArgsEliminatePass);
    (void)jit_passes.emplace_back(kEnvironConv, EnvironConversionPass);
    (void)jit_passes.emplace_back(kAddRecomputation, AddRecomputationPass);
    (void)jit_passes.emplace_back(kCseAfterRecomputation, OptAfterRecomputeGroup);
    (void)jit_passes.emplace_back(kAutoMonadReorder, OrderEnforceAction);
    (void)jit_passes.emplace_back(kGetJitBpropGraph, GetJitBpropGraph);
    (void)jit_passes.emplace_back(kRewriterAfterJitBprop, RewriterAfterOptAPassAfterJitBprop);
    (void)jit_passes.emplace_back(kEliminateSpecialOpNode, EliminateSpecialOpNode);
    (void)jit_passes.emplace_back(kValidate, ValidatePass);
  }

  auto is_precompile_only = MsContext::GetInstance()->get_param<bool>(MS_CTX_PRECOMPILE_ONLY);
  if (is_precompile_only) {
    MS_LOG(INFO) << "PrecompileOnly, stop run graph";
    return jit_passes;
  }

  if (common::GetEnv(kSimulationLevel) == kSimulationLevelCompileGraph) {
    return jit_passes;
  }

#ifndef WITH_BACKEND
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() != "ge") {
#endif
    // Compile the backend graph.
    (void)jit_passes.emplace_back(kTaskEmit, TaskEmitAction);

    // Execute the graph.
    (void)jit_passes.emplace_back(kExecute, ExecuteAction);
#ifndef WITH_BACKEND
  }
#endif

  return jit_passes;
}

void DoOptimize(const ResourcePtr &resource, bool build_top_graph = true) {
  compile::SetMindRTEnable();
  std::vector<PassItem> jit_passes = GetJitPasses(resource, build_top_graph);
  if (std::any_of(jit_passes.cbegin(), jit_passes.cend(),
                  [](const PassItem &action) { return action.first == kTaskEmit || action.first == kExecute; })) {
    // Create backend asynchronously.
    resource->SetBackendAsync([]() {
      auto backend = compile::CreateBackend();
      return backend;
    });
  }
  Optimize(resource, jit_passes);
}
}  // namespace

bool JitExecutorPy::SetSource(const py::object &source) {
  // Check if the function or net is valid.
  if (py::isinstance<py::none>(source)) {
    MS_LOG(ERROR) << "The source object to compile should not be None.";
    return false;
  }
  source_ = py::cast<std::string>(py::str(source));
  return true;
}

bool JitExecutorPy::SetPhase(const py::object &phase) {
  // Check if the phase is valid.
  if ((!py::isinstance<py::str>(phase))) {
    MS_LOG(ERROR) << "The `phase` must be string.";
    return false;
  }
  phase_ = py::cast<std::string>(phase);
  PhaseManager::GetInstance().set_phase(phase_);
  return true;
}

bool JitExecutorPy::CompileInner(const py::object &source, const py::tuple &args, const py::dict &kwargs,
                                 const py::object &phase) {
  JitCompilingScope jit_compiling_scope;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->SetCellReuseLevel(CellReuseLevel::kNoCellReuse);
  if (!SetSource(source) || !SetPhase(phase)) {
    return false;
  }
  obj_desc_ = GetObjDesc(source);
  MS_LOG(INFO) << "Start compiling, phase: " << phase_;
  MS_LOG(DEBUG) << "source: {" << source_ << "}\nargs: " << py::str(const_cast<py::tuple &>(args))
                << "\nkwargs: " << py::str(const_cast<py::dict &>(kwargs));
  EventMessage::PrintCompileStartMsg(phase_, obj_desc_);

  ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
  ResourcePtr resource = std::make_shared<Resource>(source);
  resource->set_pipeline_level(pipeline::kLevelJit);
  executor_info->resource = resource;
  InitCompileCacheResource(resource, phase_);
  // Get the parameters items and add the value to args_abs.
  ConvertArgs(args, kwargs, resource, executor_info);
  info_[phase_] = executor_info;
  DoOptimize(resource);
  // Save the compiled graph to MsPipeLine.
  SaveCompiledGraph(phase_);
  CleanCompileRes(resource);
  EventMessage::PrintCompileEndMsg(phase_, obj_desc_);
  PhaseManager::GetInstance().ClearPhase();
  MS_LOG(INFO) << "Finish compiling.";
  return true;
}

bool JitExecutorPy::CompileInner(const FuncGraphPtr &graph, const py::tuple &args, const py::dict &kwargs,
                                 const std::string &phase, bool) {
  JitRunningScope jit_running_scope;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  ms_context->SetCellReuseLevel(CellReuseLevel::kNoCellReuse);
  PhaseManager::GetInstance().set_phase(phase);
  phase_ = phase;

  ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
  ResourcePtr resource = std::make_shared<Resource>();
  resource->set_func_graph(graph);
  resource->set_pipeline_level(pipeline::kLevelJit);
  executor_info->resource = resource;
  InitCompileCacheResource(resource, phase_);
  // Get the parameters items and add the value to args_abs.
  ConvertArgs(args, kwargs, resource, executor_info);
  info_[phase] = executor_info;
  DoOptimize(resource, false);
  // Save the compiled graph to MsPipeLine.
  SaveCompiledGraph(phase_);
  CleanCompileRes(resource);
  PhaseManager::GetInstance().ClearPhase();
  MS_LOG(INFO) << "Finish compiling.";
  return true;
}

void JitExecutorPy::DelOneNetRes(const py::handle &py_phase) {
  if (!pybind11::isinstance<py::str>(py_phase)) {
    MS_LOG(ERROR) << "Expect string phase, but got " << py::str(py_phase);
    return;
  }
  auto phase = pybind11::cast<std::string>(py_phase);
  MS_LOG(INFO) << "Delete one net resource start, phase: " << phase;
  auto iter = info_.find(phase);
  auto clear = false;
  if (iter != info_.end()) {
    clear = true;
    (void)info_.erase(phase);
    MS_LOG(DEBUG) << "Delete phase: " << phase << ", info size: " << info_.size();
  }
  if (clear) {
    // Do clear here to avoid any pointer for resource.
    FuncGraphLoopBreaker::Inst().ClearCellGraphs(phase);
    FuncGraphLoopBreaker::Inst().CleanUnusedFuncGraphs(phase);
  }
  MS_LOG(INFO) << "Delete one net resource end. " << clear;
}

void JitExecutorPy::ConvertArgs(const py::tuple &args, const py::dict &kwargs, const ResourcePtr &resource,
                                const ExecutorInfoPtr &executor_info) {
  abstract::AbstractBasePtrList args_abs;
  std::vector<ValuePtr> arguments;
  for (std::size_t i = 0; i < args.size(); i++) {
    auto iter = cur_convert_input_.find(args[i].ptr());
    if (iter != cur_convert_input_.end()) {
      (void)arguments.emplace_back(iter->second.first);
      (void)args_abs.emplace_back(iter->second.second);
      SetHookForArgAbstract(args[i], iter->second.second);
      continue;
    }
    ValuePtr converted = nullptr;
    bool success = parse::ConvertData(args[i], &converted);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert the " << i << "th argument, args[" << i
                                 << "]: " << py::str(args[i]);
    }
    (void)arguments.emplace_back(converted);
    auto args_abstract_item = ArgsToAbstract(args[i], converted, enable_tuple_broaden_);
    (void)args_abs.emplace_back(args_abstract_item);
    SetHookForArgAbstract(args[i], args_abstract_item);
  }
  for (const auto &item : kwargs) {
    auto iter = cur_convert_input_.find(item.first.ptr());
    if (iter != cur_convert_input_.end()) {
      (void)arguments.emplace_back(iter->second.first);
      (void)args_abs.emplace_back(iter->second.second);
      auto keyword_arg_abs = iter->second.second->cast<abstract::AbstractKeywordArgPtr>();
      SetHookForArgAbstract(py::cast<py::object>(item.second), keyword_arg_abs->get_arg());
      continue;
    }
    ValuePtr key = nullptr;
    ValuePtr value = nullptr;
    bool success = parse::ConvertData(py::cast<py::object>(item.first), &key) &&
                   parse::ConvertData(py::cast<py::object>(item.second), &value);
    if (!success) {
      MS_LOG(INTERNAL_EXCEPTION) << "Fail to convert the argument (" << py::str(item.first) << ": "
                                 << py::str(item.second) << ").";
    }
    AbstractBasePtr value_abs = ArgsToAbstract(py::cast<py::object>(item.second), value, enable_tuple_broaden_);
    auto keyword_arg_abs = std::make_shared<abstract::AbstractKeywordArg>(GetValue<std::string>(key), value_abs);
    (void)arguments.emplace_back(value);
    (void)args_abs.emplace_back(keyword_arg_abs);
    SetHookForArgAbstract(py::cast<py::object>(item.second), value_abs);
  }
  AddManagerForFuncGraphArgs(resource, arguments);
  resource->set_arguments(arguments);
  resource->set_args_abs(args_abs);
  executor_info->arg_list_size = args.size() + kwargs.size();
}

void JitExecutorPy::SaveCompiledGraph(const string &phase) {
  // save the graph to JitExecutorPy
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Save compiled func graph(" << func_graph->ToString() << ") phase(" << phase << ")!";
  info_[phase]->func_graph = func_graph;
  func_graph->set_attr("phase", MakeValue(GetPhasePrefix(phase)));
  MS_LOG(INFO) << "End save compiled func graph!";
}

py::object JitExecutorPy::RunInner(const py::tuple &args, const py::object &phase_obj) {
  JitRunningScope jit_running_scope;
  if (!py::isinstance<py::str>(phase_obj)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Run failed, phase input is not a str";
  }
  auto phase = py::cast<std::string>(phase_obj);
  auto phase_prefix = GetPhasePrefix(phase);
  PhaseManager::GetInstance().set_phase(phase_prefix);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (enable_infer_boost) {
    PhaseManager::GetInstance().set_phase(phase);
  }
  auto ret_val = std::make_shared<py::object>();
  if (info_.count(phase) != 0 && info_[phase]->func_graph != nullptr) {
    if (IsGraphOutputValueNodeOrParameter(info_[phase]->func_graph->output(), args, ret_val)) {
      return *ret_val;
    }
  }
#ifndef WITH_BACKEND
  if (ms_context->backend_policy() == "ge") {
    // Virtual output constructed for test cases.
    if (!args.empty()) {
      return args[0];
    }
    return args;
  }
#endif
  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "No executor info. found for phase: " << phase;
  }
  auto &execute_info = iter->second;
  MS_EXCEPTION_IF_NULL(execute_info);
  if (args.size() > execute_info->arg_list_size) {
    MS_LOG(WARNING) << "The args size: " << args.size() << ", full_arg_size: " << execute_info->arg_list_size;
  }
  ProcessVmArg(args, phase, &execute_info->arg_list);
  // Start to run phase.
  ResourcePtr resource = GetResource(phase);
  MS_EXCEPTION_IF_NULL(resource);
  if (resource->HasResult(kNoBackend)) {
    MS_LOG(INFO) << "No backend.";
    return py::none();
  }
  compile::VmEvalFuncPtr run = GetVmEvalFunc(phase);
  if (run == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Can't find run graph func for " << phase;
  }

  MS_LOG(DEBUG) << "Eval run " << ms_context->backend_policy();
  const auto &output = execute_info->func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  const auto &output_abs = output->abstract();
  MS_EXCEPTION_IF_NULL(output_abs);
  BaseRef value = (*run)(execute_info->arg_list);
  bool need_recovery = distributed::recovery::RecoveryContext::GetInstance()->enable_gpu_recovery() &&
                       distributed::recovery::RecoveryContext::GetInstance()->need_reset();
  if (need_recovery) {
    // In recovery scenario, the output value could be empty, do not transform return data.
    return py::none();
  }
  py::object res = BaseRefToPyDataWithUserData(value, output_abs);
  ClearRunArgumentsResource(args.size(), &execute_info->arg_list);
  PhaseManager::GetInstance().ClearPhase();
  MS_LOG(DEBUG) << "Run end";
  return res;
}

void JitExecutorPy::ClearRes() {
  MS_LOG(INFO) << "Clean JIT executor resource!";
  executor_ = nullptr;
}

void JitExecutorPy::CleanCompileRes(const ResourcePtr &resource) {
  MS_LOG(INFO) << "Clean compile resource start";
  abstract::AnalysisContext::ClearContext();
  ClearCompileArgumentsResource();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();
  ad::g_k_prims.clear();
  ad::DFunctor::Clear();
  ReclaimOptimizer();
  resource->Clean();
  FuncGraphLoopBreaker::Inst().CleanMetaFuncGraphs();
  CompileCacheContext::GetInstance().Clear();
  parse::Parser::CleanParserResource();
  MS_LOG(INFO) << "Clean compile resource end";
}

pipeline::ExecutorPyPtr GetExecutor(const std::string &phase) {
  if (common::GetEnv("MS_DEV_JIT_PIPELINE") == "0") {
    return pipeline::GraphExecutorPy::GetInstance();
  }
  if (phase.empty() || pipeline::JitExecutorPy::GetInstance()->HasCompiled(phase)) {
    return pipeline::JitExecutorPy::GetInstance();
  }
  return pipeline::GraphExecutorPy::GetInstance();
}
}  // namespace pipeline
}  // namespace mindspore
