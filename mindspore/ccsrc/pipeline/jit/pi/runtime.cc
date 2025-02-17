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
#include "pipeline/jit/pi/runtime.h"
#include <algorithm>
#include <iomanip>
#include <iterator>
#include <regex>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "pybind11/pybind11.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi/auto_grad/function_node.h"
#include "pipeline/jit/pi/external.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include "pipeline/jit/pi/graph_capture/graph_analyzer.h"
#include "pipeline/jit/pi/graph_compiler/abstract_type_deducer.h"
#include "pipeline/jit/pi/graph_compiler/compiler.h"
#include "pipeline/jit/pi/graph_compiler/cg/byte_code_generator.h"
#include "pipeline/jit/pi/graph_compiler/inliner/func_inliner.h"
#include "pipeline/jit/pi/graph_compiler/parser/byte_code_parser.h"
#include "pipeline/jit/pi/graph_compiler/utils.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/pi/utils/opcode_declare.h"
#include "pipeline/jit/pi/graph_guard/guard.h"
#include "pipeline/jit/pi/graph_guard/strategy.h"
#include "pipeline/jit/pi/graph_guard/shape_ctx.h"
#include "pipeline/jit/pi/capture_context.h"
#include "pipeline/jit/ps/pipeline_jit.h"
#include "pipeline/pynative/pynative_utils.h"
#include "runtime/pynative/op_executor.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/pi/graph_capture/code_generator.h"
#include "utils/convert_utils_base.h"
#include "pipeline/jit/pi/eval_frame_hook.h"
#include "include/common/utils/tensor_py.h"

namespace mindspore {
namespace pijit {

void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard);
void AddGuardForParam(const PyFrameWrapper &f, OptGuardPtr guard, bool detach);
void AddGuardForGlobals(const PyFrameWrapper &f, OptGuardPtr guard, bool detach);
static void AddGradFlagForParam(const OptGuardPtr &guard, bool detach);
static void CollectTraceBack(JitCompileResults *c, PyCodeObject *code, bool is_graph_mode);

class StaticAnalysisExceptionCleaner {
 public:
  StaticAnalysisExceptionCleaner() = default;
  ~StaticAnalysisExceptionCleaner() { StaticAnalysisException::Instance().ClearException(); }
};

class RunEnvironment {
 public:
  RunEnvironment() = default;

  void fetchAndSetRunEnv(const JitCompileResults *jcr) {
    auto ms_context = MsContext::GetInstance();
    run_mode_ = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
    jit_level_ = ms_context->GetJitLevel();
    task_sink_ = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);

    auto jit_level = jcr->conf()->getJitLevel();
    auto grad_flag = pynative::PyNativeExecutor::GetInstance()->grad_flag();
    auto run_mode = jit_level == "O2" && !grad_flag ? kGraphMode : kPynativeMode;
    auto task_sink = jit_level == "O2" && !grad_flag;
    ms_context->set_param(MS_CTX_EXECUTION_MODE, run_mode);
    ms_context->set_param(MS_CTX_JIT_LEVEL, jit_level);
    ms_context->SetJitLevel(jit_level);
    ms_context->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink);
    ms_context->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, task_sink);
  }

  void resumePreviousRunEnv() {
    auto ms_context = MsContext::GetInstance();
    ms_context->set_param(MS_CTX_EXECUTION_MODE, run_mode_);
    ms_context->set_param(MS_CTX_JIT_LEVEL, jit_level_);
    ms_context->SetJitLevel(jit_level_);
    ms_context->set_param<bool>(MS_CTX_ENABLE_TASK_SINK, task_sink_);
    ms_context->set_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK, task_sink_);
  }

 private:
  int run_mode_ = kPynativeMode;
  std::string jit_level_;
  bool task_sink_ = false;
};

static void PrintGuardPerf() {
  std::map<std::string, std::pair<size_t, size_t>> guard_info;
  std::map<std::string, std::pair<size_t, size_t>> guard_freq_info;
  std::map<std::string, std::pair<size_t, size_t>> trace_info;
  std::map<std::string, std::pair<size_t, std::vector<size_t>>> item_info;
  OptGuardPerf::GetGuardPerf()->GetGuardPerfInfo(&guard_info, &item_info, &trace_info, &guard_freq_info);
  std::cout << "Guard performance info:" << std::endl;
  std::cout << "guard, count, total time, success, fail" << std::endl;
  for (const auto &item : guard_info) {
    auto iter = guard_freq_info.find(item.first);
    if (iter != guard_freq_info.end()) {
      std::cout << "guard:" << item.first << ", " << item.second.first << ", " << item.second.second << ","
                << iter->second.first << "," << iter->second.second << std::endl;
    } else {
      std::cout << "guard:" << item.first << ", " << item.second.first << ", " << item.second.second << std::endl;
    }
  }
  std::cout << "trace, count, total time" << std::endl;
  for (const auto &item : trace_info) {
    std::cout << "trace:" << item.first << ", " << item.second.first << ", " << item.second.second << std::endl;
  }
  std::cout << "item, count, [stage time]" << std::endl;
  for (const auto &item : item_info) {
    std::cout << "item:" << item.first << "," << item.second.first << ", [";
    for (auto stage : item.second.second) {
      std::cout << stage << ",";
    }
    std::cout << "]" << std::endl;
  }
}

class GuardPerfLogger {
 public:
  ~GuardPerfLogger() {
    if (kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogGuardPerf)) {
      PrintGuardPerf();
    }
  }
};
GuardPerfLogger at_exit_printer;

// jit compiler initialize
static void ensureInitialize() {
  static bool init = false;
  if (init) {
    return;
  }
  init = true;
}

void Traceback::PushInlineInfo(InlineInfo info) {
  const auto &it = inline_infos_.find(info.root_name_);
  if (it != inline_infos_.cend()) {
    it->second.push_back(info);
  } else {
    std::list<InlineInfo> inlines;
    inlines.push_back(info);
    inline_infos_.emplace(info.root_name_, inlines);
  }
}

static void PrintLabel(std::stringstream &os, const std::string &str, int distance = 30) {
  os << std::left << std::setw(distance) << str << ": ";
}

std::string Traceback::Dump(bool is_all) const {
  constexpr auto width = 10;

  std::stringstream os;
  std::string cur_name = tbs_.empty() ? "" : tbs_.back().func_name_;
  if (is_all) {
    os << "*** Dump Traceback on [" << raw_func_info_name_ << "] ***\n";
  } else {
    os << "*** Dump ByteCode After Traceback on [" << cur_name << "] ***\n";
  }
  if (tbs_.empty()) {
    return os.str();
  }
  std::list<Element> candidates;
  if (is_all) {
    candidates = tbs_;
  } else {
    // last one traceback
    candidates.emplace_back(tbs_.back());
  }
  // dump traceback list head
  int name_length = FindMaxNameLength(candidates);
  os << std::left << std::setw(name_length) << "func_name:  -->  " << std::left << std::setw(name_length)
     << "changed_func:" << std::left << std::setw(width) << "run_mode:" << std::left << std::setw(kThree * width)
     << "stop_trace:" << std::left << std::setw(width) << "code_size:" << std::endl;
  os << "--------------------------------------------------------------------------------------\n";
  // dump traceback list content
  for (const auto &tb : candidates) {
    os << std::left << std::setw(name_length) << tb.func_name_ << "  -->  ";
    os << std::left << std::setw(name_length) << tb.changed_func_;
    if (tb.is_graph_mode_) {
      os << std::left << std::setw(width) << "[GRAPH]";
    } else {
      os << std::left << std::setw(width) << "PYNATIVE";
    }
    // dump stop trace reason
    auto it_trace = stop_trace_res_.find(tb.func_name_);
    if (it_trace != stop_trace_res_.cend()) {
      os << std::left << std::setw(kThree * width) << GetStopTraceReasonDesc(it_trace->second);
    } else {
      os << std::left << std::setw(kThree * width) << "unknown";
    }
    os << std::left << std::setw(width) << tb.code_size_ << " =====>\n";
    // dump inline info
    DumpInlineInfo(os, tb.func_name_);
  }
  os << "\n\n";
  if (is_all) {
    os << DumpSummary();
  }
  return os.str();
}

void Traceback::DumpInlineInfo(std::stringstream &os, const std::string &func_name) const {
  const auto &it = inline_infos_.find(func_name);
  if (it == inline_infos_.cend()) {
    return;
  }
  for (const auto &info : it->second) {
    std::string space((info.depth + 1) * kTwo, ' ');
    os << space << "| inline_info:" << GetInlineReasonDesc(info.res) << " line:" << info.line;
    if (!info.inline_name_.empty()) {
      os << " func_name:" << info.inline_name_;
    }
    if (info.res == InlineReason::kInline || info.res == InlineReason::kInlinePartial) {
      os << " code_size:" << info.code_size_;
    }
    os << "\n";
  }
}

std::string Traceback::DumpSummary() const {
  std::stringstream os;
  if (tbs_.empty()) {
    return os.str();
  }
  os << "*** Dump Summary on [" << raw_func_info_name_ << "] ***\n";
  PrintLabel(os, "traceback_num");
  os << tbs_.size() << "\n";

  std::array<int, kStopTrace_Reason_Count> stop_trace_reason_array{0};
  std::array<int, kInline_Reason_Count> inline_reason_array{0};
  int graph_mode_num = 0;
  int raw_code_size = raw_code_size_;
  int pynative_code_size = 0;
  int graph_mode_code_size = 0;
  for (const auto &tb : tbs_) {
    if (tb.is_graph_mode_) {
      graph_mode_num++;
      graph_mode_code_size += tb.code_size_;
    } else {
      pynative_code_size += tb.code_size_;
    }
    auto it_trace = stop_trace_res_.find(tb.func_name_);
    if (it_trace != stop_trace_res_.cend()) {
      // count stop trace reason
      stop_trace_reason_array[it_trace->second]++;
    }
    const auto &it_inline = inline_infos_.find(tb.func_name_);
    if (it_inline == inline_infos_.cend()) {
      continue;
    }
    for (const auto &info : it_inline->second) {
      // count inline reason
      inline_reason_array[info.res]++;
      if (info.res == InlineReason::kInline || info.res == InlineReason::kInlinePartial) {
        raw_code_size += info.code_size_;
      }
    }
  }
  PrintLabel(os, "graph_mode_num");
  os << graph_mode_num << "\n";
  PrintLabel(os, "raw_code_size(+ inline)");
  os << raw_code_size << "\n";
  PrintLabel(os, "pynative_code_size");
  os << pynative_code_size << "\n";
  PrintLabel(os, "graph_mode_code_size");
  os << graph_mode_code_size << "\n";
  os << "----------stop_trace_reason----------\n";
  for (size_t i = 0; i < stop_trace_reason_array.size(); ++i) {
    PrintLabel(os, GetStopTraceReasonDesc(static_cast<StopTraceReason>(i)));
    os << stop_trace_reason_array[i] << "\n";
  }
  os << "----------inline_reason----------\n";
  for (size_t i = 0; i < inline_reason_array.size(); ++i) {
    PrintLabel(os, GetInlineReasonDesc(static_cast<InlineReason>(i)));
    os << inline_reason_array[i] << "\n";
  }
  os << "\n\n";
  return os.str();
}

int Traceback::FindMaxNameLength(const std::list<Element> &tbs) const {
  constexpr auto name_length = kFive * (kTwo + kFive);
  int max_length = 15;
  for (const auto &tb : tbs) {
    int len1 = SizeToInt(tb.func_name_.length());
    int len2 = SizeToInt(tb.changed_func_.length());
    max_length = std::max(max_length, std::max(len1, len2)) + kTwo;
  }
  max_length = std::min(max_length, name_length);
  return max_length;
}

static void GuardForFrame(const PyFrameWrapper &f, const OptCodePtr &oc, const GraphJitConfig &conf) {
  const char *code_name = f.GetCode().Name();
  AddConfigToGuard(conf, oc->GetGuard());
  AddGuardForParam(f, oc->GetGuard(), conf.GetBoolConfig(GraphJitConfig::kGuardDetachObject));
  AddGradFlagForParam(oc->GetGuard(), conf.GetBoolConfig(GraphJitConfig::kGuardDetachObject));
  if (conf.GetBoolConfig(GraphJitConfig::kPrintGuard)) {
    GRAPH_JIT_LOG_F("Guard on %s by %s!\n", code_name, oc->GetGuard()->GetDescript().c_str());
    return;
  }
  if (IS_OUTPUT_ON(mindspore::kDebug)) {
    // It tooks too much time in Guard's GetDescript function when trace depth is too large.
    MS_LOG(DEBUG) << "Guard on " << code_name << " by " << oc->GetGuard()->GetDescript() << "!" << std::endl;
  }
}

static void ValidateCompiledResults(const JitCompileResults *c) {
  if (c->stat() != JitCompileResults::GRAPH_CALLABLE) {
    return;
  }
  bool valid_res;
  if (c->code()->GetNativeFunc()) {
    valid_res = true;
  } else {
    valid_res = c->code()->GetPythonCode() != nullptr;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(valid_res, "check compiled result");
}

static void MarkBreak(Graph *g) {
  TimeRecorder recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
  int break_bci = g->GetStopTraceBci();
  if (break_bci == -1) {
    return;
  }
  PyCodeObject *code;
  const auto &nodes = g->GetTracedNodes();
  if (nodes.empty()) {
    code = g->GetCodeObj();
  } else {
    auto iter = std::find_if(nodes.begin(), nodes.end(), [&break_bci](ValueNode *i) { return i->bci() >= break_bci; });
    iter -= iter == nodes.end();
    for (code = (*iter)->GetGraph()->GetCodeObj(); code == nullptr && iter != nodes.begin(); --iter) {
      code = (*iter)->GetGraph()->GetCodeObj();
    }
  }
  MS_EXCEPTION_IF_NULL(code);
  auto jcr = GetJitCompileResults(code);
  if (jcr != nullptr) {
    jcr->break_count()++;
  }
}

std::vector<py::object> GetAllArgs(JitCompileResults *jcr) {
  auto all_args = jcr->origin_frame().PackArgs();
  constexpr size_t arg_index = 0;
  constexpr size_t vargs_index = 1;
  constexpr size_t kwargs_index = 2;
  auto args = py::cast<py::list>(all_args[arg_index]);
  if (all_args[vargs_index].ptr() != nullptr) {
    PyList_Append(args.ptr(), all_args[vargs_index].ptr());  // args + vargs
  }
  if (all_args[kwargs_index].ptr() != nullptr) {
    PyList_Append(args.ptr(), all_args[kwargs_index].ptr());  // args + kwargs
  }
  return args.cast<std::vector<py::object>>();
}

static void GraphCapture(JitCompileResults *jcr);
static auto HandleBreakAtLoop(JitCompileResults *jcr, const GraphBuilderPtr &g) {
  // one stage need adapter
  if (g->GetGraph()->IsBreakAtLoopAfterUnrolling()) {
    if (jcr->conf()->GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
      GRAPH_JIT_LOG_F("===> graph break after loop unrolling\n%s\n", g->GetGraph()->ToString(1).c_str());
    }
    // reset guard
    jcr->code()->SetGuard(std::make_shared<OptGuard>());
    AddConfigToGuard(*jcr->conf(), jcr->code()->GetGuard());
    // disable loop unroll
    jcr->conf()->SetBool<GraphJitConfig::kLoopUnrolling>(Py_False);
    // restart captured
    GraphCapture(jcr);
    // reset config
    jcr->conf()->SetBool<GraphJitConfig::kLoopUnrolling>(Py_True);
    return true;
  }
  return false;
}

static auto HandleUnsupportedSyntax(JitCompileResults *jcr, const GraphBuilderPtr &g) {
  int break_bci = g->GetGraph()->GetStopTraceBci();
  if (break_bci == -1) {
    return false;
  }
  int break_op = g->GetGraph()->GetCFG()->instr_pool()[break_bci]->op();
  bool unsupported =
    break_op == WITH_CLEANUP_START || break_op == WITH_CLEANUP_FINISH || break_op == END_FINALLY || break_op == RERAISE;
  if (g->StackSize() > 0 || unsupported) {
    // something happened in with syntax
    jcr->code()->SetGuard(std::make_shared<OptGuard>());
    AddConfigToGuard(*jcr->conf(), jcr->code()->GetGuard());
    jcr->conf()->SetBool<GraphJitConfig::kSkipException>(Py_True);
    GraphCapture(jcr);
    g->GetTryBlockStacks().clear();
    jcr->conf()->SetBool<GraphJitConfig::kSkipException>(Py_False);
    return true;
  }
  return false;
}

static auto TraceRun(JitCompileResults *jcr) {
  TimeRecorder recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
  GraphBuilderPtr g = GraphBuilder::Creator(jcr->origin_frame());
  (void)g->TraceRun();
  return g;
}

static auto Analyze(const GraphBuilderPtr &g) {
  TimeRecorder recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  auto analyzer = GraphAnalyzer::Creator(g);
  analyzer->Analyze();
  return analyzer;
}

// preprocess before compile, split bytecode to sub-function
// return whether the code should be modified
static void GraphCapture(JitCompileResults *jcr) {
  TimeRecorder recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
  MS_EXCEPTION_IF_NULL(jcr->code());

  GraphJitConfig &conf = *jcr->conf();
  AObject::SetTraceFlag(conf.GetBoolConfig(GraphJitConfig::kTraceFlag));
  GraphBuilderPtr g = TraceRun(jcr);
  if (HandleUnsupportedSyntax(jcr, g)) {
    return;
  }
  if (g->GetGraph()->ShouldNeverCompile()) {
    if (jcr->conf()->GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
      GRAPH_JIT_LOG_F("===> graph break after loop unrolling\n%s\n", g->GetGraph()->ToString(1).c_str());
    }
    jcr->set_stat(JitCompileResults::NEVER_COMPILE);
    return;
  }
  GraphAnalyzerPtr analyzer = Analyze(g);
  if (HandleBreakAtLoop(jcr, g)) {
    return;
  }
  MarkBreak(g->GetGraph());

  // dump DFG
  if (conf.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    g->DumpDFG();
    if (conf.GetBoolConfig(GraphJitConfig::kTraceFlag)) {
      const auto &debug_str = analyzer->GetCaptureInfo().ToString();
      PY_PRINT_F("*** Dump One Stage ByteCode Collection After CodeGen *** \n%s", debug_str.c_str());
    }
  }

  py::object new_code = MakeCodeFromCodeGen(g, analyzer, jcr->origin_frame().Globals().ptr());
  if (new_code.ptr() != nullptr) {
    jcr->code()->SetPythonCode(new_code);
    jcr->set_stat(JitCompileResults::GRAPH_CALLABLE);
  }

  if (conf.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    Utils::DisFuncObject(new_code.ptr());
    GRAPH_JIT_LOG_F("\n\n");
  }

  // collect stop trace reason to traceback
  jcr->tbs()->PushStopTraceRes(g->GetGraph()->GetCodeName(), g->GetGraph()->GetStopTraceReason());

  bool captured = !analyzer->NeedInterpret() && !conf.GetBoolConfig(GraphJitConfig::kInterpretCapturedCode);
  if (captured && !jcr->conf()->GetBoolConfig(GraphJitConfig::kTraceFlag)) {
    jcr->set_stat(JitCompileResults::GRAPH_CAPTURED);
  }
}

static void CollectTraceBack(JitCompileResults *c, PyCodeObject *code, bool is_graph_mode) {
  if (code == nullptr) {
    code = c->origin_frame().GetCode().ptr();
  }
  std::string name = c->origin_frame().GetCode().Name();
  std::string changed_name = Utils::GetPyName(code->co_name);
  int code_size = _PyCode_NBYTES(code);
  c->tbs()->PushTbs({name, changed_name, code_size, is_graph_mode});
}

std::string GetFuncGraphPhase(const PyFrameWrapper &frame, const OptCodePtr &oc) {
  PyCodeObject *co = frame.GetCode().ptr();
  const char *co_name = frame.GetCode().Name();
  const char *co_filename = frame.GetCode().FileName();
  std::string phase = std::string() + co_filename + "_" + std::to_string(co->co_firstlineno) + "_" + co_name;
  if (oc != nullptr) {
    phase += std::to_string(oc->GetGuard()->Info().Id());
  } else {
    py::dict locals = frame.Locals();
    for (const auto &pair : locals) {
      auto node = GraphUtils::ConvertPythonObjectToAnfNode(py::cast<py::object>(pair.second));
      phase += "_" + node->abstract()->ToString();
    }
  }
  phase += ".pi_jit";
  return phase;
}

void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard) {
  std::map<std::string, bool> bool_cfg;
  std::map<std::string, int> int_cfg;
  bool_cfg[kSpecializeScalar] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeScalar);
  bool_cfg[kSpecializeContainer] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeContainer);
  bool_cfg[kSpecializeTensor] = c.GetBoolConfig(GraphJitConfig::kGuardSpecializeTensor);
  int_cfg[kGuardRelaxCnt] = c.getIntConfig(GraphJitConfig::kGuardRelaxCount);
  guard->UpdateConfig(bool_cfg, int_cfg);
}

void AddGuardForParam(const PyFrameWrapper &wrapper, OptGuardPtr guard, bool detach) {
#if IS_PYTHON_3_11_PLUS
  MS_LOG(ERROR) << "not implement in python3.11";
#else
  auto lh = [&guard, &detach](PyObject *value, int fast_index) {
    RootTracePtr ptr = std::make_shared<RootTrace>(value, mindspore::pijit::TraceType::Param, fast_index);
    guard->GuardOn(ptr, mindspore::pijit::GuardLevel::GDeduce, false);
    if (detach) {
      ptr->Detach();
    }
  };
  auto ch = [&wrapper, &guard, &detach](PyObject *cell_or_local, int fast_index) {
    bool is_cell = PyCell_Check(cell_or_local);
    auto value = is_cell ? PyCell_GET(cell_or_local) : cell_or_local;
    auto type = is_cell ? TraceType::Deref : TraceType::Param;
#if IS_PYTHON_3_11_PLUS
    int guard_retrieve_index = fast_index;
    MS_LOG(ERROR) << "not implement in python3.11, retrieve deref index is error";
#else
    int guard_retrieve_index = fast_index - wrapper.GetCode().LocalSize();
#endif
    RootTracePtr ptr = std::make_shared<RootTrace>(value, type, guard_retrieve_index);
    guard->GuardOn(ptr, mindspore::pijit::GuardLevel::GDeduce, false);
    if (detach) {
      ptr->Detach();
    }
  };
  wrapper.ForEachFastLocal(lh, ch, ch);
#endif
}

void AddGuardForGlobals(const PyFrameWrapper &wrapper, OptGuardPtr guard, bool detach) {
#if IS_PYTHON_3_11_PLUS
  MS_LOG(ERROR) << "not implement in python3.11";
#else
  EvalFrameObject *f = wrapper.frame();
  PyCodeObject *co = wrapper.GetCode().ptr();
  const _Py_CODEUNIT *bytecodes = _PyCode_CODE(co);
  int size = static_cast<size_t>(_PyCode_NBYTES(co)) / sizeof(_Py_CODEUNIT);
  unsigned int exarg = 0;
  for (int bci = 0; bci < size; ++bci) {
    int opcode = _Py_OPCODE(bytecodes[bci]);
    int oparg = (exarg << 8) | _Py_OPARG(bytecodes[bci]);
    exarg = static_cast<unsigned>((opcode == EXTENDED_ARG) ? oparg : 0);
    if (opcode != LOAD_GLOBAL) {
      continue;
    }
    PyObject *k = PyTuple_GET_ITEM(co->co_names, oparg);
    PyObject *v = PyDict_GetItem(f->f_globals, k);
    std::string key = PyUnicode_AsUTF8(k);
    if (v == nullptr) {
      PyErr_Clear();
      continue;
    }

    TracePtr ptr = std::make_shared<RootTrace>(v, TraceType::Global, -1, key);

    AObject::Type t = AObject::GetPyType(v);
    GuardLevel level = GuardLevel::GType;
    if (t == AObject::kTypeCell || t == AObject::kTypePrimitive || t == AObject::kTypeMSDType) {
      level = GuardLevel::GDeduce;
    } else if (t == AObject::kTypeFunction) {
      ptr = std::make_shared<OpTrace>(PyFunction_GET_CODE(v), LOAD_ATTR, -1, std::vector<TracePtr>({ptr}), "__code__");
      level = GuardLevel::GId;
    } else if (t == AObject::kTypeTuple || t == AObject::kTypeList || t == AObject::kTypeDict) {
      /**
       * graph treat tuple, list, dict as constant variable.
       * add container guard and check it, check contains Tensor
       */
      continue;
    }

    guard->GuardOn(ptr, level, false);
    if (detach) {
      ptr->Detach();
    }
  }
#endif
}

static void AddGradFlagForParam(const OptGuardPtr &guard, bool detach) {
  bool grad_flag = pynative::PyNativeExecutor::GetInstance()->RequiresGrad();
  CustomizedTracePtr ptr = std::make_shared<CustomizedTrace>(
    grad_flag ? Py_True : Py_False,
    [](PTraceContext context) -> PyObject * {
      static pynative::PyNativeExecutor *pynative_exec = nullptr;
      if (pynative_exec == nullptr) {
        pynative_exec = pynative::PyNativeExecutor::GetInstance().get();
      }
      PyObject *ret = pynative_exec->RequiresGrad() ? Py_True : Py_False;
      Py_INCREF(ret);
      return ret;
    },
    [grad_flag](bool simple) -> std::string {
      if (simple) {
        return std::string("g\\") + std::to_string(grad_flag ? 1 : 0);
      }
      return std::string("{PyNativeExecutor::GetInstance()->RequiresGrad() == ") + std::to_string(grad_flag) +
             std::string("}(type:") + std::to_string(TraceType::Customized) + std::string(")");
    });
  guard->GuardOn(ptr, mindspore::pijit::GuardLevel::GEqual, true);
  if (detach) {
    ptr->Detach();
  }
}

#if !IS_PYTHON_3_11_PLUS
static std::string CallGraphCompiler(JitCompileResults *jcr, PyFunctionObject *func, const PyFrameWrapper &frame) {
  std::string phase = GetFuncGraphPhase(frame, jcr->code());
  MS_LOG(DEBUG) << "Phase is " << phase << "!";
  CallableGraph callable = mindspore::pijit::Compiler::Compile(*func, frame, phase);
  if (callable == nullptr) {
    jcr->set_stat(JitCompileResults::NEVER_COMPILE);
    return std::string();
  }

  ReleaseFunc rFunc = nullptr;
  if (jcr->conf()->GetBoolConfig(GraphJitConfig::kAutoCleanCache)) {
    rFunc = [phase]() {
      auto graph_executor = pipeline::GetExecutor();
      if (graph_executor->HasCompiled(phase)) {
        py::str p(phase);
        py::set s;
        s.add(phase);
        py::object o = py::none();
        graph_executor->DelNetRes(o, s);
        MS_LOG(DEBUG) << "To release " << phase;
      }
    };
  }
  jcr->code()->SetNativeFunc(phase, callable, rFunc);
  jcr->set_stat(JitCompileResults::GRAPH_CALLABLE);
  return phase;
}
#endif

std::string GraphToString(FuncGraphPtr graph) {
  std::ostringstream graph_buffer;
  DumpIR(graph_buffer, graph);
  auto ret = graph_buffer.str();
  std::regex regAddress("(0x)([0-9a-f]+)");
  ret = std::regex_replace(ret, regAddress, "");
  std::regex regFunc(std::string("(") + graph->ToString() + std::string(")"));
  ret = std::regex_replace(ret, regFunc, "");
  std::regex regVar("(\\%[0-9]+\\()([A-Za-z0-9_]+)(\\))");
  ret = std::regex_replace(ret, regVar, "$1$3");
  std::regex regNode("CNode_([0-9]+)");
  ret = std::regex_replace(ret, regNode, "");
  return ret;
}

static void GraphCompile(JitCompileResults *jcr, const PyFrameWrapper &frame) {
#if IS_PYTHON_3_11_PLUS
  MS_LOG(ERROR) << "not implement in python3.11";
#else
  TimeRecorder recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
  GuardForFrame(frame, jcr->code(), *jcr->conf());
  AddGuardForGlobals(frame, jcr->code()->GetGuard(), jcr->conf()->GetBoolConfig(GraphJitConfig::kGuardDetachObject));

  bool enable_dynamicshape = jcr->conf()->GetBoolConfig(GraphJitConfig::kEnableDynamicShape);
  OptStrategy::MakeGCStrategy(jcr->codehub(), jcr->conf()->getIntConfig(GraphJitConfig::kLimitGraphSize),
                              jcr->conf()->getIntConfig(GraphJitConfig::kLimitGraphCount), enable_dynamicshape,
                              jcr->code());
  py::object func_handler = frame.GetFunction();
  PyFunctionObject *func = reinterpret_cast<PyFunctionObject *>(func_handler.ptr());

  std::vector<PyObject *> backup;
  if (enable_dynamicshape) {
    backup = jcr->code()->GetGuard()->ApplyDynamicShape(frame.frame());
    PyFrame_FastToLocals(frame.frame());
  }

  RunEnvironment runEnvironment;
  runEnvironment.fetchAndSetRunEnv(jcr);
  std::string phase = CallGraphCompiler(jcr, func, frame);
  runEnvironment.resumePreviousRunEnv();

  if (enable_dynamicshape) {
    jcr->code()->GetGuard()->RevertDynamicShape(frame.frame(), backup);
    PyFrame_FastToLocals(frame.frame());
  }

  if (jcr->conf()->GetBoolConfig(GraphJitConfig::kReuseGraph)) {
    auto graph_executor = pipeline::GetExecutor();
    FuncGraphPtr ms_func_graph = graph_executor->GetFuncGraph(phase);
    std::string key = GraphToString(ms_func_graph);
    auto pcode = OptCodeHub::Filter(key, [jcr, graph_executor, ms_func_graph](OptCodePtr code) {
      FuncGraphPtr func_graph = graph_executor->GetFuncGraph(code->GetPhase());
      FuncGraphPairMapEquiv equiv_graph;
      NodeMapEquiv equiv_node;
      if (func_graph != nullptr && Isomorphic(ms_func_graph, func_graph, &equiv_graph, &equiv_node)) {
        return true;
      } else {
        return false;
      }
    });
    if (pcode != nullptr) {
      if (jcr->conf()->GetBoolConfig(GraphJitConfig::kPrintReuseGraph)) {
        std::ostringstream graph_buffer;
        DumpIR(graph_buffer, ms_func_graph);
        std::cout << "Graph Duplicated:" << std::endl;
        std::cout << "  Graph:" << graph_buffer.str() << std::endl;
        std::cout << "  Bytecode:" << std::endl;
        Utils::DisFuncObject(PyFunction_GET_CODE(func));
      }
      // find duplicate graph and reuse it
      pcode->Copy(jcr->code());
    } else {
      // current graph is a new one and register it
      OptCodeHub::Register(key, jcr->code());
    }
  }
#endif
}

extern bool UnsupportedCodeTypeCheck(PyCodeObject *co);
static bool JitCompile(PyThreadState *tstate, JitCompileResults *c) {
  const auto &frame = c->origin_frame();
  PyCodeObject *code = frame.GetCode().ptr();
  if (UnsupportedCodeTypeCheck(code)) {
    return false;
  }
  ShapeContext sc(c->origin_frame().frame(), c->input_signature().ptr());
  MS_LOG(DEBUG) << "---start compile " << py::str(reinterpret_cast<PyObject *>(code)) << "---";

  // new guard code
  c->set_code(c->codehub()->AddOptTarget(OptOption::CreateOptionByPoint(c)));
  AddConfigToGuard(*c->conf(), c->code()->GetGuard());

  if (c->stat() == JitCompileResults::GRAPH_CANDIDATE) {
    TimeRecorder time_recorder("kTimeCompileCapture", kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureProcess,
                                       "PIJitCapture");
    c->set_stat(JitCompileResults::GRAPH_BUILDING);
    auto aobject_resource = AObject::MakeResource();
    bool enable_dynamicshape = c->conf()->GetBoolConfig(GraphJitConfig::kEnableDynamicShape);
    std::vector<PyObject *> backup;
    if (c->conf()->GetBoolConfig(GraphJitConfig::kTraceFlag)) {
      GuardForFrame(frame, c->code(), *c->conf());
      OptStrategy::MakeGCStrategy(c->codehub(), c->conf()->getIntConfig(GraphJitConfig::kLimitGraphSize),
                                  c->conf()->getIntConfig(GraphJitConfig::kLimitGraphCount), enable_dynamicshape,
                                  c->code());
      if (enable_dynamicshape) {
        backup = c->code()->GetGuard()->ApplyDynamicShape(frame.frame());
#if !IS_PYTHON_3_11_PLUS
        PyFrame_FastToLocals(frame.frame());
#endif
      }
    }
    GraphCapture(c);
    if (c->conf()->GetBoolConfig(GraphJitConfig::kTraceFlag)) {
      if (enable_dynamicshape) {
        c->code()->GetGuard()->RevertDynamicShape(frame.frame(), backup);
#if !IS_PYTHON_3_11_PLUS
        PyFrame_FastToLocals(frame.frame());
#endif
      }
      AddGuardForGlobals(frame, c->code()->GetGuard(), c->conf()->GetBoolConfig(GraphJitConfig::kGuardDetachObject));
    }
    aobject_resource.Release();
  }
  sc.ApplySignature();

  if (c->stat() == JitCompileResults::GRAPH_CAPTURED) {
    TimeRecorder time_recorder("kTimeCompileGraph", kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureCompile,
                                       "PIJitCompile");
    c->set_stat(JitCompileResults::GRAPH_BUILDING);
    GraphCompile(c, frame);
  }

  if (c->conf()->getIntConfig(GraphJitConfig::kGuardRelaxCount) > 0) {
    auto guard = c->code()->GetGuard()->Optimize();
    if (guard != nullptr) {
      c->code()->SetGuard(guard);
    }
  }
  c->code()->GetGuard()->FilterConstItem();

  CollectTraceBack(c, c->code()->GetPythonCode(), c->code()->GetNativeFunc() != nullptr);

  if (c->conf()->GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    GRAPH_JIT_LOG_F("%s\n", c->tbs()->Dump().c_str());

    GRAPH_JIT_LOG_F("generated guard at %s\n", std::string(py::str(reinterpret_cast<PyObject *>(code))).c_str());
    GRAPH_JIT_LOG_F("%s\n", c->code()->GetGuard()->ToString().c_str());
  }
  if (c->stat() != JitCompileResults::GRAPH_CALLABLE) {
    c->set_stat(JitCompileResults::NEVER_COMPILE);
    return false;
  }
  return true;
}

static py::object ResultMutable(py::object obj) {
  py::object mutable_func = Utils::GetModuleAttr("mindspore.common", "mutable", false, true);
  if (py::isinstance<py::tuple>(obj)) {
    auto tuple_obj = obj.cast<py::tuple>();
    py::list mutable_list(tuple_obj);
    for (size_t i = 0; i < tuple_obj.size(); i++) {
      try {
        auto mutable_element = mutable_func(tuple_obj[i]);
        mutable_list[i] = mutable_element;
      } catch (py::error_already_set &e) {
        if (PyErr_Occurred()) {
          PyErr_Clear();
        }
        continue;
      }
    }
    auto mutable_tuple = py::tuple(mutable_list);
    return mutable_tuple;
  } else {
    try {
      auto mutable_obj = mutable_func(obj);
      return mutable_obj;
    } catch (py::error_already_set &e) {
      if (PyErr_Occurred()) {
        PyErr_Clear();
      }
    }
  }
  return obj;
}

static py::object CallGraph(const JitCompileResults *c, const py::object &args, const py::object &kwvargs) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureRunGraph,
                                     "PIJitRunGraph");

  StaticAnalysisExceptionCleaner exception_cleaner;
  CaptureContext::DisableScope compiler_disable_scope;

  RunEnvironment runEnvironment;
  runEnvironment.fetchAndSetRunEnv(c);
  PyObject *py_args = args.ptr();
  PyObject *py_kwvargs = kwvargs.ptr() == Py_None ? nullptr : kwvargs.ptr();
  PyObject *res;
  if (c->conf()->GetBoolConfig(GraphJitConfig::kPerfStatistics) &&
      c->code()->GetPerf(OptPerf::PerfKind::kPerfGraph)->GetStatistics()->GetTotalCount() <
        c->conf()->getIntConfig(GraphJitConfig::kPerfStatisticsCount)) {
    std::function<PyObject *(PyObject * py_args, PyObject * py_kwvargs)> func = [c](PyObject *py_args,
                                                                                    PyObject *py_kwvargs) {
      auto ret = c->code()->GetNativeFunc()(py_args, py_kwvargs);
      runtime::Pipeline::Get().WaitAll();
      return ret;
    };
    runtime::Pipeline::Get().WaitAll();
    res = CallFunction(c->code()->GetPerf(OptPerf::PerfKind::kPerfGraph), func, py_args, py_kwvargs);
  } else {
    res = c->code()->GetNativeFunc()(py_args, py_kwvargs);
  }
  runEnvironment.resumePreviousRunEnv();
  if (res == NULL && !PyErr_Occurred()) {
    PyErr_SetString(PyExc_RuntimeError, "compiled graph execute failed");
  }
  auto res_obj = py::reinterpret_steal<py::object>(res);
  if (!c->conf()->GetBoolConfig(GraphJitConfig::kTraceFlag)) {
    return ResultMutable(res_obj);
  }
  return res_obj;
}

static py::object CallCompiledCallable(PyThreadState *tstate, const PyFrameWrapper &f, const JitCompileResults *c) {
  PyObject *res;
  if (c->conf()->GetBoolConfig(GraphJitConfig::kPerfStatistics) &&
      c->code()->GetPerf(OptPerf::PerfKind::kPerfPyNative)->GetStatistics()->GetTotalCount() <
        c->conf()->getIntConfig(GraphJitConfig::kPerfStatisticsCount)) {
    auto func = [&tstate, &f, &c]() {
      auto res = f.EvalNewCode(tstate, c->code()->GetPythonCode());
      runtime::Pipeline::Get().WaitAll();
      return res;
    };
    runtime::Pipeline::Get().WaitAll();
    res = CallFunction(c->code()->GetPerf(OptPerf::PerfKind::kPerfPyNative), std::function(func));
  } else {
    res = f.EvalNewCode(tstate, c->code()->GetPythonCode());
  }
  if (res == NULL && !PyErr_Occurred()) {
    PyErr_Format(PyExc_RuntimeError, "compiled function failed with unknown error");
  }
  return py::reinterpret_steal<py::object>(res);
}

static bool CheckTensorInContainer(py::object args) {
  if (py::isinstance<py::tuple>(args)) {
    py::tuple t = py::cast<py::tuple>(args);
    for (size_t i = 0; i < t.size(); ++i) {
      if (CheckTensorInContainer(t[i])) {
        return true;
      }
    }
  } else if (py::isinstance<py::list>(args)) {
    py::list l = py::cast<py::list>(args);
    for (size_t i = 0; i < l.size(); ++i) {
      if (CheckTensorInContainer(l[i])) {
        return true;
      }
    }
  }
  if (IsStubTensor(args) || tensor::IsTensorPy(args)) {
    return true;
  }
  return false;
}

static bool CheckAbstract(abstract::AbstractBasePtr abs, bool incontainer);

static bool CheckContainer(abstract::AbstractBasePtr abs) {
  if (abs->isa<abstract::AbstractTuple>()) {
    auto elems = abs->cast<abstract::AbstractTuplePtr>()->elements();
    for (size_t idx = 0; idx < elems.size(); ++idx) {
      if (!CheckAbstract(elems[idx], true)) {
        return false;
      }
    }
  }
  if (abs->isa<abstract::AbstractList>()) {
    auto elems = abs->cast<abstract::AbstractListPtr>()->elements();
    for (size_t idx = 0; idx < elems.size(); ++idx) {
      if (!CheckAbstract(elems[idx], true)) {
        return false;
      }
    }
  }
  if (abs->isa<abstract::AbstractSequence>()) {
    auto elems = abs->cast<abstract::AbstractSequencePtr>()->elements();
    for (size_t idx = 0; idx < elems.size(); ++idx) {
      if (!CheckAbstract(elems[idx], true)) {
        return false;
      }
    }
  }
  if (abs->isa<abstract::AbstractDictionary>()) {
    auto elems = abs->cast<abstract::AbstractDictionaryPtr>()->elements();
    for (size_t idx = 0; idx < elems.size(); ++idx) {
      if (!CheckAbstract(elems[idx].first, true) || !CheckAbstract(elems[idx].first, true)) {
        return false;
      }
    }
  }
  if (abs->isa<abstract::AbstractSlice>()) {
    auto slice = abs->cast<abstract::AbstractSlicePtr>();
    return !CheckAbstract(slice->start(), true) || !CheckAbstract(slice->stop(), true) ||
           !CheckAbstract(slice->step(), true);
  }
  return true;
}

static bool CheckAbstract(abstract::AbstractBasePtr abs, bool incontainer) {
  if (incontainer && abs->isa<abstract::AbstractAny>()) {
    return false;
  }
  if (abs->isa<abstract::AbstractTuple>() || abs->isa<abstract::AbstractList>() ||
      abs->isa<abstract::AbstractSequence>() || abs->isa<abstract::AbstractDictionary>() ||
      abs->isa<abstract::AbstractSlice>()) {
    return CheckContainer(abs);
  }
  if (abs->isa<abstract::AbstractNone>() || abs->isa<abstract::AbstractNull>() || abs->isa<abstract::AbstractType>() ||
      abs->isa<abstract::AbstractFunction>() || abs->isa<abstract::AbstractAny>()) {
    return false;
  }
  if (abs->isa<abstract::AbstractScalar>()) {
    auto tp = abs->GetTypeTrack()->type_id();
    return tp != kMetaTypeNone && tp != kMetaTypeNull && tp != kNumberTypeBool;
  }
  return true;
}

static bool CheckValidReturn(const JitCompileResults *c) {
  auto graph_executor = pipeline::GetExecutor();
  FuncGraphPtr ms_func_graph = graph_executor->GetFuncGraph(c->code()->GetPhase());
  auto abs = ms_func_graph->output()->abstract();
  return CheckAbstract(abs, false);
}

static bool PreferCallGraph(const JitCompileResults *c, py::object args) {
  if (c->code()->GetNativeFunc() == nullptr) {
    return false;
  }
  if (c->conf()->GetBoolConfig(GraphJitConfig::kTraceFlag)) {
    return true;
  }
  if (!CheckValidReturn(c)) {
    return false;
  }
  py::tuple t = py::cast<py::tuple>(args);
  for (size_t i = 0; i < t.size(); ++i) {
    py::object obj = t[i];
    if (IsMutableObj(obj)) {
      continue;
    }
    if ((py::isinstance<py::list>(t[i]) || py::isinstance<py::tuple>(t[i])) && CheckTensorInContainer(t[i])) {
      return false;
    }
  }
  OptStrategy::ExecKind stat = OptStrategy::ExecKind::kExecGraph;
  if (c->conf()->GetBoolConfig(GraphJitConfig::kPerfStatistics)) {
    constexpr auto kStatisticsScale = 10000.0;
    int scale_statistics = c->conf()->getIntConfig(GraphJitConfig::kPerfStatisticsScale10000x);
    stat = OptStrategy::MakeExecStrategyByPerf(
      c->code()->GetPerf(OptPerf::PerfKind::kPerfGraph), c->code()->GetPerf(OptPerf::PerfKind::kPerfPyNative),
      c->conf()->getIntConfig(GraphJitConfig::kPerfStatisticsCount), scale_statistics / kStatisticsScale);
  }
  int graph_bytecode_min = c->conf()->getIntConfig(GraphJitConfig::kStaticGraphBytecodeMin);
  if (graph_bytecode_min > 0 && stat == OptStrategy::ExecKind::kExecGraph) {
    stat = OptStrategy::MakeExecStrategyByComplex(c->code()->GetPythonCode(), graph_bytecode_min);
  }
  return stat == OptStrategy::ExecKind::kExecGraph;
}

static void SetExecStatus(const JitCompileResults *c, const PyFrameWrapper &f, bool graph_preferred) {
  bool enable_statistics = c->conf()->GetBoolConfig(GraphJitConfig::kPerfStatistics);
  int graph_bytecode_min = c->conf()->getIntConfig(GraphJitConfig::kStaticGraphBytecodeMin);
  if (enable_statistics || (graph_bytecode_min > 0)) {
    auto globals = f.Globals();
    auto code = reinterpret_cast<PyObject *>(f.GetCode().ptr());
    PyObject_SetItem(globals.ptr(), code, graph_preferred ? Py_True : Py_False);
  }
}

static py::object CallCompiledResults(PyThreadState *tstate, const PyFrameWrapper &f, JitCompileResults *c) {
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_PRECOMPILE_ONLY) ||
      common::GetEnv("MS_DEV_PRECOMPILE_ONLY") == "1") {
    return py::none();
  }

  ValidateCompiledResults(c);

  auto packed_args = f.PackArgs();
  if (packed_args[1].ptr() != Py_None) {
    PyList_Append(packed_args[0].ptr(), packed_args[1].ptr());
  }

  py::object args = py::reinterpret_steal<py::object>(PyList_AsTuple(packed_args[0].ptr()));
  py::object kwvargs = packed_args[2];
  bool graph_preferred = PreferCallGraph(c, args);
  SetExecStatus(c, f, graph_preferred);
  py::object res;
  if (!graph_preferred) {
    res = CallCompiledCallable(tstate, f, c);
  } else if (!c->conf()->GetBoolConfig(GraphJitConfig::kCompileWithTry)) {
    res = CallGraph(c, args, kwvargs);
  } else {
    try {
      res = CallGraph(c, args, kwvargs);
    } catch (std::exception &e) {
      MS_LOG(WARNING) << "compile result has an error, de-optimization\n" << e.what();
      res = CallCompiledCallable(tstate, f, c);
      c->set_stat(JitCompileResults::NEVER_COMPILE);
    }
  }
  c->code()->Inc();

  // dump traceback
  if (c->conf()->GetBoolConfig(GraphJitConfig::kPrintTraceback)) {
    // dump all traceback for the root function
    GRAPH_JIT_LOG_F("%s\n", c->tbs()->Dump(true).c_str());
  }
  if (!PyErr_Occurred()) {
    c->tbs()->Clear();
  }
  return res;
}

static bool CheckGuard(JitCompileResults *c, const PyFrameWrapper &f) {
  TimeRecorder time_recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kCapture, runtime::ProfilerEvent::kCaptureGuard,
                                     "PIJitGuard");

  StaticAnalysisExceptionCleaner exception_cleaner;
  CaptureContext::DisableScope compiler_disable_scope;

  c->set_code(nullptr);
  std::map<size_t, PyObject *> cache;
  std::map<size_t, bool> success;
  std::map<size_t, bool> fail;
  OptOptionPtr opt = OptOption::CreateOptionByPoint(c);
  auto set = c->codehub()->GetOptTarget(opt);
  set = OptStrategy::MakeGuardListStrategyByFrame(set);
  for (size_t i = set.size(); i != 0; i--) {
    auto oc = set[i - 1];
    OptGuardPtr guard = oc->GetGuard();
    bool print_guard = c->conf()->GetBoolConfig(GraphJitConfig::kPrintGuard);
    if (guard != nullptr && guard->Check(f.frame(), print_guard, &cache, &success, &fail,
                                         c->conf()->GetBoolConfig(GraphJitConfig::kLogGuardPerf))) {
      c->set_code(oc);
      c->codehub()->UpdateOptTarget(opt, oc);
      break;
    }
  }
  for (auto item : cache) {
    Py_XDECREF(item.second);
  }
  MS_LOG(INFO) << "Check guard" << (c->code() != nullptr ? " success!" : " failed!");
  return c->code() != nullptr;
}

static bool JitCompileWithTry(PyThreadState *tstate, JitCompileResults *c) {
  TimeRecorder time_recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  MS_LOG(INFO) << "Start run PIJit";
  JitSyntaxLevelScope jit_syntax_level_scope;
  StaticAnalysisExceptionCleaner exception_cleaner;
  CaptureContext::DisableScope compiler_disable_scope;

  if (!c->conf()->GetBoolConfig(GraphJitConfig::kCompileWithTry)) {
    return JitCompile(tstate, c);
  }

  bool compiled = false;
  try {
    compiled = JitCompile(tstate, c);
  } catch (std::exception &e) {
    MS_LOG(ERROR) << "got an unexpected c++ error [" << e.what() << "]";
  }
  if (PyErr_Occurred()) {
    MS_LOG(ERROR) << "got an unexpected python error [" << py::error_already_set().what() << "]";
    PyErr_Clear();
    compiled = false;
  }
  return compiled;
}

py::tuple EliminateStubTensor(const py::tuple &args) {
  py::tuple new_args = py::reinterpret_steal<py::tuple>(PyTuple_New(args.size()));
  for (size_t idx = 0; idx < args.size(); idx++) {
    new_args[idx] = IsStubTensor(args[idx]) ? python_adapter::CallPyObjMethod(args[idx], "stub_sync") : args[idx];
  }
  return new_args;
}

// bellowing code is used for debugging code generate, and will be remove soon
py::object test_graph_ir_code_gen(const PyFrameWrapper &f) {
  auto co_wrapper = f.GetCode();
  PyObject *globals = f.Globals().ptr();
  PyCodeObject *co = co_wrapper.ptr();
  py::object f_locals = f.Locals();
  bool has_va;
  bool has_kw_va;
  int arg_cnt = co_wrapper.ArgCount(&has_va, &has_kw_va);

  auto func = py::reinterpret_steal<py::object>(PyFunction_New(reinterpret_cast<PyObject *>(co), globals));
  mindspore::pijit::Utils::DisFuncObject(func.ptr());
  auto byteCodeParser = std::make_shared<mindspore::pijit::ByteCodeParser>(func);
  mindspore::pijit::ir::FunctionNodePtr func_node = byteCodeParser->Parse();
  auto inliner = std::make_shared<mindspore::pijit::FuncInliner>(func_node);
  inliner->Run();

  py::list locals = py::reinterpret_steal<py::list>(PyMapping_Values(f_locals.ptr()));
  arg_cnt -= has_kw_va;
  py::tuple args = py::reinterpret_steal<py::tuple>(PyList_AsTuple(PyList_GetSlice(locals.ptr(), 0, arg_cnt)));
  py::dict kwargs = has_kw_va ? py::dict() : py::cast<py::dict>(locals[arg_cnt]);

  args = EliminateStubTensor(args);
  mindspore::pijit::AbstractTypeDeducer::Deduce(func_node, args, kwargs);
  func_node->Sort();
  std::cout << func_node->ToString() << std::endl;
  auto func_obj = mindspore::pijit::ByteCodeGenerator::GenFunction(func_node);
  mindspore::pijit::Utils::DisFuncObject(func_obj.ptr());
  if ((static_cast<unsigned int>(func_node->GetFlags()) & CO_VARARGS) != 0) {
    auto pos_cnt = args.size() - 1;
    auto var_vargs = py::cast<py::tuple>(args[pos_cnt]);
    auto new_args = py::reinterpret_steal<py::tuple>(PyTuple_New(pos_cnt + var_vargs.size()));
    size_t index = 0;
    std::for_each(args.begin(), args.end() - 1, [&index, &new_args](const py::handle &arg) {
      new_args[index] = arg;
      index++;
    });
    std::for_each(var_vargs.begin(), var_vargs.end(), [&index, &new_args](const py::handle &arg) {
      new_args[index] = arg;
      index++;
    });
    args = new_args;
  }
  auto res = py::reinterpret_steal<py::object>(PyObject_Call(func_obj.ptr(), args.ptr(), kwargs.ptr()));
  res.inc_ref();
  return res;
}

static py::object CodeHook(PyThreadState *tstate, JitCompileResults *c, EvalFrameObject *frame) {
  if (c->conf()->GetBoolConfig(GraphJitConfig::kTestGraphIR)) {
    return test_graph_ir_code_gen(PyFrameWrapper(frame));
  }
  PyCodeObject *co = PyFrameWrapper(frame).GetCode().ptr();
  bool just_compiled = false;
  switch (c->stat()) {
    case JitCompileResults::NEVER_COMPILE:
      break;
    case JitCompileResults::GRAPH_CAPTURED:
      if (c->conf()->GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
        break;
      }
    /* fallthrough */
    case JitCompileResults::GRAPH_CANDIDATE:
      MS_EXCEPTION_IF_CHECK_FAIL(c->origin_frame().frame() == nullptr || c->origin_frame().frame() == frame,
                                 "check recursive call compiling function");
      c->set_origin_frame(frame);
      if (c->conf()->GetBoolConfig(GraphJitConfig::kCompileWithoutCapture)) {
        c->set_stat(JitCompileResults::GRAPH_CAPTURED);
      }
      if (!JitCompileWithTry(tstate, c)) {
        c->set_stat(JitCompileResults::NEVER_COMPILE);
        break;
      }
      just_compiled = true;
    /* fallthrough */
    case JitCompileResults::GRAPH_CALLABLE: {
      if (CheckGuard(c, PyFrameWrapper(frame))) {
        c->set_origin_frame(nullptr);
        return CallCompiledResults(tstate, PyFrameWrapper(frame), c);
      }
      if (c->stat() == JitCompileResults::NEVER_COMPILE) {
        break;
      }
      if (!just_compiled) {
        c->set_stat(JitCompileResults::GRAPH_CANDIDATE);
        return CodeHook(tstate, c, frame);
      }
      MS_LOG(EXCEPTION) << "shouldn't reach here";
    }
    case JitCompileResults::GRAPH_BUILDING:
      MS_LOG(ERROR) << "recursive call, compiler call the code "
                    << std::string(py::str(reinterpret_cast<PyObject *>(co))) << " which is compiling";
      break;
    default:
      MS_LOG(EXCEPTION) << "shouldn't reach here";
      break;
  }
  c->set_origin_frame(nullptr);
  PyObject *res = _PyEval_EvalFrameDefault(tstate, frame, 0);
  return py::reinterpret_steal<py::object>(res);
}

PyObject *CallCodeHook(PyThreadState *tstate, EvalFrameObject *f, JitCompileResults *c) {
  py::object res;
  try {
    res = CodeHook(tstate, c, f);
  } catch (py::error_already_set &e) {
    e.restore();
  } catch (py::builtin_exception &e) {
    e.set_error();
  }
  return res.inc_ref().ptr();
}

py::list CollectGradientArguments(PyCodeObject *co, PyObject **fast_locals) {
  py::list arguments;
  bool has_va;
  bool has_kw_va;
  auto argc = PyCodeWrapper(co).ArgCount(&has_va, &has_kw_va);
  argc = argc - has_va - has_kw_va;

  // Collect Positional Arguments
  for (int index = 1; index < argc; index++) {
    arguments.append(py::cast<py::object>(fast_locals[index]));
  }

  // Collect Variable Arguments
  if (has_va) {
    auto var_args = py::cast<py::tuple>(fast_locals[argc++]);
    std::for_each(var_args.begin(), var_args.end(), [&arguments](const auto &arg) { arguments.append(arg); });
  }

  // Collect Variable Arguments
  if (has_kw_va) {
    auto kw_args = py::cast<py::dict>(fast_locals[argc++]);
    std::for_each(kw_args.begin(), kw_args.end(), [&arguments](const auto &item) { arguments.append(item.second); });
  }

  return arguments;
}

void AutoGrad(EvalFrameObject *frame, PyObject *ret) {
  // improve performance for infer
  if (kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kInferOnly)) {
    return;
  }
  PyFrameWrapper f(frame);
  auto co_wrapper = f.GetCode();
  // must have a return value and prim must have argument
  if (ret == nullptr || co_wrapper.FastLocalSize() <= 0) {
    return;
  }
  PyCodeObject *co = co_wrapper.ptr();
  const char *co_name = PyUnicode_AsUTF8(co->co_name);
  // the call function of primitive
  if (std::string(co_name) != "__call__") {
    return;
  }
  // only record primitive now
  PyObject **f_localsplus = f.FastLocal();
  if (f_localsplus[0] == nullptr) {
    return;
  }
  if (!py::isinstance<Primitive>(f_localsplus[0]) && !py::isinstance<PrimitivePy>(f_localsplus[0]) &&
      !py::isinstance<PrimitivePyAdapter>(f_localsplus[0])) {
    return;
  }
  // gradient info check
  if (!grad::FunctionNode::HasAttrReqGrad(ret) && !py::isinstance<py::tuple>(ret)) {
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(co->co_kwonlyargcount == 0, "Must not have kw only args.");
  auto inputs = CollectGradientArguments(co, f_localsplus);
  if (!std::any_of(inputs.begin(), inputs.end(),
                   [](const auto &input) { return grad::FunctionNode::IsRequiresGradient(input); })) {
    return;
  }
  grad::FunctionNode::RecordPrimitive(py::cast<py::object>(f_localsplus[0]), py::cast<py::object>(ret), inputs);
}

PyObject *EvalFrame(PY_FRAME_EVAL_FUNCTION_SIGNATURE) {
#ifdef PY_FRAME_EVAL_FUNCTION_DECLARE_THREAD_STATE
  PY_FRAME_EVAL_FUNCTION_DECLARE_THREAD_STATE();
#endif
  // exception handler
  if (exc != 0) {
    return _PyEval_EvalFrameDefault(ts, f, exc);
  }
  return PyFrameEvalHookManager::GetInstance()->RunHook(ts, f);
}
}  // namespace pijit
}  // namespace mindspore

namespace mindspore {

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION >= 7) && (PY_MINOR_VERSION <= 11)

py::bool_ pi_jit_enable() {
  PyInterpreterState *inter = PyInterpreterState_Main();
  _PyFrameEvalFunction prev = _PyInterpreterState_GetEvalFrameFunc(inter);
  _PyFrameEvalFunction def = _PyEval_EvalFrameDefault;
  if (prev != def) {
    return false;
  }
  mindspore::pijit::ensureInitialize();
  _PyInterpreterState_SetEvalFrameFunc(inter, mindspore::pijit::EvalFrame);
  return true;
}

py::bool_ pi_jit_disable() {
  PyInterpreterState *inter = PyInterpreterState_Main();
  _PyFrameEvalFunction prev = _PyInterpreterState_GetEvalFrameFunc(inter);
  _PyFrameEvalFunction def = _PyEval_EvalFrameDefault;
  if (prev != mindspore::pijit::EvalFrame) {
    return false;
  }
  _PyInterpreterState_SetEvalFrameFunc(inter, def);
  return true;
}

bool pi_jit_should_compile(const py::handle &funcHandle, const py::handle &tag, const py::handle &signature) {
  PyObject *func = funcHandle.ptr();
  PyObject *code = NULL;
  if (PyFunction_Check(func)) {
    code = PyFunction_GET_CODE(func);
  } else if (PyMethod_Check(func)) {
    func = PyMethod_GET_FUNCTION(func);
    code = PyFunction_GET_CODE(func);
  } else if (PyCode_Check(func)) {
    code = func;
  } else {
    return false;
  }
  mindspore::pijit::JitCompileResults *c = mindspore::pijit::CreateJitCompileResults(code);
  MS_LOG(DEBUG) << "mark to compile " << std::string(py::str(code));

  if (c == nullptr) {
    return false;
  }
  c->set_input_signature(py::reinterpret_borrow<py::object>(signature));
  auto new_config = mindspore::pijit::GraphJitConfig(py::reinterpret_borrow<py::object>(tag));
  // When switching between one-stage and two-stage, reset the config.
  if (c->conf()->GetBoolConfig(pijit::GraphJitConfig::kTraceFlag) !=
      new_config.GetBoolConfig(pijit::GraphJitConfig::kTraceFlag)) {
    c->set_code(nullptr);
    c->set_codehub(std::make_shared<pijit::OptCodeHub>());
  }
  if (c->stat() != mindspore::pijit::JitCompileResults::NEVER_COMPILE) {
    *c->conf() = new_config;
    return true;
  }

  pijit::PyCodeWrapper co(code);
  py::object bytes = co.Code();

  auto raw_code_size = PyBytes_GET_SIZE(bytes.ptr());
  std::string raw_func_info_name = py::str(code).cast<std::string>();
  std::string raw_func_name = "";
  if (PyFunction_Check(func)) {
    const char *module_name = PyUnicode_AsUTF8(PyFunction_GET_MODULE(func));
    const char *s = strchr(module_name, '.');
    std::string top_module = s ? std::string(module_name, s - module_name) : module_name;
    mindspore::pijit::kPIJitConfigDefault.AddAllowedInlineModules(top_module);

    raw_func_name = mindspore::pijit::Utils::GetPyName(reinterpret_cast<PyFunctionObject *>(func)->func_qualname);
  }

  c->set_stat(mindspore::pijit::JitCompileResults::GRAPH_CANDIDATE);
  *c->conf() = new_config;
  *c->tbs() = mindspore::pijit::Traceback(raw_func_name, raw_func_info_name, raw_code_size);
  return true;
}
#else

py::bool_ pi_jit_enable() {
  MS_LOG(ERROR) << "PiJit not support this python version " << PY_MAJOR_VERSION << '.' << PY_MINOR_VERSION
                << " only support on python3.7, python3.8, python3.9, python3.10";
  return py::bool_(false);
}
py::bool_ pi_jit_disable() { return py::bool_(false); }
py::bool_ pi_jit_should_compile(const py::object &func, const py::object &tag, const py::object &signature) {
  return py::bool_(false);
}

#endif

static py::object ConvertCodeExtra(mindspore::pijit::JitCompileResults *c) {
  if (c->code() == nullptr) {
    return py::object();
  }
  PyCodeObject *compiled_code = c->code()->GetPythonCode();
  auto compiled_func = c->code()->GetNativeFunc();
  auto guard = c->code()->GetGuard();
  if (compiled_func == nullptr && compiled_code == nullptr) {
    return py::object();
  }
  py::dict code;
  if (compiled_code != nullptr) {
    PyDict_SetItemString(code.ptr(), "compiled_code_", reinterpret_cast<PyObject *>(compiled_code));
  }
  if (compiled_func != nullptr) {
    PyDict_SetItemString(code.ptr(), "phase_", py::str(c->code()->GetPhase()).ptr());
  }
  if (guard != nullptr && !guard->IsEmpty()) {
    PyDict_SetItemString(code.ptr(), "guard_", py::str(guard->ToString()).ptr());
  }
  PyDict_SetItemString(code.ptr(), "call_count_", py::int_(c->code()->Count()).ptr());
  return code;
}

py::object get_code_extra(const py::object &func) {
  py::object code = mindspore::pijit::GetPyCodeObject(func);
  if (code.ptr() == nullptr) {
    return py::none();
  }
  auto c = mindspore::pijit::GetJitCompileResults(code.ptr());
  if (c == nullptr) {
    return py::none();
  }

  constexpr const char *stat_str[] = {
    "NEVER_COMPILE", "GRAPH_CANDIDATE", "GRAPH_CAPTURED", "GRAPH_BUILDING", "GRAPH_CALLABLE",
  };

  py::dict result;
  py::object compiled_code = ConvertCodeExtra(c);
  if (compiled_code.ptr() != nullptr) {
    PyDict_SetItemString(result.ptr(), "code", compiled_code.ptr());
  }
  PyDict_SetItemString(result.ptr(), "stat", py::str(stat_str[c->stat()]).ptr());
  PyDict_SetItemString(result.ptr(), "compile_count_", py::int_(c->compile_count()).ptr());
  PyDict_SetItemString(result.ptr(), "break_count_", py::int_(c->break_count()).ptr());
  return result;
}

size_t FunctionId(const py::object &callable) {
  // filter special cpp function
  auto py_cfunction_filter = [](PyObject *op) -> void * {
    // pybind11::cpp_function::dispatcher;
    static PyCFunction pybind_dispatcher = PyCFunction_GET_FUNCTION(py::cpp_function([]() {}).ptr());
    PyCFunction result = PyCFunction_GET_FUNCTION(op);
    return result == pybind_dispatcher ? op : reinterpret_cast<void *>(result);
  };
  PyObject *op = callable.ptr();
  if (PyMethod_Check(op)) {
    op = PyMethod_GET_FUNCTION(op);
  }
  if (PyInstanceMethod_Check(op)) {
    op = PyInstanceMethod_GET_FUNCTION(op);
  }
  void *result = op;
  if (PyCFunction_Check(op)) {
    // types.BuiltinFunctionType = type(len) same as types.BuiltinMethodType = type(list().append)
    result = py_cfunction_filter(op);
  } else if (Py_IS_TYPE(op, &PyMethodDescr_Type)) {
    // types.MethodDescriptorType = type(list.append)
    PyCFunction func = reinterpret_cast<PyMethodDescrObject *>(op)->d_method->ml_meth;
    result = reinterpret_cast<void *>(func);
  } else if (Py_IS_TYPE(op, &PyWrapperDescr_Type)) {
    // types.WrapperDescriptorType = type(object.__init__)
    result = reinterpret_cast<PyWrapperDescrObject *>(op)->d_wrapped;
  } else if (Py_IS_TYPE(op, &_PyMethodWrapper_Type)) {
    // types.WrapperDescriptorType = type(object().__str__)
    PyObject *self = PyObject_GetAttrString(op, "__self__");
    PyObject *attr = PyObject_GetAttrString(op, "__name__");
    PyObject *descr = PyObject_GetAttr(reinterpret_cast<PyObject *>(Py_TYPE(self)), attr);
    result = reinterpret_cast<PyWrapperDescrObject *>(descr)->d_wrapped;
    Py_DECREF(self);
    Py_DECREF(attr);
    Py_DECREF(descr);
  }
  return reinterpret_cast<size_t>(result);
}

void PIJitSetContext(py::args va, py::kwargs kw) {
  auto ctx = pijit::CaptureContext::GetInstance();
  ctx->SetContext(va, kw);
}

}  // namespace mindspore
