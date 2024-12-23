/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/pi/graph_capture/graph_analyzer.h"
#include <algorithm>
#include <list>
#include <unordered_set>
#include <utility>
#include <string>
#include <vector>
#include "pipeline/jit/pi/pi_jit_config.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/graph_capture/graph.h"
#include "pipeline/jit/pi/graph_capture/special_func_infer.h"
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include "pipeline/jit/pi/graph_capture/side_effect.h"

#define ADD_NODE(container, node)                                                   \
  do {                                                                              \
    auto _tmp_node = (node);                                                        \
    (container).push_back(_tmp_node);                                               \
    MS_LOG(INFO) << "Add node to " #container " [" << _tmp_node->ToString() << "]"; \
  } while (0)

namespace mindspore {
namespace pijit {

extern TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);

const int kMsFlagSet = AObject::kMsFlagGradFunc | AObject::kMsFlagStandardFunc | AObject::kMsFlagShardFunc |
                       AObject::kMsFlagVmapFunc | AObject::kMsFlagJitFunc;
static bool IsRepeatWithoutSideEffect(ValueNode *v, bool repeat_attr_item_access);

static bool CheckBuildTupleRepeatable(ValueNode *value, bool repeat_attr_item_access) {
  for (auto i : value->getInputs()) {
    if (i->GetOpcode() == BUILD_TUPLE || !IsRepeatWithoutSideEffect(i, repeat_attr_item_access)) {
      return false;
    }
  }
  return true;
}

static bool CheckBuildSliceRepeatable(const std::vector<ValueNode *> &inputs, bool repeat_attr_item_access) {
  for (auto i : inputs) {
    if (i->GetOpcode() != LOAD_CONST) {
      return false;
    }
  }
  return true;
}

// These are operations that are repeated and have no side effects.
static bool IsRepeatWithoutSideEffect(ValueNode *v, bool repeat_attr_item_access) {
  if (IsNonLocalValue(v)) {
    return true;
  }

  AObject::Type type = v->GetVobj() ? v->GetVobj()->GetType() : AObject::kTypeAnyValue;
  auto opcode = v->GetOpcode();
  if (opcode == BUILD_TUPLE) {
    return CheckBuildTupleRepeatable(v, repeat_attr_item_access);
  } else if (opcode == BUILD_SLICE) {
    // NOTE: mindspore can't resolve call 'slice' class
    return CheckBuildSliceRepeatable(v->getInputs(), repeat_attr_item_access);
  } else if (opcode == BINARY_SUBSCR || opcode == LOAD_ATTR) {
    return type == AObject::kTypeAnyValue ? false : repeat_attr_item_access;
  } else if (opcode == BUILD_MAP) {
    if (type == AObject::kTypeDict) {
      AbstractDict *d = static_cast<AbstractDict *>(v->GetVobj());
      return d->size() == 0 || d->KeyType() != AObject::kTypeAnyValue;
    }
    return false;
  }
  return false;
}

namespace {
/**
 * mindspore func_graph assume these unsupported value is constant, so it same as global.
 * avoid parameter unsupported error by global
 */
bool ValidateGraphParameters(ValueNode *node) {
  static const std::set<AObject::Type> unsupported_parameter = {
    AObject::kTypeAnyValue,  AObject::kTypeFunction,      AObject::kTypeBoundMethod,
    AObject::kTypePrimitive, AObject::kTypeMetaFuncGraph, AObject::kTypeCell,
  };
  AObject *info = node->GetVobj();
  if (info == nullptr) {
    return false;
  }
  return unsupported_parameter.find(info->GetType()) == unsupported_parameter.end();
}
}  // namespace

bool GraphAnalyzer::ProduceInterpretValue(ValueNode *v) {
  bool repeat_op = graph_->Config().GetBoolConfig(GraphJitConfig::kEnableOptimizeForAttrItem);
  auto &locals = GetCaptureInfo().interpret_.values;
  auto &values = GetCaptureInfo().captured_.values;
  for (auto i : v->getInputs()) {
    if (IsNonLocalValue(i) || locals.find(i) != locals.end()) {
      continue;
    }
    if (values.find(i) == values.end()) {
      MS_LOG(INTERNAL_EXCEPTION) << "capture info can't find the value [" << i->ToString() << "]";
    }
    if (!IsRepeatWithoutSideEffect(i, repeat_op)) {
      return false;
    }
    // duplicate some operations if possible
    if (ProduceInterpretValue(i)) {
      continue;
    }
    return false;
  }
  AddToEscaped(v);
  return true;
}

// if operation can't be repeated, or block has attr access side effect
// can't reorder attr access op, must be interpret all attr, item access operation
static bool CheckAttrItemSupport(ValueNode *v, bool repeat_op) {
  int op = v->GetOpcode();
  AObject::Type type = v->input(0)->GetVobj() ? v->input(0)->GetVobj()->GetType() : AObject::kTypeAnyValue;
  // item access
  if (op == BINARY_SUBSCR) {
    return type != AObject::kTypeAnyValue;
  }
  // attr access
  if (type == AObject::kTypeAnyValue || type == AObject::kTypeBoundMethod) {
    return false;
  }
  if (type == AObject::kTypeTensor && !FindTensorName(v->GetName())) {
    return false;
  }
  return true;
}

static bool IsSideEffect(ValueNode *v) {
  static const std::set<std::string> funcs = {"assign", "Assign"};
  static const std::set<int> unsupported_op = {
    STORE_DEREF,  DELETE_DEREF,  STORE_GLOBAL, DELETE_GLOBAL, STORE_ATTR, DELETE_ATTR,
    STORE_SUBSCR, DELETE_SUBSCR, IMPORT_STAR,  RAISE_VARARGS, RERAISE,    FORMAT_VALUE,
  };
  Opcode opcode(v->GetOpcode());
  if (opcode.MayDelete()) {
    return false;
  }
  if (opcode.IsCall()) {
    AObject *f = v->input(0)->GetVobj();
    if (f == nullptr) {
      return true;
    }
    if (f->TestMsFlag(AObject::kMsFlagGradFunc)) {
      return false;
    }
    return funcs.find(GetFuncName(f->GetPyObject())) != funcs.end();
  }
  return unsupported_op.find(v->GetOpcode()) != unsupported_op.end();
}

bool GraphAnalyzer::HandleCallableToGraph(AObject *f) {
  static bool known_type[AObject::kTypeCount] = {false};
  if (known_type[AObject::kTypePrimitive] == false) {
    known_type[AObject::kTypePrimitive] = true;
    known_type[AObject::kTypeCell] = true;
    known_type[AObject::kTypeMetaFuncGraph] = true;
    known_type[AObject::kTypePrimitiveFunction] = true;
  }
  if (f == nullptr) {
    return false;
  }
  // don't pass unknown callable to graph
  bool is_known_func = known_type[f->GetType()] || CheckJitConstexpr(f->GetPyObject());
  bool is_ms_support_func = f->TestMsFlag(kMsFlagSet);
  if (!is_known_func && !is_ms_support_func) {
    return false;
  }
  if (f->GetType() == AObject::kTypePrimitive) {
    PyTypeObject *tp = f->GetTypeObject();
    std::string name = (tp && tp->tp_name ? tp->tp_name : "");
    if (name == "Assign") {
      return false;
    }
  }
  return true;
}

bool GraphAnalyzer::AddToCaptured(ValueNode *v) {
  if (IsNonLocalValue(v)) {
    return true;
  }
  if (v->GetVobj() && v->GetVobj()->TestMsFlag(AObject::kMsFlagGradFunc)) {
    GetCaptureInfo().has_grad_ = true;
    GetCaptureInfo().captured_.values.insert(v);
    GetCaptureInfo().captured_.operations.push_back(v);
    return true;
  }

  int op = v->GetOpcode();
  bool repeat_op = graph_->Config().GetBoolConfig(GraphJitConfig::kEnableOptimizeForAttrItem);
  if ((op == LOAD_ATTR || op == BINARY_SUBSCR) && !CheckAttrItemSupport(v, repeat_op)) {
    return false;
  }

  bool is_call_op = Opcode(v->GetOpcode()).IsCall();
  if (is_call_op) {
    AObject *f = v->input(0)->GetVobj();
    bool can_pass = HandleCallableToGraph(f);
    if (!can_pass) {
      return false;
    }
    GetCaptureInfo().has_grad_ |= f->TestMsFlag(AObject::kMsFlagGradFunc);
  }

  auto &locals = GetCaptureInfo().interpret_.values;  // interpret values
  auto &values = GetCaptureInfo().captured_.values;   // graph produced values
  for (auto i : v->getInputs()) {
    bool produced_in_graph = values.find(i) != values.end() || IsNonLocalValue(i);
    MS_EXCEPTION_IF_CHECK_FAIL(produced_in_graph || locals.find(i) != locals.end(),
                               "check values order, all input must be generate before this value " + i->ToString());
    if (i->GetVobj() == nullptr) {
      return false;
    }
    AObject::Type type = i->GetVobj()->GetType();
    PyTypeObject *tp = i->GetVobj()->GetTypeObject();
    if (type == AObject::kTypeAnyValue && !IsMsClass(reinterpret_cast<PyObject *>(tp))) {
      // don't pass unknown object to graph
      return false;
    }
    if (type == AObject::kTypeCell && !is_call_op) {
      // don't pass a cell object that not call to graph.
      return false;
    }
  }

  GetCaptureInfo().captured_.values.insert(v);
  GetCaptureInfo().captured_.operations.push_back(v);
  return true;
}

void GraphAnalyzer::AddToEscaped(ValueNode *v) {
  GetCaptureInfo().interpret_.values.insert(v);
  GetCaptureInfo().interpret_.operations.push_back(v);
}

extern TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);

bool GraphAnalyzer::TryToCapture(AbstractNode *n) {
  ValueNode *v = static_cast<ValueNode *>(n);
  if (IsNonLocalValue(v)) {
    return true;
  }
  if (graph_->GetSideEffect()->IsRecord(v) && !graph_->GetSideEffect()->NeedTrack(v)) {
    return true;
  }
  bool is_side_effect = IsSideEffect(v);
  if (!is_side_effect && AddToCaptured(v)) {
    return true;
  }
  if (!GetCaptureInfo().captured_.values.empty() && is_side_effect) {
    return false;
  }
  if (ProduceInterpretValue(v)) {
    return true;
  }
  if (!HasTensorOperation()) {
    CleanCapturedValue();
    AddToEscaped(v);
    return true;
  }

  if (v->GetGraph() != nullptr && this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    GRAPH_JIT_LOG_F("capture failed, operations is unsupported [%s] at [%U: %d]", v->ToString().c_str(),
                    v->GetGraph()->GetCodeObj()->co_filename, v->GetLineNo());
    GRAPH_JIT_LOG_F("parameters");
    for (auto &i : v->getInputs()) {
      PyObject *op = i->GetVobj() ? i->GetVobj()->GetPyObject().ptr() : nullptr;
      GRAPH_JIT_LOG_F("%s", op ? AObject::ToString(op).c_str() : "NULL");
    }
  }

  MS_LOG(DEBUG) << "---operation that depend on the graph outputs, break graph---";
  return false;
}

bool GraphAnalyzer::AnalyzeCall(CallNode *call_node) {
  if (call_node->GetSubGraph() == nullptr) {
    return false;
  }
  if (call_node->GetInlineReason() != InlineReason::kInline) {
    return false;
  }

  Graph *g = call_node->GetGraph();

  CapturedInfo back_up = info_;
  const auto &p = call_node->GetParams();
  // capture parameter handle operations
  auto iter = std::find_if(p.begin(), p.end(), [this](ValueNode *i) { return !this->TryToCapture(i); });
  // capture sub-graph
  if (iter == p.end() && AnalyzeRecursive(call_node->GetSubGraph())) {
    return true;
  }
  info_ = back_up;
  g->StopTraceAt(call_node->bci(), StopTraceReason::kStopTraceDataDependsOnGraphOut);
  return false;
}

bool GraphAnalyzer::AnalyzeRecursive(Graph *g) {
  for (auto n : g->GetTracedNodes()) {
    int bci = static_cast<ValueNode *>(n)->bci();
    if (n->GetType() == AbstractNode::Call && AnalyzeCall(static_cast<CallNode *>(n))) {
      continue;
    }
    if (bci != -1 && g->GetStopTraceBci() == bci) {
      return false;
    }
    if (!TryToCapture(n)) {
      g->StopTraceAt(bci, StopTraceReason::kStopTraceDataDependsOnGraphOut);
      return false;
    }
  }
  return true;
}

void GraphAnalyzer::CollectCapturedInputs() {
  auto &locals = GetCaptureInfo().interpret_.values;
  auto &values = GetCaptureInfo().captured_.values;
  mindspore::CompactSet<ValueNode *> inputs;
  for (ValueNode *i : GetCaptureInfo().captured_.operations) {
    for (auto input : i->getInputs()) {
      if (values.find(input) != values.end() || IsNonLocalValue(input)) {
        continue;
      }
      MS_EXCEPTION_IF_CHECK_FAIL(locals.find(input) != locals.end(), "check graph input");
      inputs.insert(input);
    }
  }
  GetCaptureInfo().captured_.inputs = {inputs.begin(), inputs.end()};
}

void GraphAnalyzer::UseDefAnalyze() {
  // UD analyze: alive nodes analysis
  std::vector<ValueNode *> aliveLocals = GetAliveLocals(graph_);
  if (aliveLocals.empty()) {
    return;
  }
  while (!AnalyzeAliveLocals(aliveLocals)) {
    aliveLocals = GetAliveLocals(graph_);
  }
}

void GraphAnalyzer::OptimizeSideEffectRecord() const {
  if (graph_->GetSideEffect()->IsEmpty()) {
    return;
  }
  auto alive = graph_->CollectAliveNode(graph_->GetStopTraceBci());
  auto side_effect_required_size = graph_->GetSideEffect()->GetRequiredNodes().size();
  auto size = alive.size() - side_effect_required_size;
  graph_->GetSideEffect()->Optimize({alive.begin(), alive.begin() + size});
}

void GraphAnalyzer::ResetSideEffectRecord() const {
  // if break point is changed, rollback graph nodes(only reset break bci) and side-effect record
  int break_bci = graph_->GetStopTraceBci();
  if (graph_->GetSideEffect()->IsEmpty()) {
    return;
  }
  const auto &nodes = graph_->GetTracedNodes();
  if (break_bci == -1) {
    graph_->GetSideEffect()->ResetRecord({nodes.begin(), nodes.end()});
  } else {
    auto iter = std::find_if(nodes.begin(), nodes.end(), [&break_bci](ValueNode *i) { return i->bci() > break_bci; });
    graph_->GetSideEffect()->ResetRecord({nodes.begin(), iter});
  }
  OptimizeSideEffectRecord();  // after reset record, rollback side-effect record status
}

void GraphAnalyzer::Analyze() {
  OptimizeSideEffectRecord();  // first optimize, remove dead local side-effects and it's required nodes

  const FrameStates &enter_frame = graph_->GetFrame(0);
  GetCaptureInfo().interpret_.values.insert(enter_frame.GetLocals().begin(), enter_frame.GetLocals().end());
  AnalyzeRecursive(graph_);
  if (!HasTensorOperation()) {
    CleanCapturedValue();
  }
  UseDefAnalyze();
  ResetSideEffectRecord();  // if rollback nodes, rollback side-effects

  CollectCapturedAndInterpret();
  CollectGraphInputs();

  need_interpret_ = true;

  if (graph_->GetStopTraceBci() != -1 || !GetCaptureInfo().interpret_.operations.empty()) {
    return;
  }
  bool support_ret = graph_->GetRetVal()->GetVobj() && graph_->GetRetVal()->GetVobj()->IsMindSporeSupportedType();
  if (!support_ret) {
    return;
  }
  PyCodeObject *co = graph_->GetCodeObj();
  const auto &args = enter_frame.GetLocals();
  int argc = co->co_argcount + co->co_kwonlyargcount;
  // check all parameters is graph supported, but here not check variable arguments
  auto end = args.begin() + argc;
  auto iter = std::find_if(args.begin(), end, [](ValueNode *i) { return !ValidateGraphParameters(i); });
  if (iter == end) {
    need_interpret_ = false;
  }
  need_interpret_ |= !graph_->GetSideEffect()->IsEmpty();
}

FrameStates buildLastFrame(Graph *g) { return g->GetFrame(g->GetStopTraceBci()); }

std::vector<ValueNode *> GraphAnalyzer::GetAliveLocals(Graph *g) {
  int bci = g->GetStopTraceBci();
  if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    GRAPH_JIT_LOG_F("UD analyze: enter GetAliveLocals bci %d", bci);
  }
  std::vector<ValueNode *> outputs = g->CollectAliveNode(bci);
  mindspore::CompactSet<ValueNode *> uniques;
  for (auto output : outputs) {
    uniques.insert(output);
  }
  outputs.assign(uniques.begin(), uniques.end());

  if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    GRAPH_JIT_LOG_F("UD analyze: alive node size : %ld", outputs.size());
    for (auto node : outputs) {
      if (node) {
        GRAPH_JIT_LOG_F("UD analyze: alive node: %s", node->ToString().c_str());
      }
    }
  }
  return outputs;
}

bool GraphAnalyzer::AnalyzeAliveLocals(std::vector<ValueNode *> aliveNodes) {
  bool isAllNodesSupportOutput = true;
  for (auto node : aliveNodes) {
    AObject *o = node->GetVobj();
    bool supported_type = o && o->IsMindSporeSupportedType();
    if (supported_type) {
      continue;
    }
    auto capturedLocals = info_.captured_.operations;
    if (std::find(capturedLocals.begin(), capturedLocals.end(), node) == capturedLocals.end()) {
      continue;
    }
    if (ProduceInterpretValue(node)) {
      continue;  // try to construct this value in python
    }

    if (!HasTensorOperation()) {
      CleanCapturedValue();
      break;
    }
    /**
     * produce the values if it can be produced by interpret values before call graph
     * e.g
     *   return parameter.some_attribute
     *   return build_map(parameters, other_constants)
     */
    if (ProduceInterpretValue(node)) {
      continue;
    }
    /**
     * produce the values if it can be produced by interpret values and graph outputs after call graph
     * e.g
     *   graph_outputs = call_graph()
     *   return graph_outputs[0].dtype, graph_outputs[1].asnumpy
     * ...save alive nodes and reconstruct these values when generated the code
     */

    //  reset break graph point
    isAllNodesSupportOutput = false;
    int new_break_point = node->bci();
    auto curNode = node;
    MS_EXCEPTION_IF_CHECK_FAIL(new_break_point != -1, "break point cannot be -1");
    MS_EXCEPTION_IF_NULL(curNode->GetGraph());
    if (this->graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
      GRAPH_JIT_LOG_F("reset break point: %d", new_break_point);
    }
    this->graph_->StopTraceAt(new_break_point, StopTraceReason::kStopTraceDataDependsOnGraphOut);

    // re-collect captured info
    info_.clear();
    const FrameStates &enter_frame = graph_->GetFrame(0);
    GetCaptureInfo().interpret_.values.insert(enter_frame.GetLocals().begin(), enter_frame.GetLocals().end());
    (void)AnalyzeRecursive(graph_);
    break;
  }
  return isAllNodesSupportOutput;
}

static bool SkipSpecialFuncOrPrimitive(const py::object &callable) {
  if (callable.ptr() == nullptr) {
    return false;
  }
  if (CheckJitConstexpr(callable) || CheckMSConstexpr(callable)) {
    return true;
  }
  if (IsPrimitiveType<true>(Py_TYPE(callable.ptr()))) {
    std::string name = callable.attr("name").cast<std::string>();
    return GetSpecialPrimitiveInferFunc().find(name) != GetSpecialPrimitiveInferFunc().end();
  }
  return false;
}

bool GraphAnalyzer::HasTensorOperation() const {
  bool has_tensor_cal = false;
  for (auto i : info_.captured_.values) {
    AObject *value = i->GetVobj();
    Opcode op(i->GetOpcode());
    if (op.IsCall()) {
      if (SkipSpecialFuncOrPrimitive(i->input(0)->GetVobj()->GetPyObject())) {
        continue;
      }
      if (value->GetType() == AObject::kTypeCFunction) {
        continue;
      }
      return true;
    }
    if (op.IsBinaryMath() && value->GetType() == AObject::kTypeTensor) {
      return true;
    }
  }
  return has_tensor_cal;
}

void GraphAnalyzer::CapturedInfo::Info::clear() {
  values.clear();
  inputs.clear();
  operations.clear();
  outputs.clear();
}

void GraphAnalyzer::CapturedInfo::GraphInputs::clear() {
  args.clear();
  globals.clear();
  vargs = nullptr;
  kwargs = nullptr;
}

void GraphAnalyzer::CapturedInfo::clear() {
  captured_.clear();
  interpret_.clear();
  outputs_optimize_.clear();
  graph_inputs_.clear();
}

std::string GraphAnalyzer::CapturedInfo::Info::ToString() {
  std::stringstream s;
  s << "inputs: {" << std::endl;
  for (auto i : inputs) {
    s << "  " << i->ToString() << std::endl;
  }
  s << "}" << std::endl;
  s << "operations: {" << std::endl;
  for (auto i : operations) {
    s << "  " << i->ToString() << std::endl;
  }
  s << "}" << std::endl;
  s << "outputs: {" << std::endl;
  for (auto i : outputs) {
    s << "  " << i->ToString() << std::endl;
  }
  s << "}" << std::endl;
  return s.str();
}

std::string GraphAnalyzer::CapturedInfo::GraphInputs::ToString() {
  std::stringstream s;
  s << "globals: ";
  for (auto i : globals) {
    s << i->ToString() << "\n";
  }
  s << "args: \n";
  for (auto i : args) {
    s << i->ToString() << "\n";
  }
  s << "vargs: ";
  if (vargs != nullptr) {
    s << vargs->ToString();
  }
  s << "\n";
  s << "kwargs: ";
  if (kwargs != nullptr) {
    s << kwargs->ToString();
  }
  s << "\n";
  return s.str();
}

std::string GraphAnalyzer::CapturedInfo::ToString() {
  std::stringstream s;
  s << "1. captured_ info: \n";
  s << captured_.ToString();
  s << "2. outputs_optimize_ info: \n";
  s << outputs_optimize_.ToString();
  s << "3. interpret_ info: \n";
  s << interpret_.ToString();
  s << "4. graph_inputs_: \n";
  s << graph_inputs_.ToString();
  s << "5. has_grad_: " << has_grad_ << "\n";
  return s.str();
}

void GraphAnalyzer::CleanCapturedValue() {
  auto &locals = info_.interpret_.values;
  for (auto i : info_.captured_.operations) {
    if (locals.find(i) == locals.end()) {
      locals.insert(i);
      info_.interpret_.operations.emplace_back(i);
    }
  }
  info_.captured_.values.clear();
  info_.captured_.operations.clear();
}

static std::vector<ValueNode *> CollectGraphOutputs(const mindspore::CompactSet<ValueNode *> &interpret,
                                                    const std::vector<ValueNode *> &alive) {
  std::vector<ValueNode *> outputs;
  for (auto i : alive) {
    if (interpret.find(i) == interpret.end() && !IsNonLocalValue(i)) {
      outputs.push_back(i);
    }
  }
  return outputs;
}

void GraphAnalyzer::CollectCapturedAndInterpret() {
  CollectCapturedInputs();
  int break_bci = graph_->GetStopTraceBci();
  std::vector<ValueNode *> alive_nodes = graph_->CollectAliveNode(break_bci, &alive_locals_);

  GetCaptureInfo().captured_.outputs = CollectGraphOutputs(GetCaptureInfo().interpret_.values, alive_nodes);
  GetCaptureInfo().interpret_.inputs = graph_->GetFrame(0).GetLocals();
  GetCaptureInfo().interpret_.outputs = std::move(alive_nodes);
}

void GraphAnalyzer::CollectGraphInputs() {
  PyCodeObject *co_ = graph_->GetCodeObj();
  auto &interpret_ = GetCaptureInfo().interpret_;
  auto &captured_ = GetCaptureInfo().captured_;
  auto &graph_inputs = GetCaptureInfo().graph_inputs_;

  // NOTE: if *vargs is cell variable, it is not parameter node
  MS_EXCEPTION_IF_CHECK_FAIL(co_->co_nlocals == static_cast<int>(interpret_.inputs.size()),
                             "interpret inputs must be same as locals");

  ValueNode *vargs = nullptr;
  ValueNode *kwargs = nullptr;
  int arg_index = co_->co_argcount + co_->co_kwonlyargcount;
  if ((co_->co_flags & CO_VARARGS) && interpret_.inputs[arg_index] != &ValueNode::kUnboundLocal) {
    vargs = interpret_.inputs[arg_index];
  }
  arg_index += (IntToSize(co_->co_flags) & CO_VARARGS) != 0;
  if ((IntToSize(co_->co_flags) & CO_VARKEYWORDS) && interpret_.inputs[arg_index] != &ValueNode::kUnboundLocal) {
    kwargs = interpret_.inputs[arg_index];
  }

  // Identify parameters and global variables
  for (auto input : captured_.inputs) {
    if (input == graph_inputs.vargs) {
      graph_inputs.vargs = vargs;
    } else if (input == graph_inputs.kwargs) {
      graph_inputs.kwargs = kwargs;
    } else if (ValidateGraphParameters(input)) {
      graph_inputs.args.push_back(input);
    } else {
      graph_inputs.globals.push_back(input);
    }
  }

  size_t inputs_count = captured_.inputs.size();
  captured_.inputs = graph_inputs.args;
  if (graph_inputs.vargs != nullptr) {
    captured_.inputs.push_back(graph_inputs.vargs);
  }
  if (graph_inputs.kwargs != nullptr) {
    captured_.inputs.push_back(graph_inputs.kwargs);
  }
  captured_.inputs.insert(captured_.inputs.end(), graph_inputs.globals.begin(), graph_inputs.globals.end());
  MS_EXCEPTION_IF_CHECK_FAIL(inputs_count == captured_.inputs.size(), "error parameters");
}

void MindGraphAnalyzer::CollectCapturedInputs() {
  GetCaptureInfo().captured_.inputs = graph_->prepare().inputs_;
  // check inputs is valid if break point is rollback
}

void MindGraphAnalyzer::Analyze() {
  auto collect_trace_nodes = [this]() {
    const auto &nodes = graph_->GetTracedNodes();
    if (graph_->GetStopTraceBci() == -1) {
      return nodes;
    }
    std::vector<ValueNode *> result;
    for (const auto &node : nodes) {
      if (node->bci() >= graph_->GetStopTraceBci()) {
        break;
      }
      result.push_back(node);
    }
    return result;
  };

  CollectClosureSideEffect();
  OptimizeSideEffectRecord();

  auto origin_stop_bci = graph_->GetStopTraceBci();
  // assume all values is captured to func_graph
  GetCaptureInfo().captured_.operations = collect_trace_nodes();
  UseDefAnalyze();

  auto mind_graph_builder = std::static_pointer_cast<MindGraphBuilder>(graph_builder_);
  MS_EXCEPTION_IF_NULL(mind_graph_builder);
  auto func_graph_builder = mind_graph_builder->FGBuilder();
  if (func_graph_builder->graph() == nullptr) {
    // Graph build failed, add all nodes to ordered_escaped_locals.
    PyCodeWrapper co(graph_->GetCodeObj());
    if (origin_stop_bci == -1) {
      MS_LOG(INFO) << "no graph in " << py::str(reinterpret_cast<PyObject *>(co.ptr()));
    } else {
      MS_LOG(INFO) << "no graph captured, trace break at " << co.FileName() << ", line "
                   << PyCode_Addr2Line(co.ptr(), origin_stop_bci);
    }
    graph_->StopTraceAt(origin_stop_bci, StopTraceReason::kStopTraceDataDependsOnGraphOut);
    need_interpret_ = true;
    GetCaptureInfo().clear();

    GetCaptureInfo().interpret_.inputs = graph_->GetFrame(0).GetLocals();
    GetCaptureInfo().interpret_.operations = collect_trace_nodes();
    GetCaptureInfo().interpret_.outputs = graph_->CollectAliveNode(origin_stop_bci, &alive_locals_);
    // remove side-effect node
    auto is_remove = [this](ValueNode *node) {
      const auto &rec = this->graph_->GetSideEffect();
      return rec->IsRecord(node) && !rec->NeedTrack(node);
    };
    auto *ops = &GetCaptureInfo().interpret_.operations;
    ops->erase(std::remove_if(ops->begin(), ops->end(), is_remove), ops->end());
    return;
  }

  ResetSideEffectRecord();
  CollectCapturedAndInterpret();
  CollectGraphInputs();

  need_interpret_ = true;
  if (graph_->GetStopTraceBci() != -1 || !GetCaptureInfo().interpret_.operations.empty()) {
    return;
  }
  bool support_ret = graph_->GetRetVal()->GetVobj() && graph_->GetRetVal()->GetVobj()->IsMindSporeSupportedType();
  if (!support_ret) {
    return;
  }
  int param_index = 0;
  for (const auto &node : GetCaptureInfo().captured_.inputs) {
    if (node->GetType() != ValueNode::Param /* LOAD_DEREF, LOAD_GLOBAL, LOAD_ATTR, BINARY_SUBSCR */
        || node->GetOparg() != param_index /* missing arguments */) {
      return;
    }
    param_index++;
  }
  need_interpret_ = !graph_->GetSideEffect()->IsEmpty() || !GetCaptureInfo().outputs_optimize_.operations.empty();
}

void MindGraphAnalyzer::CollectClosureSideEffect() {
  if (graph_->GetFrame(0).GetClosures().empty()) {
    return;
  }
  const std::vector<CellVarNode *> &closures = graph_->GetFrame(0).GetClosures();
  std::vector<CellVarNode *> nodes;
  if (graph_->GetStopTraceBci() == -1) {
    // If no graph break, we only need to restore free-variables. Because cell-variables are equivalent to
    // local-variables and do not need to be restored.
    int cellvar_size = PyCodeWrapper(graph_->GetCodeObj()).CellVarsSize();
    nodes.insert(nodes.begin(), closures.begin() + cellvar_size, closures.end());
  } else {
    nodes.insert(nodes.begin(), closures.begin(), closures.end());
  }

  for (const CellVarNode *cell_node : nodes) {
    if (cell_node == &ValueNode::kUnboundLocal || cell_node->GetCellOper().empty()) {
      continue;
    }
    const std::vector<ValueNode *> &cell_ops = cell_node->GetCellOper();
    auto it = std::find_if(cell_ops.rbegin(), cell_ops.rend(), [](ValueNode *node) {
      return node->GetOpcode() == STORE_DEREF || node->GetOpcode() == DELETE_DEREF;
    });
    if (it == cell_ops.rend() || (*it)->GetOpcode() == DELETE_DEREF) {
      // Currently, the recovery of DELETE_DEREF is not supported.
      continue;
    }
    graph_->GetSideEffect()->Record(*it);
  }
}

// check whether the node can be added to the output of the graph
// or can be added to the output of the graph through transformation
// support : none, scalar, tensor, tuple, list, dict, and combination during them
inline bool IsValidGraphOutput(const AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return false;
  }
  if (abstract->isa<abstract::AbstractSlice>() && abstract->BuildValue() != kValueAny) {
    return true;
  }
  if (abstract->isa<abstract::AbstractSequence>()) {
    const auto elements = abstract->cast<abstract::AbstractSequencePtr>()->elements();
    return std::all_of(elements.begin(), elements.end(), IsValidGraphOutput);
  }
  if (abstract->isa<abstract::AbstractDictionary>()) {
    const auto elements = abstract->cast<abstract::AbstractDictionaryPtr>()->elements();
    return std::all_of(elements.begin(), elements.end(), [](const abstract::AbstractElementPair &elem) {
      return IsValidGraphOutput(elem.first) && IsValidGraphOutput(elem.second);
    });
  }
  return FuncGraphBuilder::IsValidScalar(abstract) || FuncGraphBuilder::IsValidTensor(abstract) ||
         // none is transform to LOAD_CONST
         abstract->isa<abstract::AbstractNone>();
}

inline bool IsValidOutput(const ValueNode *node) {
  return node != nullptr && node->abstract_wrapper() != nullptr &&
         IsValidGraphOutput(node->abstract_wrapper()->abstract());
}

std::vector<ValueNode *> CollectInputs(const std::vector<ValueNode *> &nodes) {
  std::set<ValueNode *> inputs;
  for (const auto &node : nodes) {
    inputs.insert(node->getInputs().begin(), node->getInputs().end());
  }
  for (const auto &node : nodes) {
    inputs.erase(node);
  }
  return std::vector<ValueNode *>(inputs.begin(), inputs.end());
}

void ReplaceSequenceNoneElementWithConst(ValueNode *node, Graph *graph) {
  auto opcode = node->GetOpcode();
  if (opcode != BUILD_LIST && opcode != BUILD_TUPLE) {
    return;
  }
  for (auto iter = node->getInputs().begin(); iter != node->getInputs().end(); iter++) {
    auto abstract_wrapper = (*iter)->abstract_wrapper();
    MS_EXCEPTION_IF_NULL(abstract_wrapper);
    auto abstract = abstract_wrapper->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    if (abstract->isa<abstract::AbstractNone>()) {
      *iter = graph->NewValueNode(AObject::Convert(Py_None), LOAD_CONST, 0, {});
      (*iter)->set_abstract_wrapper(abstract_wrapper);
    }
  }
}

void UpdateUseDefOrder(std::vector<ValueNode *> *nodes) {
  MS_EXCEPTION_IF_NULL(nodes);
  std::list<ValueNode *> node_list(nodes->begin(), nodes->end());
  nodes->clear();
  while (!node_list.empty()) {
    auto front = node_list.front();
    node_list.pop_front();
    auto inputs = front->getInputs();
    auto independent = std::all_of(inputs.begin(), inputs.end(), [&node_list](const auto &input) {
      return std::find(node_list.begin(), node_list.end(), input) == node_list.end();
    });
    if (inputs.empty() || independent) {
      nodes->push_back(front);
    } else {
      node_list.push_back(front);
    }
  }
}

ValueNode *MindGraphAnalyzer::MutateSequenceNode(ValueNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract_wrapper = node->abstract_wrapper();
  MS_EXCEPTION_IF_NULL(abstract_wrapper);
  auto abstract = abstract_wrapper->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return node;
  }
  auto &captured = GetCaptureInfo().captured_;
  auto sequence = abstract->cast<abstract::AbstractSequencePtr>();
  auto func_graph_builder = std::static_pointer_cast<MindGraphBuilder>(graph_builder_)->FGBuilder();
  auto graph_node = func_graph_builder->GetNodeByWrapper(abstract_wrapper);
  auto func_graph = func_graph_builder->graph(true);
  bool is_tuple = abstract->isa<abstract::AbstractTuple>();

  auto mutated_node = graph_->NewValueNode(node->GetVobj(), is_tuple ? BUILD_TUPLE : BUILD_LIST, sequence->size(), {});
  mutated_node->set_abstract_wrapper(std::make_shared<AbstractWrapper>(abstract));  // used to remove duplicate data
  auto prim = is_tuple ? prim::kPrimTupleGetItem : prim::kPrimListGetItem;
  for (size_t index = 0; index < sequence->size(); index++) {
    auto graph_index = NewValueNode(SizeToLong(index));
    auto graph_item = func_graph->NewCNodeInOrder(prim, {graph_node, graph_index});
    graph_item->set_abstract(sequence->elements()[index]);
    auto item_abstract_wrapper = std::make_shared<AbstractWrapper>(sequence->elements()[index]);
    auto index_abstract_wrapper = std::make_shared<AbstractWrapper>(graph_index->abstract());
    func_graph_builder->AddLocalVariableNode(item_abstract_wrapper, graph_item);
    func_graph_builder->AddLocalVariableNode(index_abstract_wrapper, graph_index);

    auto bc_index = graph_->NewValueNode(AObject::Convert(py::int_(index)), LOAD_CONST, -1, {});
    bc_index->set_abstract_wrapper(index_abstract_wrapper);
    auto bc_item = graph_->NewValueNode(AObject::Convert(item_abstract_wrapper), BINARY_SUBSCR, 0, {node, bc_index});
    bc_item->set_abstract_wrapper(item_abstract_wrapper);

    ADD_NODE(captured.operations, bc_index);
    ADD_NODE(captured.operations, bc_item);
    mutated_node->AddInput(bc_item);
  }
  ReplaceSequenceNoneElementWithConst(mutated_node, graph_);
  GetCaptureInfo().replaced_nodes_[node] = mutated_node;
  return mutated_node;
}

ValueNode *MindGraphAnalyzer::MutateNamedtupleNode(ValueNode *tuple_node, ValueNode *namedtuple_node) {
  MS_LOG(DEBUG) << "Start mutate namedtuple node, origin namedtuple node: " << namedtuple_node->ToString()
                << ", tuple node: " << tuple_node->ToString();
  MS_EXCEPTION_IF_NULL(namedtuple_node->GetVobj());
  // Delete the abstract wrapper, then it will be executed in pynative.
  tuple_node->set_abstract_wrapper(nullptr);
  auto namedtuple_type = reinterpret_cast<PyObject *>(namedtuple_node->GetVobj()->GetTypeObject());
  MS_EXCEPTION_IF_NULL(namedtuple_type);
  auto method_node = graph_->NewValueNode(AObject::Convert(namedtuple_type), LOAD_CONST, -1, {});

  // Create namedtuple, it will be executed in pynative. eval 'namedtuple(*tuple)' expression.
  ValueNode *mutated_node = graph_->NewCallNode(CALL_FUNCTION_EX, 0, {method_node, tuple_node});
  mutated_node->SetVobj(namedtuple_node->GetVobj());  // used to remove duplicate data
  GetCaptureInfo().replaced_nodes_[namedtuple_node] = mutated_node;
  // add to interpret
  return mutated_node;
}

// return keys and values
std::pair<ValueNode *, ValueNode *> MindGraphAnalyzer::MutateDictNode(ValueNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto abstract_wrapper = node->abstract_wrapper();
  MS_EXCEPTION_IF_NULL(abstract_wrapper);
  auto abstract = abstract_wrapper->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractDictionary>()) {
    return std::make_pair(node, nullptr);
  }
  auto &captured = GetCaptureInfo().captured_;
  auto &interpret = GetCaptureInfo().interpret_;
  auto &outputs_optimize = GetCaptureInfo().outputs_optimize_;

  auto dict = abstract->cast<abstract::AbstractDictionaryPtr>();
  AbstractBasePtrList key_abstracts;
  AbstractBasePtrList value_abstracts;
  std::for_each(dict->elements().begin(), dict->elements().end(),
                [&key_abstracts, &value_abstracts](const abstract::AbstractElementPair &ele) {
                  key_abstracts.push_back(ele.first);
                  value_abstracts.push_back(ele.second);
                });
  auto keys_wrapper = std::make_shared<AbstractWrapper>(std::make_shared<abstract::AbstractTuple>(key_abstracts));
  auto values_wrapper = std::make_shared<AbstractWrapper>(std::make_shared<abstract::AbstractTuple>(value_abstracts));
  auto func_graph_builder = std::static_pointer_cast<MindGraphBuilder>(graph_builder_)->FGBuilder();
  auto graph_node = func_graph_builder->GetNodeByWrapper(abstract_wrapper);
  MS_EXCEPTION_IF_NULL(graph_node);
  auto func_graph = func_graph_builder->graph(true);
  auto keys = func_graph->NewCNodeInOrder(prim::kPrimDictGetKeys, {graph_node});
  auto values = func_graph->NewCNodeInOrder(prim::kPrimDictGetValues, {graph_node});
  keys->set_abstract(keys_wrapper->abstract());
  values->set_abstract(values_wrapper->abstract());
  func_graph_builder->AddLocalVariableNode(keys_wrapper, keys);
  func_graph_builder->AddLocalVariableNode(values_wrapper, values);

  // find unique node for builtin method
  ValueNode *dict_keys_method = GetBuiltinMethodNode(&captured.operations, "keys", "dict");
  ValueNode *dict_values_method = GetBuiltinMethodNode(&captured.operations, "values", "dict");
  ValueNode *zip_method = GetBuiltinMethodNode(&interpret.operations, "zip");    // always interpret
  ValueNode *dict_method = GetBuiltinMethodNode(&interpret.operations, "dict");  // always interpret
  // use 'dict(zip(dict.keys(obj), dict.value(obj)))' to restore dict
  // consider use expression 'dict(dict.items(obj))'
  auto bc_keys = graph_->NewCallNode(CALL_FUNCTION, 1, {dict_keys_method, node});
  bc_keys->set_abstract_wrapper(keys_wrapper);
  bc_keys->SetVobj(AObject::Convert(keys_wrapper));
  auto bc_values = graph_->NewCallNode(CALL_FUNCTION, 1, {dict_values_method, node});
  bc_values->set_abstract_wrapper(values_wrapper);
  bc_values->SetVobj(AObject::Convert(values_wrapper));
  auto call_zip = graph_->NewCallNode(CALL_FUNCTION, 2, {zip_method, bc_keys, bc_values});
  auto make_dict = graph_->NewCallNode(CALL_FUNCTION, 1, {dict_method, call_zip});
  make_dict->set_abstract_wrapper(std::make_shared<AbstractWrapper>(abstract));
  make_dict->SetVobj(node->GetVobj());

  // keys and values is graph values
  ADD_NODE(captured.operations, bc_keys);
  ADD_NODE(captured.operations, bc_values);
  // call zip and call dict is interpret operations
  ADD_NODE(outputs_optimize.operations, make_dict);
  ADD_NODE(outputs_optimize.operations, call_zip);

  GetCaptureInfo().replaced_nodes_[node] = make_dict;
  return std::make_pair(bc_keys, bc_values);
}

namespace {
constexpr auto kPiJitOutputDepthKey = "pi_jit_output_depth";
constexpr int kAllowMaxDepth = 3;

bool IsNeedExpand(const ValueNode *node) {
  auto wrapper = node->abstract_wrapper();
  MS_EXCEPTION_IF_NULL(wrapper);
  auto abs = wrapper->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  constexpr int allow_tuple_max_depth = 3;
  return abs->has_user_data(kPiJitOutputDepthKey) && *abs->user_data<int>(kPiJitOutputDepthKey) > allow_tuple_max_depth;
}
}  // namespace

void MindGraphAnalyzer::ExpandGraphOutput() {
  if (!graph_->Config().GetBoolConfig(GraphJitConfig::kExpandGraphOutput)) {
    return;
  }
  std::function<int(const abstract::AbstractBasePtr &)> depth_marker =
    [&depth_marker](const abstract::AbstractBasePtr &abstract) {
      MS_EXCEPTION_IF_NULL(abstract);
      if (!abstract->isa<abstract::AbstractSequence>()) {
        return 0;
      }
      std::vector<int> depths;
      const auto &elements = abstract->cast<abstract::AbstractSequencePtr>()->elements();
      std::transform(elements.begin(), elements.end(), std::back_inserter(depths),
                     [&depth_marker](const auto &element) { return depth_marker(element); });
      auto depth = (depths.empty() ? 0 : *std::max_element(depths.begin(), depths.end())) + 1;
      abstract->set_user_data<int>(kPiJitOutputDepthKey, std::make_shared<int>(depth));
      return depth;
    };
  auto mind_graph_builder = std::static_pointer_cast<MindGraphBuilder>(graph_builder_);
  MS_EXCEPTION_IF_NULL(mind_graph_builder);
  auto func_graph_builder = mind_graph_builder->FGBuilder();
  MS_EXCEPTION_IF_NULL(func_graph_builder);
  func_graph_builder->ClearOutputNodes();
  auto &captured = GetCaptureInfo().captured_;
  auto &outputs_optimize = GetCaptureInfo().outputs_optimize_;
  mindspore::CompactSet<ValueNode *> nodes;
  nodes.insert(captured.outputs.begin(), captured.outputs.end());
  captured.outputs.clear();
  for (const auto &node : nodes) {
    auto wrapper = node->abstract_wrapper();
    MS_EXCEPTION_IF_NULL(wrapper);
    auto abs = wrapper->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    (void)depth_marker(abs);
  }
  while (!nodes.empty()) {
    auto node = nodes.pop();
    MS_LOG(DEBUG) << "Start process : " << node->ToString();
    if (!IsNeedExpand(node)) {
      MS_LOG(DEBUG) << "Add to output : " << node->ToString();
      captured.outputs.push_back(node);
      auto succ = func_graph_builder->AddOutput(node->abstract_wrapper(), true);
      MS_EXCEPTION_IF_CHECK_FAIL(succ, "Add " + node->ToString() + " to graph outputs failed.");
      continue;
    }
    MS_LOG(DEBUG) << "Start expand : " << node->ToString();
    auto opcode = node->GetOpcode();
    if (opcode != BUILD_LIST && opcode != BUILD_TUPLE) {
      MS_LOG(DEBUG) << "Start mutate : " << node->ToString();
      node = MutateSequenceNode(node);  // transform to build_list or build_tuple
      MS_LOG(DEBUG) << "After mutate : " << node->ToString();
    }
    ADD_NODE(outputs_optimize.operations, node);
    nodes.insert(node->getInputs().begin(), node->getInputs().end());
  }
}

bool MindGraphAnalyzer::AnalyzeAliveLocals(std::vector<ValueNode *> aliveNodes) {
  auto mind_graph_builder = std::static_pointer_cast<MindGraphBuilder>(graph_builder_);
  MS_EXCEPTION_IF_NULL(mind_graph_builder);
  auto func_graph_builder = mind_graph_builder->FGBuilder();
  MS_EXCEPTION_IF_NULL(func_graph_builder);
  func_graph_builder->ClearOutputNodes();
  auto &captured = GetCaptureInfo().captured_;
  captured.outputs.clear();
  auto &outputs_optimize = GetCaptureInfo().outputs_optimize_;

  // use order set as work list
  mindspore::CompactSet<ValueNode *> nodes;
  nodes.insert(aliveNodes.begin(), aliveNodes.end());
  while (!nodes.empty()) {
    auto node = *nodes.begin();
    nodes.erase(nodes.begin());
    MS_LOG(INFO) << "Start analyze : " << node->ToString() << " abs : "
                 << (node->abstract_wrapper() == nullptr ? "nullptr"
                                                         : node->abstract_wrapper()->abstract()->ToString());
    if (NeedSkipAddGraphOutput(node)) {
      continue;
    }
    // add output for func_graph
    if (func_graph_builder->AddOutput(node->abstract_wrapper(), true)) {
      MS_LOG(INFO) << "Add graph output : " << node->ToString();
      ADD_NODE(captured.outputs, node);  // must be equal as FuncGraph outputs
      continue;
    }

    // Every node that appears here should have a corresponding anf node in top func graph.
    // Unfortunately, due to some defectuve side-effect node processing, they do not have
    // This issue must be fixed, just pass-by and reminder here
    // This code will be redundant after the issue fixed.
    bool is_not_in_top_graph = (func_graph_builder->FindNodeByWrapper(node->abstract_wrapper()) == nullptr);
    // it is top graph node but not find in top func_graph

    // Contains data whose type is not supported by the graph, analyze its inputs
    if (!IsValidOutput(node) || is_not_in_top_graph) {
      auto msg = (is_not_in_top_graph ? "Not in top graph node : " : "Invalid output : ");
      MS_LOG(INFO) << msg << node->ToString();
      if (graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak) && Opcode(node->GetOpcode()).IsCall()) {
        GRAPH_JIT_LOG_F("This call node will executed in pynative : [%s]", node->ToString().c_str());
      }
      ADD_NODE(outputs_optimize.operations, node);
      nodes.insert(node->getInputs().begin(), node->getInputs().end());
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(node->abstract_wrapper() && node->abstract_wrapper()->abstract(),
                               "Error check at IsValidOutput");
    if (node->abstract_wrapper()->abstract()->isa<abstract::AbstractDictionary>()) {
      // try to add keys and value to graph outputs, here is sequence, continue to handle sequence
      auto pair = MutateDictNode(node);
      nodes.insert(pair.first);
      nodes.insert(pair.second);
    } else if (node->abstract_wrapper()->abstract()->isa<abstract::AbstractSequence>()) {
      auto sequence = MutateSequenceNode(node);  // transform to build_list or build_tuple
      if (node->abstract_wrapper()->abstract()->isa<abstract::AbstractNamedTuple>()) {
        // specialize for named tuple, 1: graph output tuple, 2: python reconstruct namedtuple
        ADD_NODE(outputs_optimize.operations, MutateNamedtupleNode(sequence, node));
      }
      ADD_NODE(outputs_optimize.operations, sequence);
      nodes.insert(sequence->getInputs().begin(), sequence->getInputs().end());
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "the node can't add graph out and not handle by output optimize, it's missing ["
                                 << node->ToString();
    }
  }
  ExpandGraphOutput();
  std::reverse(outputs_optimize.operations.begin(), outputs_optimize.operations.end());
  // avoid missing value, update use-def at last, update all inputs use new node
  UpdateUseDefNode();
  return true;
}

void MindGraphAnalyzer::UpdateCapturedOrder() {
  const auto &locals = graph_->GetFrame(0).GetLocals();
  GetCaptureInfo().interpret_.inputs = locals;
  GetCaptureInfo().interpret_.values.clear();
  GetCaptureInfo().interpret_.values.insert(locals.begin(), locals.end());
  GetCaptureInfo().interpret_.values.insert(graph_->prepare().inputs_.begin(), graph_->prepare().inputs_.end());
  GetCaptureInfo().interpret_.values.insert(graph_->prepare().operations_.begin(), graph_->prepare().operations_.end());
}

void MindGraphAnalyzer::CollectCapturedAndInterpret() {
  CollectCapturedInputs();

  GetCaptureInfo().outputs_optimize_.inputs = CollectInputs(GetCaptureInfo().outputs_optimize_.operations);
  GetCaptureInfo().interpret_.inputs = graph_->GetFrame(0).GetLocals();
  GetCaptureInfo().interpret_.outputs = graph_->CollectAliveNode(graph_->GetStopTraceBci(), &alive_locals_);
  const auto &prepare = graph_->prepare().operations_;
  auto &interpret = GetCaptureInfo().interpret_.operations;
  interpret.insert(interpret.begin(), prepare.begin(), prepare.end());

  // remove side-effect node
  auto is_remove = [this](ValueNode *node) {
    const auto &rec = this->graph_->GetSideEffect();
    return rec->IsRecord(node) && !rec->NeedTrack(node);
  };
  auto *ops = &GetCaptureInfo().captured_.operations;
  ops->erase(std::remove_if(ops->begin(), ops->end(), is_remove), ops->end());
  ops = &GetCaptureInfo().interpret_.operations;
  ops->erase(std::remove_if(ops->begin(), ops->end(), is_remove), ops->end());
}

void MindGraphAnalyzer::UseDefAnalyze() {
  // UD analyze: alive nodes analysis
  std::vector<ValueNode *> aliveLocals = GetAliveLocals(graph_);
  if (!aliveLocals.empty()) {
    bool stop_analyze = false;
    while (!stop_analyze) {
      UpdateCapturedOrder();
      // Add graph output according to leaf nodes.
      stop_analyze = AnalyzeAliveLocals(aliveLocals);
      if (!stop_analyze) {
        aliveLocals = GetAliveLocals(graph_);
      }
    }
  }
}

void MindGraphAnalyzer::CollectGraphInputs() {
  // graph inputs is ordered by MindGraphBuilder, here do nothing
  // not care variable args, variable key words
  GetCaptureInfo().graph_inputs_.args = GetCaptureInfo().captured_.inputs;
}

void MindGraphAnalyzer::ResetSideEffectRecord() const {
  // side-effect rollback, adapter later
  // sub-graph side-effect rollback, adapter later
  int break_bci = graph_->GetStopTraceBci();
  if (break_bci == -1 || graph_->GetSideEffect()->IsEmpty()) {
    return;
  }
  const auto &nodes = graph_->GetSideEffect()->nodes();
  for (const auto &pair : nodes) {
    Graph *g = pair.first->GetGraph();
    if (g != nullptr && g != graph_) {
      MS_LOG(ERROR) << "function " << PyCodeWrapper(g->GetCodeObj()).Name()
                    << " has side-effect but not implement side-effect rollback";
      return;
    }
  }
  this->GraphAnalyzer::ResetSideEffectRecord();
}

// specialize simple data, not all equal
static bool IsDuplicateData(const AbstractBasePtr &left, const AbstractBasePtr &right) {
  if (left == nullptr || right == nullptr || left->tid() != right->tid()) {
    return false;
  }
  // first check ptr
  if (left == right) {
    return true;
  }
  // check type
  if (left->isa<abstract::AbstractTensor>()) {
    return false;  // tensor always not duplicate
  }
  if (left->isa<abstract::AbstractSequence>()) {
    const auto &arr_l = left->cast<abstract::AbstractSequencePtr>()->elements();
    const auto &arr_r = right->cast<abstract::AbstractSequencePtr>()->elements();
    return std::equal(arr_l.begin(), arr_l.end(), arr_r.begin(), arr_r.end(), IsDuplicateData);
  }
  if (left->isa<abstract::AbstractDictionary>()) {
    const auto &arr_l = left->cast<abstract::AbstractDictionaryPtr>()->elements();
    const auto &arr_r = right->cast<abstract::AbstractDictionaryPtr>()->elements();
    auto comp = [](const abstract::AbstractElementPair &a, const abstract::AbstractElementPair &b) {
      return IsDuplicateData(a.first, b.first) && IsDuplicateData(a.second, b.second);
    };
    return std::equal(arr_l.begin(), arr_l.end(), arr_r.begin(), arr_r.end(), comp);
  }
  if (left->isa<abstract::AbstractScalar>()) {
    return left->BuildValue() != kValueAny && *left == *right;
  }
  return false;  // invalid output
}

static ValueNode *FindDuplicateData(const std::vector<ValueNode *> &nodes, size_t end_idx, ValueNode *node) {
  MS_EXCEPTION_IF_CHECK_FAIL(end_idx <= nodes.size(), "error arguments");
  const auto end_iter = nodes.begin() + end_idx;
  auto iter = std::find(nodes.begin(), end_iter, node);
  if (iter != end_iter) {
    return *iter;
  }
  iter = std::find_if(nodes.begin(), end_iter, [&node](ValueNode *k) {
    auto left = node->abstract_wrapper() ? node->abstract_wrapper()->abstract() : nullptr;
    auto right = k->abstract_wrapper() ? k->abstract_wrapper()->abstract() : nullptr;
    return IsDuplicateData(left, right) || (node->GetVobj() != nullptr && node->GetVobj() == k->GetVobj());
  });
  if (iter != end_iter) {
    return *iter;
  }
  return nullptr;
}

bool MindGraphAnalyzer::NeedSkipAddGraphOutput(ValueNode *node) {
  const auto &values = GetCaptureInfo().interpret_.values;
  const auto &captured = GetCaptureInfo().captured_;
  const auto &outputs_optimize = GetCaptureInfo().outputs_optimize_;
  const auto &replaced = GetCaptureInfo().replaced_nodes_;
  // If the value can get from local, no need to add to graph output.
  if (IsNonLocalValue(node)) {
    MS_LOG(INFO) << "Skip non local value used as graph output: " << node->ToString();
    return true;
  }
  // This node is defined out of the graph
  if (std::find(values.begin(), values.end(), node) != values.end()) {
    return true;
  }
  // This node has been added to the output
  if (std::find(captured.outputs.begin(), captured.outputs.end(), node) != captured.outputs.end()) {
    return true;
  }
  // This node has been handle
  auto &handled_nodes = outputs_optimize.operations;
  if (std::find(handled_nodes.begin(), handled_nodes.end(), node) != handled_nodes.end()) {
    return true;
  }

  if (node->abstract_wrapper() && node->abstract_wrapper()->IsConstant()) {
    PyObject *op = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
    if (op == nullptr) {
      return false;  // constant value no python object, can't make instruction
    } else if (CheckConstPyObject(op)) {
      // now, only python constant
      if (PyUnicode_Check(op) && !FuncGraphBuilder::IsValidScalar(node->abstract_wrapper()->abstract())) {
        // filter FakeNodeKey
        return false;
      }
      node->ClearInputs();
      node->SetOpcode(LOAD_CONST);
      return true;
    } else if (PySlice_Check(op)) {
      const auto &slice_inputs = node->abstract_wrapper()->GetSliceInputsPyObject();
      auto start = graph_->NewValueNode(AObject::Convert(slice_inputs[0]), LOAD_CONST, -1, {});
      auto stop = graph_->NewValueNode(AObject::Convert(slice_inputs[1]), LOAD_CONST, -1, {});
      auto step = graph_->NewValueNode(AObject::Convert(slice_inputs[2]), LOAD_CONST, -1, {});
      auto ret = graph_->NewValueNode(AObject::Convert(op), BUILD_SLICE, 3, {start, stop, step});
      GetCaptureInfo().replaced_nodes_[node] = ret;
      GetCaptureInfo().outputs_optimize_.operations.push_back(ret);
    }
  }
  // remove duplicate data
  auto duplicate_data = FindDuplicateData(captured.outputs, captured.outputs.size(), node);
  if (duplicate_data == nullptr) {
    duplicate_data = FindDuplicateData(outputs_optimize.outputs, outputs_optimize.outputs.size(), node);
  }
  if (duplicate_data != nullptr) {
    GetCaptureInfo().replaced_nodes_[node] = duplicate_data;
    MS_LOG(INFO) << "skip same data: [" << node->ToString() << "] and [" << duplicate_data->ToString();
    return true;
  }
  auto iter_replaced = replaced.find(node);
  if (iter_replaced != replaced.end()) {
    MS_LOG(INFO) << "duplicate node " << node->ToString();
    return NeedSkipAddGraphOutput(iter_replaced->second);
  }
  return false;
}

ValueNode *MindGraphAnalyzer::GetBuiltinMethodNode(std::vector<ValueNode *> *out, const std::string &name,
                                                   const std::string &cls) {
  PyObject *builtin_module = PyEval_GetBuiltins();
  MS_EXCEPTION_IF_NULL(builtin_module);
  if (PyModule_Check(builtin_module)) {
    builtin_module = PyModule_GetDict(builtin_module);
    MS_EXCEPTION_IF_NULL(builtin_module);
  }
  py::object builtin_object;
  if (cls.empty()) {
    auto method = PyDict_GetItemString(builtin_module, name.c_str());
    builtin_object = py::reinterpret_borrow<py::object>(method);
  } else {
    auto cls_object = PyDict_GetItemString(builtin_module, cls.c_str());
    MS_EXCEPTION_IF_NULL(cls_object);
    auto method = PyObject_GetAttrString(cls_object, name.c_str());
    builtin_object = py::reinterpret_steal<py::object>(method);
  }
  MS_EXCEPTION_IF_NULL(builtin_object.ptr());
  auto method_node = graph_->NewValueNode(AObject::Convert(builtin_object), LOAD_CONST, 0, {});
  out->push_back(method_node);
  return method_node;
}

static void UpdateNodeInputs(Graph *graph, std::vector<ValueNode *> *nodes_p,
                             std::map<ValueNode *, ValueNode *> *map_p) {
  const auto &map = *map_p;
  const auto &nodes = *nodes_p;
  auto latest = [&map](ValueNode *node) {
    int limit = 10000;
    for (auto iter = map.find(node); limit > 0 && iter != map.end(); iter = map.find(node), --limit) {
      node = iter->second;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(limit > 0, "maybe circle map");
    return node;
  };
  bool changed;
  do {
    changed = false;
    // for each node check it's inputs
    for (auto node_iter = nodes.begin(); node_iter != nodes.end(); ++node_iter) {
      auto node = *node_iter;
      auto &in = node->getInputs();
      auto in_iter = std::find_if(in.begin(), in.end(), [&map](ValueNode *k) { return map.find(k) != map.end(); });
      if (in_iter == in.end()) {
        continue;  // not find, do nothing
      }
      // collect latest node
      std::vector<ValueNode *> new_in = node->getInputs();
      for (; in_iter != in.end(); ++in_iter) {
        new_in[in_iter - in.begin()] = latest(*in_iter);
      }
      // if node is a a new node, update inputs
      if (node->GetLineNo() < 0) {
        in = std::move(new_in);
        continue;
      }
      Opcode opcode(node->GetOpcode());
      ValueNode *new_node;
      if (opcode.IsCall()) {
        new_node = graph->NewCallNode(opcode, node->GetOparg(), std::move(new_in));
        new_node->SetVobj(node->GetVobj());
      } else {
        new_node = graph->NewValueNode(node->GetVobj(), opcode, node->GetOparg(), std::move(new_in), node->GetName());
      }
      (*nodes_p)[node_iter - nodes.begin()] = new_node;
      (*map_p)[node] = new_node;
      changed = true;
    }
  } while (changed);
}

void MindGraphAnalyzer::UpdateUseDefNode() {
  auto &map = GetCaptureInfo().replaced_nodes_;
  auto &nodes = GetCaptureInfo().outputs_optimize_.operations;
  if (map.empty()) {
    UpdateUseDefOrder(&nodes);
    return;
  }
  UpdateNodeInputs(graph_, &nodes, &map);
  UpdateUseDefOrder(&nodes);
}

}  // namespace pijit
}  // namespace mindspore
