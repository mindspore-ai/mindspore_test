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
#include "pipeline/jit/pi/graph_capture/graph.h"
#include <set>
#include <algorithm>
#include <memory>
#include <string>
#include <regex>
#include <utility>
#include "pipeline/jit/pi/graph_build/func_graph_builder.h"
#include "pipeline/jit/pi/runtime.h"

namespace mindspore {
namespace pijit {
Graph::Graph(PyCodeObject *co, PyObject *globals, const GraphJitConfig &conf)
    : ret_val_(nullptr),
      generator_result_(nullptr),
      co_(py::cast<py::object>(reinterpret_cast<PyObject *>(co))),
      f_globals_(py::cast<py::object>(globals)),
      conf_(conf),
      guard_(nullptr),
      prune_branch_count_(0),
      func_graph_builder_(nullptr) {
  stop_trace_info_ = {-1, StopTraceReason::kNonStopTrace};
  if (!co) {
    frame_states_[0] = std::make_unique<FrameStates>();  // emplace empty frame
    module_name_ = "";
    return;
  }
  cfg_ = std::make_unique<CFG>(co);
  cfg_->GenerateCFG();

  if (conf_.GetBoolConfig(GraphJitConfig::kPrintBB)) {
    GRAPH_JIT_LOG_F("%s\n\n", cfg_->DumpBBs().c_str());
  }
  if (conf_.GetBoolConfig(GraphJitConfig::kPrintCFG)) {
    cfg_->DumpCFGGraph();
  }

  auto pyname = PyDict_GetItemString(globals, "__name__");
  if (pyname) {
    module_name_ = PyUnicode_AsUTF8(pyname);
  } else {
    module_name_ = "";
    PyErr_Clear();
  }

  if (conf_.GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    LoopFinder loop_finder(this);
    loop_finder.FormSimpleLoopInfo();
  }
}

bool Graph::PrepareParameter(ValueNode *node) {
  using PrepareHelper = bool (*)(ValueNode *, std::vector<ValueNode *> *);
  static PrepareHelper prepare_oper = [](ValueNode *node, std::vector<ValueNode *> *oper) {
    int opcode = node->GetOpcode();
    if (opcode == LOAD_CONST || opcode == LOAD_GLOBAL || opcode == LOAD_DEREF || node->GetType() == ValueNode::Param ||
        oper->end() != std::find(oper->begin(), oper->end(), node)) {
      return true;
    }
    while (node->GetType() == ValueNode::Call) {
      auto g = static_cast<CallNode *>(node)->GetSubGraph();
      if (g != nullptr && g->GetRetVal() != nullptr) {
        node = g->GetRetVal();
      }
    }
    if (opcode != LOAD_ATTR && opcode != BINARY_SUBSCR) {
      return false;
    }
    for (const auto &in : node->getInputs()) {
      if (!prepare_oper(in, oper)) {
        return false;
      }
    }
    oper->push_back(node);
    return true;
  };

  prepare_.inputs_.push_back(node);
  size_t backup = prepare_.operations_.size();
  if (prepare_oper(node, &prepare_.operations_)) {
    return true;
  }
  prepare_.operations_.resize(backup);
  prepare_.inputs_.pop_back();
  return false;
}

const std::shared_ptr<SideEffect> &Graph::GetSideEffect() const { return side_effect_; }
void Graph::SetSideEffect(const std::shared_ptr<SideEffect> &handler) { side_effect_ = handler; }

static void SetNodeInfo(Graph *graph, ValueNode *node, AObject *obj_info, const std::string &name) {
  node->SetName(name);
  node->SetGraph(graph);
  ConstantInfo::CollectConstantInfo(node);
  if (node->IsConstantValue() && obj_info && CheckConstPyObject(obj_info->GetPyObject().ptr())) {
    node->SetOpcode(LOAD_CONST);
    node->SetOparg(-1);
    node->ClearInputs();
  }
  auto new_object = obj_info ? obj_info->GetPyObject().ptr() : nullptr;
  if (new_object != nullptr && !CheckConstPyObject(new_object)) {  // literal not need track
    graph->GetSideEffect()->data()->Track(new_object, node);
  }
}

CallNode *Graph::NewCallNode(int op, int arg, const std::vector<ValueNode *> &inputs) {
  MS_EXCEPTION_IF_CHECK_FAIL(Opcode(op).IsCall(), "must be call function opcode");
  CallNode *node = this->allocator().NewNode<CallNode>(op, arg, inputs);
  node->SetGraph(this);
  return node;
}

ValueNode *Graph::NewValueNode(AObject *obj_info, int op, int arg, const std::vector<ValueNode *> &inputs,
                               const std::string &name) {
  // when got a new object, check it's side-effect replaced node ......
  MS_EXCEPTION_IF_CHECK_FAIL(!Opcode(op).IsCall(), "must not be call function opcode");
  ValueNode *node = this->allocator().NewNode<ValueNode>(obj_info, op, arg, inputs);
  SetNodeInfo(this, node, obj_info, name);
  return node;
}

CellVarNode *Graph::NewCellNode(AObject *obj_info, int op, int arg, const std::vector<ValueNode *> &inputs,
                                const std::string &name) {
  CellVarNode *node = this->allocator().NewNode<CellVarNode>(CellVarNode::CellVar);
  SetNodeInfo(this, node, obj_info, name);
  for (auto i : inputs) {
    node->AddInput(i);
  }
  node->SetOpcode(op);
  node->SetOparg(arg);
  node->SetVobj(obj_info);
  return node;
}

/**
 * FindLoopEnd, FindLoopBegin, reset break bci
 * restore graph status. clean the variable that loop produced
 * restore frame status of break bci that override by loop analyze
 */
bool Graph::IsBreakAtLoop() const {
  int break_bci = this->GetStopTraceBci();
  if (break_bci == -1) {
    return false;
  }
  const auto &instr = this->cfg_->instr_pool();
  // find the last backward edge overlapping this break point
  int res = break_bci;
  for (int i = break_bci; i < SizeToInt(instr.size()); ++i) {
    MS_EXCEPTION_IF_CHECK_FAIL(i == instr[i]->bci(), "!!!");
    if (instr[i]->extra_jump() != nullptr) {
      res = std::min(instr[i]->extra_jump()->bci(), res);
    }
  }
  return res != break_bci;
}

/**
 * Should Never Compile
 */
bool Graph::ShouldNeverCompile() const {
  if (this->IsBreakAtLoop() && !this->RestoreLoopStatus()) {
    return true;
  }

  return found_inner_class;
}

bool Graph::IsBreakAtLoopAfterUnrolling() const {
  if (!Config().GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    return false;
  }
  if (GetStopTraceBci() == -1) {
    return false;
  }
  if (traced_nodes_.empty()) {
    return false;
  }
  if (GetStopTraceBci() > traced_nodes_.back()->bci()) {
    return false;
  }
  bool break_with_loop_unroll = false;
  for (auto node : traced_nodes_) {
    if (node->bci() >= GetStopTraceBci()) {
      break_with_loop_unroll |= node->GetBlock() != nullptr && node->GetBlock()->is_loop_body();
    }
  }
  return break_with_loop_unroll;
}

void Graph::SetFrame(int bci, const FrameStates &f) {
  // just set once, used to restore the first status if has a loop
  auto &ptr = frame_states_[bci];
  if (ptr == nullptr) {
    ptr = std::make_unique<FrameStates>(f);
  }
}

const FrameStates &Graph::GetFrame(int bci) const {
  auto iter = frame_states_.find(bci);
  MS_EXCEPTION_IF_CHECK_FAIL(iter != frame_states_.end(), "can't find frame status");
  return *(iter->second);
}

static bool CheckObjPtr(ValueNode *node) {
  return node->GetVobj() == nullptr || node->GetVobj()->GetPyObject().ptr() == nullptr;
}

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);
static bool PrepareTraceParam(ValueNode *node, TraceVector *tv, int depth, int max_depth, bool *has_unsupported,
                              bool strict, bool print) {
  const std::vector<ValueNode *> &inputs = node->getInputs();
  for (auto it : inputs) {
    auto t = GetTrace(it, strict, print, depth + 1, max_depth);
    if (t == nullptr) {
      if (it->GetTrace() != nullptr) {
        tv->push_back(it->GetTrace());
      }
      return false;
    } else if (t->GetTraceType() == TraceType::Unsupported) {
      *has_unsupported = true;
    }
    tv->push_back(t);
  }
  return !CheckObjPtr(node);
}

static bool CheckDepth(int depth, int max_depth) { return depth < max_depth || max_depth == -1; }

static bool CheckDepthForTrace(TracePtr *ret, ValueNode *node, int depth, int max_depth) {
  if (!CheckDepth(depth, max_depth)) {
    MS_LOG(DEBUG) << "too deep trace for guard";
    return false;
  }
  auto ct = node->GetTrace();
  if (ct != nullptr) {
    if (ct->GetDepth() + depth > max_depth && max_depth != -1) {
      MS_LOG(DEBUG) << "too deep trace for guard";
      return false;
    } else {
      *ret = ct;
      return false;
    }
  }
  return true;
}

static TracePtr CacheTrace(ValueNode *node, TracePtr ret, bool strict, TraceVector tv, int opcode, int oparg,
                           PyObject *obj) {
  if (ret == nullptr && !strict) {
    return std::make_shared<UnsupportedTrace>(obj, tv, opcode, oparg);
  } else {
    if (ret != nullptr) {
      node->SetTrace(ret);
    }
    return ret;
  }
}

TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth) {
  TracePtr ret = nullptr;
  if (!CheckDepthForTrace(&ret, node, depth, max_depth)) {
    return ret;
  }
  if (node->GetType() == AbstractNode::Type::Call) {
    Graph *sub_graph = static_cast<CallNode *>(node)->GetSubGraph();
    if (sub_graph && sub_graph->GetRetVal() != nullptr) {
      return GetTrace(sub_graph->GetRetVal(), strict, print, depth, max_depth);
    }
  }

  PyObject *obj = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
  int opcode = node->GetOpcode();
  int oparg = node->GetOparg();
  const std::string &name = node->GetName();
  const char *module_name = node->GetGraph() ? node->GetGraph()->GetModuleName() : "";

  TraceVector tv;
  bool has_unsupported = false;
  if (!PrepareTraceParam(node, &tv, depth, max_depth, &has_unsupported, strict, print)) {
    return strict ? nullptr : std::make_shared<UnsupportedTrace>(nullptr, tv, opcode, oparg);
  }
  switch (node->GetType()) {
    case AbstractNode::Type::CellVar: /* fall-through */
    case AbstractNode::Type::FreeVar: /* fall-through */
    case AbstractNode::Type::Value:
      if (!has_unsupported) {
        ret = CreateOpTrace(obj, opcode, oparg, tv, module_name, name, strict, print);
      }
      break;
    case AbstractNode::Type::Call:
      if (!has_unsupported) {
        const std::string &func_name = node->input(0)->GetName();
        ret = CreateOpTrace(obj, opcode, oparg, tv, module_name, func_name, strict, print);
      }
      break;
    case AbstractNode::Type::Param:
      if (oparg == -1) {
        return nullptr;
      }
      ret = std::make_shared<RootTrace>(obj, mindspore::pijit::TraceType::Param, oparg, name);
      break;
    case AbstractNode::Type::kUnbound:
    default:
      break;
  }
  return CacheTrace(node, ret, strict, tv, opcode, oparg, obj);
}

static bool IsVariableNode(PyObject *obj) {
  if (obj == nullptr) {
    return false;
  }
  return IsTensorPyObject(obj) || CheckScalar(obj);
}

static bool IsAttrClosure(ValueNode *node) {
  while (node != nullptr) {
    switch (node->GetType()) {
      case AbstractNode::Type::Call:
      case AbstractNode::Type::Value: {
        int opcode = node->GetOpcode();
        if (opcode != LOAD_ATTR) {
          return false;
        } else {
          auto inputs = node->getInputs();
          if (inputs.size() == 0) {
            return false;
          }
          node = inputs[0];
        }
      } break;
      case AbstractNode::Type::Param:
      case AbstractNode::Type::CellVar:
      case AbstractNode::Type::FreeVar:
      case AbstractNode::Type::kUnbound:
        return true;
      default:
        return false;
    }
  }
  return false;
}

static std::vector<TracePtr> GetTraceClosure(ValueNode *node, bool *succ, bool strict, bool print, int depth,
                                             int max_depth) {
  std::vector<ValueNode *> todo;
  std::map<ValueNode *, bool> done;
  todo.push_back(node);
  std::vector<TracePtr> ret;
  while (todo.size() > 0) {
    node = todo[0];
    done[node] = true;
    todo.erase(todo.begin());
    switch (node->GetType()) {
      case AbstractNode::Type::Call:
      case AbstractNode::Type::Value: {
        PyObject *obj = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
        if (IsVariableNode(obj)) {
          if (IsAttrClosure(node)) {
            auto item = GetTrace(node, strict, print, depth, max_depth);
            if (item != nullptr) {
              ret.insert(ret.end(), item);
            } else {
              *succ = false;
              MS_LOG(DEBUG) << "too deep trace for guard";
              return {};
            }
          } else {
            auto inputs = node->getInputs();
            for (auto input : inputs) {
              if (done.find(input) != done.end()) {
                continue;
              } else {
                todo.push_back(input);
              }
            }
          }
        }
      } break;
      case AbstractNode::Type::Param:
      case AbstractNode::Type::CellVar:
      case AbstractNode::Type::FreeVar:
      case AbstractNode::Type::kUnbound:
      default:
        break;
    }
  }
  return ret;
}

bool Graph::GuardValueNode(ValueNode *node, GuardLevel level) {
  if (node->IsConstantValue()) {
    return true;
  }
  TracePtr tr = this->TraceValueNode(node);
  if (tr == nullptr) {
    bool retg = this->GuardValueNodeClosure(node);
    if (retg) {
      if (level == GuardLevel::GEqual || level == GuardLevel::GId) {
        node->SetConstantValue(retg);
      }
    }
    return retg;
  }
  bool ret = guard_->GetGuard()->GuardOn(tr, level);
  if (level == GuardLevel::GEqual || level == GuardLevel::GId) {
    node->SetConstantValue(ret);
  }
  return ret;
}

bool Graph::GuardValueNodeClosure(ValueNode *node, GuardLevel level) {
  bool bSucc = true;
  auto vec = this->TraceValueNodeClosure(node, &bSucc);
  if (bSucc) {
    std::map<size_t, TracePtr> rep;
    for (auto item : vec) {
      auto id = item->Info().Id();
      if (rep.find(id) == rep.end()) {
        GetGuard()->GetGuard()->GuardOn(item, level);
      }
      rep[id] = item;
    }
    return true;
  } else {
    MS_LOG(DEBUG) << "too deep trace for guard";
    return false;
  }
}

TracePtr Graph::TraceValueNode(ValueNode *node, int max_trace_depth) {
  AObject *vo = node->GetVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return nullptr;
  }
  if (max_trace_depth < 0) {
    max_trace_depth = Config().getIntConfig(GraphJitConfig::GraphJitConfig::kMaxTraceDepth);
  }
  return GetTrace(node, Config().GetBoolConfig(GraphJitConfig::kStrictTrace),
                  Config().GetBoolConfig(GraphJitConfig::kPrintGuard), 0, max_trace_depth);
}

std::vector<TracePtr> Graph::TraceValueNodeClosure(ValueNode *node, bool *ret, int max_trace_depth) {
  AObject *vo = node->GetVobj();
  if (guard_ == nullptr || !vo || vo->GetPyObject().ptr() == nullptr) {
    return {};
  }
  bool succ = true;
  auto list = GetTraceClosure(node, &succ, Config().GetBoolConfig(GraphJitConfig::kStrictTrace),
                              Config().GetBoolConfig(GraphJitConfig::kPrintGuard), 0, max_trace_depth);
  if (ret != nullptr) {
    *ret = succ;
  }
  return list;
}

std::vector<ValueNode *> Graph::CollectAliveNode(int bci, std::vector<int> *ids, BitMap *map) const {
  std::vector<ValueNode *> result;
  if (bci == -1) {
    result = {this->GetRetVal()};
  } else {
    BitMap alive = this->GetCFG()->GetLiveness()->CollectAlive(bci);
    result = CollectAliveNode(this->GetFrame(bci), &alive, ids);
    if (map != nullptr) {
      *map = std::move(alive);
    }
  }
  if (GetSideEffect()->IsEmpty()) {
    return result;
  }
  // alive locals must be original node
  result.insert(result.end(), side_effect_->GetRequiredNodes().begin(), side_effect_->GetRequiredNodes().end());
  for (auto &node : result) {
    auto new_node = this->GetSideEffect()->GetSource(node);
    if (new_node->GetOpcode() == LOAD_ATTR) {  // transform the alive attribute source
      auto &attr_source = new_node->getInputs()[0];
      attr_source = this->GetSideEffect()->GetSource(attr_source);
    }
    node = new_node;
  }
  return result;
}

std::vector<ValueNode *> Graph::CollectAliveNode(const FrameStates &last_frame, BitMap *alive, std::vector<int> *ids) {
  std::vector<ValueNode *> outputs = last_frame.GetStacks();
  // collect alive locals
  for (BitMap::Iter iter(alive, true), end(alive, false); iter != end; ++iter) {
    size_t i = *iter;
    // exclude undefined locals
    if (last_frame.Local(i) != &ValueNode::kUnboundLocal) {
      if (ids != nullptr) {
        ids->push_back(i);
      }
      outputs.push_back(last_frame.Local(i));
    } else {
      alive->Clear(i);
    }
  }
  return outputs;
}

bool Graph::GuardSequenceNodeLength(ValueNode *sequence_node, Py_ssize_t sequence_size) {
  if (sequence_node->IsConstantValue()) {
    return true;
  }
  const auto &cnst = sequence_node->GetConstantInfo();
  if (cnst != nullptr && cnst->len() != -1) {
    MS_EXCEPTION_IF_CHECK_FAIL(sequence_size == cnst->len(), "error sequence length");
    return true;
  }
  TracePtr tr = this->TraceValueNode(sequence_node);
  if (tr == nullptr) {
    if (this->GuardValueNodeClosure(sequence_node)) {
      sequence_node->MakeConstantInfo()->set_len(sequence_size);
      return true;
    }
    return false;
  }
  const auto &guard = this->GetGuard()->GetGuard();
  bool strict = this->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);

  PyObject *builtin_len = PyDict_GetItemString(PyEval_GetBuiltins(), "len");
  MS_EXCEPTION_IF_NULL(builtin_len);
  TracePtr len_func = CreateOpTrace(builtin_len, LOAD_CONST, -1, {}, "", "", strict);
  TracePtr len_trace = CreateOpTrace(py::int_(sequence_size).ptr(), CALL_FUNCTION, 1, {len_func, tr}, "", "", strict);
  guard->GuardOn(len_trace, GuardLevel::GEqual, false);

  sequence_node->MakeConstantInfo()->set_len(sequence_size);
  return true;
}

bool Graph::GuardType(ValueNode *node) {
  if (node->IsConstantValue()) {
    return true;
  }
  const auto &cnst = node->GetConstantInfo();
  if (cnst != nullptr && cnst->type() != nullptr) {
    return true;
  }
  TracePtr tr = this->TraceValueNode(node);
  if (tr == nullptr) {
    if (this->GuardValueNodeClosure(node)) {
      node->MakeConstantInfo()->set_type(node->GetVobj()->GetTypeObject());
      return true;
    }
    return false;
  }
  bool ret = guard_->GetGuard()->GuardOn(tr, mindspore::pijit::GuardLevel::GType);
  node->MakeConstantInfo()->set_type(node->GetVobj()->GetTypeObject());
  return ret;
}

static bool SkipGuardInlinedFunc(ValueNode *func_node) {
  if (func_node->IsConstantValue()) {
    return true;
  }
  AObject::Type value_type = func_node->GetVobj()->GetType();
  if (func_node->GetOpcode() == LOAD_ATTR) {
    AObject *src_info = func_node->input(0)->GetVobj();
    if (src_info->GetType() == AObject::kTypeTensor && value_type == AObject::kTypeBoundMethod) {
      // function from Tensor
      return true;
    }
  }
  return false;
}

bool Graph::GuardInlinedFunc(CallNode *call_node) {
  if (SkipGuardInlinedFunc(call_node->input(0))) {
    return true;
  }
  TracePtr tr = this->TraceValueNode(call_node->input(0));
  if (tr == nullptr) {
    if (this->GuardValueNodeClosure(call_node->input(0))) {
      AObject *callable_info = call_node->input(0)->GetVobj();
      AObject::Type func_type = callable_info->GetType();
      if (func_type == AObject::kTypeBoundMethod) {
      } else if (func_type == AObject::kTypeCell || func_type == AObject::kTypeAnyValue) {
        call_node->input(0)->MakeConstantInfo()->set_type(callable_info->GetTypeObject());
      } else if (func_type == AObject::kTypeFunction) {
        call_node->input(0)->SetConstantValue(true);
      } else {
        return false;
      }
      return true;
    }
    return false;
  }
  const auto &guard = this->GetGuard()->GetGuard();
  bool strict = this->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);

  AObject *callable_info = call_node->input(0)->GetVobj();
  AObject::Type func_type = callable_info->GetType();
  PyObject *callable = callable_info->GetPyObject().ptr();
  if (func_type == AObject::kTypeBoundMethod) {
    PyObject *func = PyMethod_GET_FUNCTION(callable);
    tr = CreateOpTrace(func, LOAD_ATTR, 0, {tr}, "", "__func__", strict);
    guard->GuardOn(tr, GuardLevel::GId);
  } else if (func_type == AObject::kTypeCell || func_type == AObject::kTypeAnyValue) {
    guard->GuardOn(tr, GuardLevel::GType, false);
    call_node->input(0)->MakeConstantInfo()->set_type(callable_info->GetTypeObject());
  } else if (func_type == AObject::kTypeFunction) {
    guard->GuardOn(tr, GuardLevel::GId);
    call_node->input(0)->SetConstantValue(true);
  } else {
    return false;
  }
  return true;
}

static void PrintAnfNode(std::ostream *out, mindspore::AnfNode *anf_node, const std::string &prefix) {
  auto &s = *out;
  std::string str;
  str += anf_node->DebugString(1) + " ";
  str += anf_node->abstract() == nullptr ? "<NULL>" : anf_node->abstract()->ToString();
  std::replace(str.begin(), str.end(), '\n', ' ');
  s << " AnfNode(" << anf_node << ") [" << str << "]";
}

static void PrintValueNode(std::ostream *out, FuncGraphBuilder *fg, ValueNode *node, const std::string &prefix) {
  auto &s = *out;
  s << prefix << node->ToString();
  if (fg == nullptr) {
    s << "<FuncGraphBuilder is nullptr>" << std::endl;
    return;
  }
  auto anf_node = fg->FindNodeByWrapper(node->abstract_wrapper());
  if (anf_node != nullptr) {
    bool is_any = (*mindspore::kValueAny == *anf_node->abstract()->mindspore::abstract::AbstractBase::BuildValue());
    bool is_cnst = node->IsConstantValue();
    PrintAnfNode(out, anf_node.get(), prefix);
    s << " is_constant " << (is_cnst == is_any ? "==" : "!=") << " is_any";
  } else if (node->IsConstantValue()) {
    s << " Is Constant";
  } else {
    s << " Not find AnfNode";
  }
  s << std::endl;
}

static void PrintCallNode(std::ostream *os, FuncGraphBuilder *fg, CallNode *node, int depth) {
  std::string prefix(depth * kTwo, ' ');
  auto &s = *os;
  s << prefix << "{ inline stat " << GetInlineReasonDesc(node->GetInlineReason());
  if (!node->GetParams().empty()) {
    s << ", has extra operations to handle parameters:" << std::endl;
    for (const auto &ex : node->GetParams()) {
      s << prefix << "  " << ex->ToString() << std::endl;
      PrintValueNode(&s, fg, ex, prefix);
    }
  }
  s << std::endl;
  if (node->GetSubGraph() != nullptr) {
    s << node->GetSubGraph()->ToString(depth + 1);
  }
  s << prefix << "}" << std::endl;
}

void Graph::PrintFrame(std::ostream *out, const std::string &prefix) const {
  auto &s = *out;
  const auto &f = GetFrame(0);
  auto *fg = func_graph_builder_.get();
  s << "locals:" << std::endl;
  for (const auto &node : f.GetLocals()) {
    if (node != &ValueNode::kUnboundLocal) {
      PrintValueNode(&s, fg, node, prefix);
    }
  }
  s << "stacks:" << std::endl;
  for (auto iter = f.GetStacks().rbegin(), end = f.GetStacks().rend(); iter != end; ++iter) {
    PrintValueNode(&s, fg, *iter, prefix);
  }
  s << "cell free:" << std::endl;
  for (const auto &node : f.GetClosures()) {
    PrintValueNode(&s, fg, node, prefix);
  }
}

std::string Graph::ToString(int depth) const {
  std::stringstream s;
  std::string prefix(depth << 1, ' ');
  std::string code_name = co_.ptr() != nullptr ? std::string(py::str(co_.ptr())) : "<no code>";

  s << prefix << "*** Trace Nodes [" << code_name << "] ***" << std::endl;
  if (depth == 0) {
    PrintFrame(&s, prefix);
    s << std::endl;
  }

  s << prefix << "Graph " << this << " Nodes:" << std::endl;
  for (auto i : GetTracedNodes()) {
    PrintValueNode(&s, func_graph_builder().get(), i, prefix);
    if (i->GetType() == AbstractNode::Call) {
      PrintCallNode(&s, func_graph_builder().get(), static_cast<CallNode *>(i), depth);
    }
  }
  if (GetRetVal()) {
    s << prefix << "Return the Node " << GetRetVal() << std::endl;
  }
  s << prefix << GetStopTraceReasonDesc(GetStopTraceReason()) << std::endl;
  s << prefix << "break bci: " << GetStopTraceBci();
  if (GetStopTraceBci() != -1) {
    const auto &instr = cfg_->instr_pool()[GetStopTraceBci()];
    s << " " << instr->ToString() << " line " << instr->line() << " at " << PyUnicode_AsUTF8(GetCodeObj()->co_filename);
  }
  s << std::endl << std::endl;

  if (depth == 0) {
    this->DumpBreakInfo(&s);
  }
  return s.str();
}

static void TraceInferFailed(std::ostream *out, ValueNode *node, int depth = 0) {
  auto &s = *out;
  std::string prefix(IntToSize(depth) << 1, ' ');
  s << prefix << node << " ";
  switch (node->GetType()) {
    case AbstractNode::Call:
    case AbstractNode::Value:
      s << "ValueNode " << node;
      break;
    case AbstractNode::Param:
      s << "Parameter (" << node->GetOparg() << ")" << node;
      break;
    case AbstractNode::CellVar:
    case AbstractNode::FreeVar:
      s << "Closure (" << node->GetOparg() << ")" << node;
      break;
    case AbstractNode::kUnbound:
      s << "(kUnboundLocal)";
      break;
    default:
      break;
  }
  s << " object is ";
  PyObject *op = node->GetVobj() ? node->GetVobj()->GetPyObject().ptr() : nullptr;
  if (op != nullptr) {
    s << AObject::ToString(op);
    return;
  }
  s << "<NULL>:" << std::endl;
  for (size_t i = 0; i < node->getInputs().size(); ++i) {
    TraceInferFailed(out, node->input(i), depth + 1);
  }
}

void DumpUnsupportedByteCodeInfo(std::ostream &s, Opcode op, int arg) {
  if (op == SETUP_WITH || op == SETUP_FINALLY) {
    s << op.name() << " " << arg << " is skipped in break_graph or a exception happened.\n";
  } else {
    s << op.name() << " " << arg << " is not support.\n";
  }
}

void Graph::DumpBreakInfo(std::ostream *out) const {
  auto &s = *out;
  if (GetRetVal()) {
    TraceInferFailed(out, GetRetVal());
  }
  if (GetStopTraceBci() == -1) {
    return;
  }
  const auto &f = GetFrame(GetStopTraceBci());
  const auto &nodes = GetTracedNodes();
  const auto &instrs = cfg_->instr_pool();
  int break_bci = GetStopTraceBci();

  s << "graph break at: " << break_bci << " " << instrs[break_bci]->ToString() << std::endl;
  std::vector<ValueNode *> parameters;
  if (nodes.size() == 0 || nodes.back()->bci() < break_bci) {
    // break at unsupported bytecode
    Opcode op(instrs[break_bci]->op());
    int arg = instrs[break_bci]->arg();
    DumpUnsupportedByteCodeInfo(s, op, arg);
    if (op == POP_JUMP_IF_FALSE || op == POP_JUMP_IF_TRUE || op == JUMP_IF_FALSE_OR_POP || op == JUMP_IF_TRUE_OR_POP ||
        op == FOR_ITER || op == UNPACK_SEQUENCE || op == UNPACK_EX) {
      arg = 0;
    } else if (op == CALL_FUNCTION_EX) {
      arg = (arg & 0x01) + 1;
    } else if (op == CALL_FUNCTION_KW) {
      arg++;
    } else {
      return;
    }
    for (int i = arg; i >= 0; --i) {
      parameters.push_back(f.Peek(i));
    }
  } else {
    auto iter = std::find_if(nodes.begin(), nodes.end(), [break_bci](ValueNode *n) { return n->bci() == break_bci; });
    MS_EXCEPTION_IF_CHECK_FAIL(iter != nodes.end(), "can't find break info.");
    parameters.push_back(*iter);
  }
  // print traced value
  for (auto node : parameters) {
    TraceInferFailed(out, node);
  }
}

std::string FrameStates::ToString() const {
  std::stringstream s;
  s << "locals:\n";
  for (size_t i = 0; i < locals.size(); ++i) {
    if (locals[i] != &ValueNode::kUnboundLocal) {
      s << i << ": " << locals[i]->ToString() << "\n";
    }
  }
  s << "\nstacks:\n";
  std::for_each(stack.rbegin(), stack.rend(), [&s](ValueNode *i) { s << i->ToString() << "\n"; });
  s << "\ncell free:\n";
  std::for_each(cell_free.begin(), cell_free.end(), [&s](ValueNode *i) { s << i->ToString() << "\n"; });
  s << "\n";
  return s.str();
}

}  // namespace pijit
}  // namespace mindspore
