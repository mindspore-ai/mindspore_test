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
#include "pipeline/jit/pi/graph_capture/code_generator.h"
#include <set>
#include <regex>
#include "pipeline/jit/pi/graph_capture/local_liveness.h"
#include "pipeline/jit/pi/graph_capture/graph.h"
#include "pipeline/jit/pi/graph_capture/cfg.h"
#include "pipeline/jit/pi/graph_capture/side_effect.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/pi/runtime.h"
#include "pipeline/jit/pi/external.h"
#include "pipeline/jit/pi/graph_compiler/compiler.h"

#ifndef PY_MAKECODEUNIT
#ifdef WORDS_BIGENDIAN
#define PY_MAKECODEUNIT(opcode, oparg) (MS_ASSERT((opcode) < NO_IMPL_OPCODE), ((opcode) << 8) | (oparg))
#else
#define PY_MAKECODEUNIT(opcode, oparg) (MS_ASSERT((opcode) < NO_IMPL_OPCODE), (opcode) | ((oparg) << 8))
#endif
#endif

namespace mindspore {
namespace pijit {
constexpr const size_t MoveEightBits = 8;

class GraphParameterBuilder {
 public:
  static std::string Key(int, ValueNode *n);

  void Init(const std::vector<ValueNode *> &args, const std::vector<ValueNode *> &globals, ValueNode *vargs,
            ValueNode *kwargs);
  void Build(const std::unordered_map<ValueNode *, int> &locals);

  std::vector<ValueNode *> args_;
  std::vector<ValueNode *> globals_;
  std::vector<std::unique_ptr<Instr>> load_;  // load parameters and store parameters to global, for caller
  std::vector<std::unique_ptr<Instr>> dele_;  // delete global parameters, for caller
  std::vector<std::unique_ptr<Instr>> sort_;  // load global parameter and store to locals, for callee
  ValueNode *vargs_;
  ValueNode *kwargs_;

 private:
  void BuildVargs(const std::unordered_map<ValueNode *, int> &locals);
  void BuildKwVargs(const std::unordered_map<ValueNode *, int> &locals);
};

static bool FindBlock(int start_bci, const CFG *cfg, int *target_bci, int *stack_effect);
std::string PrintInstr(const std::vector<std::unique_ptr<Instr>> &list);
std::string PrintNodeSet(const NodeSet &);

std::string GenerateObjectKey(const py::object &value) {
  PyTypeObject *tp = Py_TYPE(value.ptr());
  std::stringstream s;
  s << (tp->tp_name ? tp->tp_name : "<unnamed>");
  if (tp == &PyFunction_Type) {
    s << "[" << PyUnicode_AsUTF8(reinterpret_cast<PyFunctionObject *>(value.ptr())->func_qualname) << "]";
  }
  if (tp == &PyModule_Type) {
    s << "[" << PyModule_GetName(value.ptr()) << "]";
  }
  s << "<" << value.ptr() << ">";
  return s.str();
}

void MapAdd(const py::dict &dict, const std::string &key, const py::object &value, std::string *rename) {
  py::str key_object(key);
  PyObject *old = PyDict_GetItem(dict.ptr(), key_object.ptr());
  if (old == value.ptr()) {
    return;
  }
  if (old == nullptr) {
    PyDict_SetItem(dict.ptr(), key_object.ptr(), value.ptr());
    return;
  }
  if (rename != nullptr) {
    std::string new_key = GenerateObjectKey(value);
    if (new_key != key) {
      PyDict_SetItem(dict.ptr(), py::str(new_key).ptr(), value.ptr());
      *rename = new_key;
      return;
    }
  }
  MS_LOG(INTERNAL_EXCEPTION) << "duplicate dict value, key: " << key << ", old value at " << old << ": "
                             << std::string(py::str(old)) << " -> new value at " << value.ptr() << ": "
                             << std::string(py::str(value.ptr()));
}

static int GetOpcodeMaxStackEffect(int op, int arg, bool jump) {
  int off;
  off = PyCompile_OpcodeStackEffect(op, arg);
  if (op == NOP || op == EXTENDED_ARG) {
    return 0;
  }
  if (op == END_FINALLY) {
    return -1;
  }
  return off;
}

int CodeGenerator::CalculateStackSize(const std::vector<std::unique_ptr<Instr>> &list, int sp) {
  std::unordered_map<Instr *, int> blocks;
  int max_depth = 0;
  int flag = 0;
  for (size_t i = 0; i < list.size(); ++i) {
    Instr *instr = list[i].get();
    int op = instr->op();
    int arg = instr->arg();
    Instr *jump = instr->extra_jump();
    auto iter = blocks.find(instr);
    if (iter != blocks.end()) {
      flag = 0;
      sp = iter->second;
    } else if (flag == 1) {
      continue;
    }
    if (op == RAISE_VARARGS || op == RETURN_VALUE || op == RERAISE) {
      flag = 1;
    }
    if (jump != nullptr) {
      iter = blocks.find(jump);
      int jump_sp = sp + GetOpcodeMaxStackEffect(op, arg, true);
      blocks[jump] = (iter == blocks.end()) ? jump_sp : std::max(iter->second, jump_sp);
    }
    sp += GetOpcodeMaxStackEffect(op, arg, false);
    max_depth = std::max(sp, max_depth);
  }
  return sp < 0 ? -1 : max_depth;
}

// reset bci, reset jump offset
static void CalculateOffset(const std::vector<std::unique_ptr<Instr>> &list) {
  constexpr auto InstrSize = [](unsigned arg) constexpr {
    return arg <= 0xff ? 1 : arg <= 0xffff ? 2 : arg <= 0xffffff ? 3 : 4;
  };

  bool re_calc;
  do {
    re_calc = false;
    int bci = -1;
    for (const auto &i : list) {
      bci += InstrSize(i->arg());
      i->set_bci(bci);
    }
    for (const auto &i : list) {
      int isize = InstrSize(i->arg());
      Instr *tar = i->extra_jump();
      if (tar) {
        i->set_arg(Opcode(i->op()).JumpOffset(i->bci(), tar->bci() - InstrSize(tar->arg()) + 1));
        re_calc |= isize != InstrSize(i->arg());
      }
    }
  } while (re_calc);
}

std::pair<py::bytes, py::bytes> CodeGenerator::ConvertToCodeBytes(const std::vector<std::unique_ptr<Instr>> &list,
                                                                  int first_line) {
  std::vector<char> co_lnotab;
  std::vector<_Py_CODEUNIT> co_code;

  CalculateOffset(list);

  int line = first_line > 0 ? first_line : 0;
  int bci = 0;
  for (const auto &i : list) {
    int addr_off = sizeof(_Py_CODEUNIT) * (i->bci() - bci);
    int line_off = i->line() - line;
    if (i->line() != -1 && line_off > 0 && line_off < INT8_MAX && addr_off < INT8_MAX) {
      co_lnotab.push_back(addr_off);
      co_lnotab.push_back(line_off);
      bci = i->bci();
      line = i->line();
    }
    int oparg = i->arg();
    for (unsigned c = 0, exa = IntToSize(oparg) >> MoveEightBits; exa > 0; exa >>= MoveEightBits, ++c) {
      co_code.insert(co_code.end() - c, PY_MAKECODEUNIT(EXTENDED_ARG, exa & 0xff));
    }
    co_code.push_back(PY_MAKECODEUNIT(i->op(), (signed)oparg & 0xff));
  }
  const char *code_data = reinterpret_cast<const char *>(co_code.data());
  const size_t code_size = co_code.size() * sizeof(co_code[0]);
  return {py::bytes(code_data, code_size), py::bytes(co_lnotab.data(), co_lnotab.size())};
}

static void SetNamedInstrIndex(const std::unique_ptr<Instr> &i, std::unordered_map<std::string, int> *co_names) {
  if (!Opcode(i->op()).HasName()) {
    return;
  }
  int arg;
  auto iter = co_names->find(i->name());
  if (iter != co_names->end()) {
    arg = iter->second;
  } else {
    arg = SizeToInt(co_names->size());
    co_names->insert({i->name(), arg});
  }
  i->set_arg(arg);
}

static void SetLoadConstIndex(const std::unique_ptr<Instr> &i, const py::dict &consts) {
  if (i->op() != LOAD_CONST) {
    return;
  }
  PyObject *co_consts = consts.ptr();
  PyObject *cnst = i->cnst().ptr();
  MS_EXCEPTION_IF_CHECK_FAIL(cnst != nullptr, "LOAD_CONST instruction not set object");

  PyObject *key = _PyCode_ConstantKey(cnst);
  if (key != nullptr) {
    PyObject *index = PyDict_GetItem(co_consts, key);
    Py_ssize_t arg;
    if (index != nullptr) {
      arg = PyLong_AsLong(index);
    } else {
      arg = PyDict_GET_SIZE(co_consts);
      PyDict_SetItem(co_consts, key, py::int_(arg).ptr());
    }
    i->set_arg(arg);
    Py_DECREF(key);
    if (!PyErr_Occurred()) {
      return;
    }
  }
  throw py::error_already_set();
}

static py::tuple ConstsMapToTuple(const py::dict &consts) {
  const Py_ssize_t size = PyDict_GET_SIZE(consts.ptr());
  py::tuple co_consts(size);

  PyObject *key;
  PyObject *val;
  Py_ssize_t pos = 0;
  while (PyDict_Next(consts.ptr(), &pos, &key, &val)) {
    Py_ssize_t index = PyLong_AsLong(val);
    if (PyTuple_CheckExact(key)) {
      key = PyTuple_GET_ITEM(key, 1);
    }
    Py_INCREF(key);
    PyTuple_SET_ITEM(co_consts.ptr(), index, key);
  }
  return co_consts;
}

static py::tuple NamesMapToTuple(const std::unordered_map<std::string, int> &names) {
  py::tuple co_names(names.size());
  for (const auto &i : names) {
    PyTuple_SET_ITEM(co_names.ptr(), i.second, PyUnicode_FromStringAndSize(i.first.data(), i.first.size()));
  }
  return co_names;
}

static py::object ConvertVector(const std::vector<std::string> &names, bool to_tuple = true) {
  size_t size = names.size();
  PyObject *list = to_tuple ? PyTuple_New(size) : PyList_New(size);
  for (; size > 0; --size) {
    const std::string &n = names[size - 1];
    if (to_tuple) {
      PyTuple_SET_ITEM(list, size - 1, PyUnicode_FromStringAndSize(n.data(), n.size()));
    } else {
      PyList_SET_ITEM(list, size - 1, PyUnicode_FromStringAndSize(n.data(), n.size()));
    }
  }
  return py::reinterpret_steal<py::object>(list);
}

static py::tuple FillVariableName(const std::vector<std::string> &varnames, int nlocals) {
  MS_EXCEPTION_IF_CHECK_FAIL(varnames.size() <= static_cast<size_t>(nlocals), "too small local count !!");
  std::set<std::string> vars;
  py::tuple co_varnames(nlocals);
  int size = SizeToInt(varnames.size());
  for (int i = 0; i < nlocals; ++i) {
    std::string n;
    if (i < size) {
      n = varnames[i];
    } else {
      n = std::to_string(i) + "_local";
    }
    while (vars.find(n) != vars.end()) {
      n = n + "_" + std::to_string(i);
    }
    vars.insert(n);
    PyTuple_SET_ITEM(co_varnames.ptr(), i, PyUnicode_FromStringAndSize(n.data(), n.size()));
  }
  return co_varnames;
}

static std::string AttachCodeID(const std::string &co_name) {
  static size_t id = 0;
  constexpr const char *mark = "I.";
  constexpr const char *reg_mark = "\\d+I.";
  return std::to_string(id++) + mark + std::regex_replace(co_name, std::regex(reg_mark), "");
}

static std::string MakeCompiledName(const std::string &co_name) {
  static size_t id = 0;
  constexpr const char *reg_mark = "<compile\\[\\d+\\]>";
  return "<compile[" + std::to_string(id++) + "]>" + std::regex_replace(co_name, std::regex(reg_mark), "");
}

static std::string MakeBrkName(const std::string &co_name, int bci) {
  constexpr const char *mark = "B.";
  constexpr const char *reg_mark = "\\d+B.";
  return std::to_string(bci) + mark + std::regex_replace(co_name, std::regex(reg_mark), "");
}

py::object CodeGenerator::Transform(const Code &ccode) {
  std::unordered_map<std::string, int> names;
  py::dict consts;
  int co_stacksize;

  for (const auto &i : ccode.co_code) {
    SetNamedInstrIndex(i, &names);
    SetLoadConstIndex(i, consts);
  }
  co_stacksize = CalculateStackSize(ccode.co_code);
  if (co_stacksize < 0) {
    MS_LOG(ERROR) << "\n" << PrintInstr(ccode.co_code);
    MS_EXCEPTION_IF_CHECK_FAIL(co_stacksize >= 0, "check instruction list, computer stack size failed");
  }

  std::pair<py::bytes, py::bytes> code_info = ConvertToCodeBytes(ccode.co_code, ccode.co_firstlineno);
  py::bytes co_code = std::move(code_info.first);
  py::bytes co_lnotab = std::move(code_info.second);
  py::tuple co_consts = ConstsMapToTuple(consts);
  py::tuple co_names = NamesMapToTuple(names);
  py::object co_varnames = FillVariableName(ccode.co_varnames, ccode.co_nlocals);
  py::object co_freevars = ConvertVector(ccode.co_freevars);
  py::object co_cellvars = ConvertVector(ccode.co_cellvars);
  py::str co_name(AttachCodeID(ccode.co_name));

  PyCodeObject *new_code = PyCode_New(ccode.co_argcount,        // co_argcount
                                      ccode.co_kwonlyargcount,  // co_kwonlyargcount
                                      ccode.co_nlocals,         // co_nlocals
                                      co_stacksize,             // co_stacksize
                                      ccode.co_flags,           // co_flags
                                      co_code.ptr(),            // co_code
                                      co_consts.ptr(),          // co_consts
                                      co_names.ptr(),           // co_names
                                      co_varnames.ptr(),        // co_varnames
                                      co_freevars.ptr(),        // co_freevars
                                      co_cellvars.ptr(),        // co_cellvars
                                      ccode.co_filename.ptr(),  // co_filename
                                      co_name.ptr(),            // co_name
                                      ccode.co_firstlineno,     // co_firstlineno
                                      co_lnotab.ptr());         // co_lnotab

  if (new_code != nullptr) {
    return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(new_code));
  }
  throw py::error_already_set();
}

std::vector<std::unique_ptr<Instr>> CodeGenerator::CopyInstr(const std::vector<std::unique_ptr<Instr>> &list,
                                                             size_t start_bci, size_t end_bci) {
  std::vector<std::pair<size_t, size_t>> edges;
  std::vector<std::unique_ptr<Instr>> instrs;

  bool insert_nop_to_end = false;
  size_t size = std::min(list.size(), end_bci);
  for (size_t bci = start_bci; bci < size; ++bci) {
    const auto &i = list[bci];
    size_t index = (size_t)i->bci() - start_bci;
    instrs.emplace_back(std::make_unique<Instr>(i->op(), i->arg(), index, i->line()));
    instrs.back()->set_name(i->name());
    instrs.back()->set_cnst(i->cnst());
    if (i->op() == LOAD_METHOD) {
      instrs.back()->set_op(LOAD_ATTR);
    } else if (i->op() == CALL_METHOD) {
      instrs.back()->set_op(CALL_FUNCTION);
    }
    if (i->extra_jump()) {
      size_t tar = i->extra_jump()->bci();
      bool valid = i->bci() == SizeToInt(bci) && start_bci <= tar && tar <= size;
      if (!valid) {
        MS_LOG(INTERNAL_EXCEPTION) << "check instruction index failed," << i->bci() << " == " << bci << " && "
                                   << start_bci << " <= " << tar << " && " << tar << " <= " << size;
      }
      insert_nop_to_end |= (tar == size);
      edges.push_back({index, tar - start_bci});
    }
  }
  if (insert_nop_to_end) {
    instrs.emplace_back(std::make_unique<Instr>(NOP, 0, instrs.size()));
  }
  for (const auto &i : edges) {
    instrs[i.first]->set_extra_jump(instrs[i.second].get());
  }
  return instrs;
}

void CodeGenerator::EraseUnusedInstr(std::vector<std::unique_ptr<Instr>> *list) {
  auto NeedRemove = [](const std::vector<std::unique_ptr<Instr>>::iterator &i) {
    int op = (*i)->op();
    if (op == NOP || op == EXTENDED_ARG) {
      return true;
    }
    if (op == JUMP_ABSOLUTE || op == JUMP_FORWARD) {  // jump to next
      return (*i)->extra_jump() == (i + 1)->get();
    }
    return false;
  };
  // mark unused instruction
  auto erase_iter = list->begin();
  int bci = 0;
  for (auto i = list->begin(); i != list->end(); ++i) {
    if (NeedRemove(i)) {
      (*i)->set_bci(-1);
      (*i)->set_extra_jump((i + 1)->get());
    } else {
      (*i)->set_bci(bci);
      std::swap(*erase_iter, *i);
      ++erase_iter;
      ++bci;
    }
  }
  if (erase_iter == list->end()) {
    return;
  }
  // reset jump
  for (auto i = list->begin(); i != erase_iter; ++i) {
    Instr *tar = (*i)->extra_jump();
    if (tar == nullptr) {
      continue;
    }
    while (tar->bci() == -1) {
      MS_EXCEPTION_IF_NULL(tar->extra_jump());
      tar = tar->extra_jump();
    }
    (*i)->set_extra_jump(tar);
  }
  list->erase(erase_iter, list->end());
}

void CodeGenerator::EraseUnusedInstr() { EraseUnusedInstr(&code_.co_code); }

std::vector<std::unique_ptr<Instr>> CodeGenerator::RotStack(int stack) {
  std::vector<std::unique_ptr<Instr>> res;
  if (stack == 0) {
    return res;
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 10)
  } else {
    res.push_back(std::make_unique<Instr>(ROT_N, stack + 1));
#elif PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 10 && PY_MINOR_VERSION >= 7
  } else if (stack == 1) {
    res.push_back(std::make_unique<Instr>(ROT_TWO));
  } else if (stack == 2) {
    res.push_back(std::make_unique<Instr>(ROT_THREE));
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION > 7)
  } else if (stack == 3) {
    res.push_back(std::make_unique<Instr>(ROT_FOUR));
#endif
  } else {
    MS_LOG(DEBUG) << ("too many stack value, will build tuple to process\n");
    res.insert(res.begin(), std::make_unique<Instr>(BUILD_TUPLE, stack));
    res.insert(res.begin(), std::make_unique<Instr>(UNPACK_SEQUENCE, stack));
    res.insert(res.begin(), std::make_unique<Instr>(BUILD_TUPLE, stack));  // reverse tuple
    res.push_back(std::make_unique<Instr>(ROT_TWO));
    res.push_back(std::make_unique<Instr>(UNPACK_SEQUENCE, stack));
#endif
  }

  return res;
}

std::string CodeGenerator::PrintAlive() const {
  std::stringstream s;
  std::unordered_map<int, std::vector<ValueNode *>> sorted;
  for (const auto &i : nodes_alive_) {
    sorted[i.second].push_back(i.first);
  }
  for (const auto &i : sorted) {
    s << i.first << ": ";
    for (const auto &node : i.second) {
      s << node << " ";
    }
    s << "\n";
  }
  return s.str();
}

/**
 * traverse all values in reverse order, set alive time for each input value
 * the inputs of all values is before these values
 */
void CodeGenerator::MarkAlive() {
  for (auto i : nodes_->outputs) {
    MarkAlive(i);
  }
  for (int index = nodes_->operations.size() - 1; index >= 0; --index) {
    ValueNode *node = nodes_->operations[index];
    for (auto input : node->getInputs()) {
      int cur = nodes_alive_[input];
      nodes_alive_[input] = std::max(cur, index);
    }
  }
}

int CodeGenerator::AllocLocal(ValueNode *node, int index) {
  auto iter = locals_map_.find(node);
  if (iter != locals_map_.end()) {
    return iter->second;
  }
  int res;
  std::set<int> used_slots;  // order set
  for (iter = locals_map_.begin(); iter != locals_map_.end(); ++iter) {
    if (index != INT_MAX && nodes_alive_[iter->first] <= index) {
      res = iter->second;
      locals_map_.erase(iter);
      locals_map_.insert({node, res});
      return res;
    }
    used_slots.insert(iter->second);
  }
  res = 0;
  for (auto i = used_slots.begin(); i != used_slots.end() && res == (*i); ++i, ++res) {
  }
  locals_map_.insert({node, res});
  SetLocalsCount(res);
  return res;
}

void CodeGenerator::NewInstr(int op, int arg, int line) {
  code_.co_code.emplace_back(std::make_unique<Instr>(op, arg, -1, line));
}

void CodeGenerator::AddInstr(std::unique_ptr<Instr> &&instr) { code_.co_code.emplace_back(std::move(instr)); }

void CodeGenerator::AddInstrs(std::vector<std::unique_ptr<Instr>> &&l) {
  code_.co_code.insert(code_.co_code.end(), std::make_move_iterator(l.begin()), std::make_move_iterator(l.end()));
}

void CodeGenerator::LoadValue(ValueNode *node) {
  auto iter = locals_map_.find(node);
  if (iter != locals_map_.end()) {
    NewInstr(LOAD_FAST, iter->second);
    return;
  }
  if (node->GetType() == ValueNode::CellVar || node->GetType() == ValueNode::FreeVar) {
    int index = static_cast<CellVarNode *>(node)->GetIndex();
    if (index < 0) {
      LoadConst(py::reinterpret_steal<py::object>(PyCell_New(nullptr)));
    } else {
      NewInstr(LOAD_CLOSURE, index);
    }
    return;
  }
  int opcode = node->GetOpcode();
  if (opcode == LOAD_DEREF) {
    NewInstr(opcode, node->GetOparg());
    return;
  }
  std::string key = node->GetName();
  if (opcode == LOAD_GLOBAL) {
    PyObject *globals = node->GetGraph() ? node->GetGraph()->GetGlobals().ptr() : nullptr;
    MS_EXCEPTION_IF_NULL(globals);
    if (globals != GetGlobals().ptr()) {
      py::str key_object(key);
      PyObject *value = PyObject_GetItem(globals, key_object.ptr());
      if (value != nullptr) {
        py::object handle_value = py::reinterpret_steal<py::object>(value);
        MapAdd(GetGlobals(), key, handle_value, &key);
      } else {
        // name error, global undefined
        PyErr_Clear();
      }
    }
    NewInstr(LOAD_GLOBAL);
    code_.co_code.back()->set_name(key);
    return;
  }

  py::object cnst = node->GetVobj()->GetPyObject();
  if (opcode == LOAD_CONST) {
    LoadConst(cnst);
    return;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "missing value, [" << node->ToString() << "]";
}

void CodeGenerator::LoadConst(const py::object &cnst) {
  MS_EXCEPTION_IF_NULL(cnst.ptr());
  if (CheckConstPyObject(cnst.ptr())) {
    NewInstr(LOAD_CONST);
    code_.co_code.back()->set_cnst(cnst);
    return;
  }
  std::string key = GenerateObjectKey(cnst);
  MapAdd(GetGlobals(), key, cnst);
  NewInstr(LOAD_GLOBAL);
  code_.co_code.back()->set_name(key);
}

void CodeGenerator::BuildOper(ValueNode *node, int index) {
  static const std::set<int> not_value_oper = {
    STORE_DEREF,  DELETE_DEREF,  STORE_GLOBAL, DELETE_GLOBAL, STORE_ATTR, DELETE_ATTR,
    STORE_SUBSCR, DELETE_SUBSCR, IMPORT_STAR,  RAISE_VARARGS, RERAISE,
  };
  static const std::unordered_map<int, int> const_arg_oper = {
    {LIST_APPEND, 1}, {LIST_EXTEND, 1}, {DICT_MERGE, 1}, {DICT_UPDATE, 1}, {SET_UPDATE, 1}, {SET_ADD, 1}, {MAP_ADD, 2},
  };

  if (IsNonLocalValue(node)) {
    return;
  }

  for (auto param : node->getInputs()) {
    LoadValue(param);
  }
  int op = node->GetOpcode();
  int arg = pijit::Opcode(op).HasArg() ? node->GetOparg() : 0;
  auto const_arg_oper_iter = const_arg_oper.find(op);
  if (const_arg_oper_iter != const_arg_oper.end()) {
    arg = const_arg_oper_iter->second;
  }
  NewInstr(op, arg, node->GetLineNo());
  code_.co_code.back()->set_name(node->GetName());

  if (not_value_oper.find(op) != not_value_oper.end()) {
    return;
  }
  if (nodes_alive_[node] == 0) {
    NewInstr(POP_TOP);
  } else {
    NewInstr(STORE_FAST, AllocLocal(node, index), node->GetLineNo());
  }
}

void CodeGenerator::Init() {
  const int size = SizeToInt(nodes_->inputs.size());
  code_.co_nlocals = size;
  for (int i = 0; i < size; ++i) {
    ValueNode *param = nodes_->inputs[i];
    locals_map_[param] = i;
    MS_EXCEPTION_IF_CHECK_FAIL(!IsNonLocalValue(param), "got nonlocal parameter node: " + param->ToString());
  }
}

void CodeGenerator::Build() {
  // build operations
  MarkAlive();
  for (size_t index = 0; index < nodes_->operations.size(); ++index) {
    BuildOper(nodes_->operations[index], index);
  }
  SetLocalsCount(locals_map_.size());
}

void CodeGenerator::GenReturn() {
  for (const auto &i : nodes_->outputs) {
    LoadValue(i);
  }
  if (nodes_->outputs.size() > 1) {
    NewInstr(BUILD_TUPLE, nodes_->outputs.size());
  }
  if (nodes_->outputs.size() == 0) {
    NewInstr(LOAD_CONST, 0);
    code_.co_code.back()->set_cnst(py::none());
  }
  NewInstr(RETURN_VALUE);
  SetLocalsCount(locals_map_.size());
}

static bool IsNotNeedTrack(const std::vector<std::unique_ptr<Instr>> &list, int start = -1) {
  if (list.empty() || start == -1) {
    return true;
  }
  auto iter = std::find_if(list.begin() + start, list.end(), [](const std::unique_ptr<Instr> &i) {
    return Opcode(i->op()).IsCall() || Opcode(i->op()).IsBinaryMath();
  });
  return iter == list.end();
}

static std::vector<std::unique_ptr<Instr>> MakeFunc(const py::object &code, const std::string &name, int closures) {
  std::vector<std::unique_ptr<Instr>> instrs;
  for (int i = 0; i < closures; ++i) {
    instrs.emplace_back(std::make_unique<Instr>(LOAD_CLOSURE, i));
  }
  unsigned make_oparg = 0;
  if (closures != 0) {
    make_oparg |= 0x08;
    instrs.emplace_back(std::make_unique<Instr>(BUILD_TUPLE, closures));
  }
  instrs.emplace_back(std::make_unique<Instr>(LOAD_CONST, 0, code));
  instrs.emplace_back(std::make_unique<Instr>(LOAD_CONST, 0, py::str(name)));
  instrs.emplace_back(std::make_unique<Instr>(MAKE_FUNCTION, make_oparg));
  return instrs;
}

std::vector<std::string> CodeBreakGenerator::GetClosureNames() const {
  std::vector<std::string> names;
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(co_->co_cellvars); ++i) {
    names.push_back(PyUnicode_AsUTF8(PyTuple_GET_ITEM(co_->co_cellvars, i)));
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(co_->co_freevars); ++i) {
    names.push_back(PyUnicode_AsUTF8(PyTuple_GET_ITEM(co_->co_freevars, i)));
  }
  return names;
}

py::object CodeBreakGenerator::MakeCapturedCode(std::vector<std::unique_ptr<Instr>> &&load_oper,  // prepare parameters
                                                int argc, unsigned code_flag) const {
  CodeGenerator code_gen(&captured_);
  code_gen.SetGlobals(GetGlobals());
  code_gen.Init();
  code_gen.AddInstrs(std::move(load_oper));
  code_gen.Build();
  code_gen.GenReturn();

  unsigned flags = co_->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS);
  code_gen.SetArgsInfo(argc, 0);
  code_gen.SetCodeFlags(flags | code_flag);
  code_gen.SetFirstLineNumber(captured_.operations[0]->GetLineNo());
  code_gen.SetFreeVariableNames(GetClosureNames());
  code_gen.SetCodeName(MakeCompiledName(py::str(co_->co_name)));
  code_gen.SetFileName(py::cast<py::object>(co_->co_filename));

  code_gen.EraseUnusedInstr();
  py::object code = CodeGenerator::Transform(code_gen.GetCode());
  auto parent = GetJitCompileResults(co_);
  JitCompileResults *child = CreateJitCompileResults(code.ptr());
  child->set_stat(JitCompileResults::GRAPH_CAPTURED);
  child->set_conf(parent->conf());
  child->set_tbs(parent->tbs());
  return code;
}

void CodeBreakGenerator::CallCapturedCode(CodeGenerator *code_gen) {
  if (captured_.operations.empty()) {
    return;
  }
  GraphParameterBuilder param_info;
  BuildGraphParameters(code_gen->GetLocalsMap(), &param_info);
  int flag = (param_info.vargs_ ? CO_VARARGS : 0) | (param_info.kwargs_ ? CO_VARKEYWORDS : 0);
  py::object code = MakeCapturedCode(std::move(param_info.sort_), param_info.args_.size(), flag);

  int closures = PyTuple_GET_SIZE(co_->co_cellvars) + PyTuple_GET_SIZE(co_->co_freevars);
  code_gen->AddInstrs(MakeFunc(code, "<pijit.compile>", closures));
  code_gen->AddInstrs(std::move(param_info.load_));
  if (flag) {
    code_gen->NewInstr(CALL_FUNCTION_EX, static_cast<bool>(flag & CO_VARKEYWORDS));
  } else {
    code_gen->NewInstr(CALL_FUNCTION, param_info.args_.size());
  }
  extra_local_ = code_gen->AllocLocal(nullptr);
  code_gen->NewInstr(STORE_FAST, extra_local_);
  code_gen->AddInstrs(std::move(param_info.dele_));
}

void CodeBreakGenerator::FixInterpretOuput(CodeGenerator *code_gen) {
  if (captured_.outputs.empty()) {
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(extra_local_ != -1, "can't find graph output");
  code_gen->NewInstr(LOAD_FAST, extra_local_);
  if (captured_.outputs.size() > 1) {
    code_gen->NewInstr(UNPACK_SEQUENCE, captured_.outputs.size());
  }
  std::for_each(captured_.outputs.begin(), captured_.outputs.end(), [code_gen](ValueNode *i) {
    // fill interpret local map
    code_gen->NewInstr(STORE_FAST, code_gen->AllocLocal(i));
  });
  // reconstruct interpret values if need
}

void CodeBreakGenerator::RestoreStack(CodeGenerator *code_gen) const {
  auto begin = interpret_.outputs.begin();
  auto end = interpret_.outputs.end() - alive_locals_.size();
  std::for_each(begin, end, [code_gen](ValueNode *i) { code_gen->LoadValue(i); });
}

void CodeBreakGenerator::RestoreLocals(CodeGenerator *code_gen, bool only_load) const {
  auto begin = interpret_.outputs.end() - alive_locals_.size();
  auto end = interpret_.outputs.end();
  if (only_load) {
    std::for_each(begin, end, [code_gen](ValueNode *i) { code_gen->LoadValue(i); });
    return;
  }
  std::vector<std::unique_ptr<Instr>> st;
  auto index_iter = alive_locals_.begin();
  for (auto node_iter = begin; node_iter != end; ++node_iter, ++index_iter) {
    auto target = code_gen->GetLocalsMap().find(*node_iter);
    if (target != code_gen->GetLocalsMap().end() && target->second == *index_iter) {
      continue;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(index_iter != alive_locals_.end(), "error alive local");
    code_gen->LoadValue(*node_iter);
    st.push_back(std::make_unique<Instr>(STORE_FAST, *index_iter));
  }
  std::reverse(st.begin(), st.end());
  code_gen->AddInstrs(std::move(st));
}

py::object CodeBreakGenerator::MakeUntrackedCode(int untracked_bci, int untracked_stack_effect) const {
  const int argc = SizeToInt(interpret_.outputs.size()) + untracked_stack_effect;
  int stack_count = argc - SizeToInt(alive_locals_.size());

  std::vector<std::unique_ptr<Instr>> ld;
  std::vector<std::unique_ptr<Instr>> st;
  for (int i = 0; i < stack_count; ++i) {
    ld.emplace_back(std::make_unique<Instr>(LOAD_FAST, i));
  }
  int index = stack_count;
  for (auto iter = alive_locals_.begin(); iter != alive_locals_.end(); ++iter, ++index) {
    if (*iter != index) {
      ld.emplace_back(std::make_unique<Instr>(LOAD_FAST, index));
      st.emplace_back(std::make_unique<Instr>(STORE_FAST, *iter));
    }
  }

  std::vector<std::unique_ptr<Instr>> list = std::move(ld);
  std::move(st.rbegin(), st.rend(), std::back_inserter(list));
  std::vector<std::unique_ptr<Instr>> untracked = CodeGenerator::CopyInstr(GetCFG()->instr_pool(), untracked_bci);
  int first_line = untracked[0]->bci();
  std::move(untracked.begin(), untracked.end(), std::back_inserter(list));

  int nlocals = GetCFG()->GetLocalCount();

  CodeGenerator::Code ccode = {
    argc,
    0,
    std::max(argc, nlocals),
    (signed)co_->co_flags & ~(CO_VARARGS | CO_VARKEYWORDS),
    first_line,
    std::move(list),
    py::cast<std::vector<std::string>>(co_->co_varnames),
    std::vector<std::string>(),
    GetClosureNames(),
    MakeBrkName(PyUnicode_AsUTF8(co_->co_name), untracked_bci),
    py::reinterpret_borrow<py::object>(co_->co_filename),
  };
  CodeGenerator::EraseUnusedInstr(&ccode.co_code);
  py::object code = CodeGenerator::Transform(ccode);
  auto parent = GetJitCompileResults(co_);
  JitCompileResults *child = CreateJitCompileResults(code.ptr());
  child->set_stat(JitCompileResults::GRAPH_CANDIDATE);
  child->set_conf(parent->conf());
  child->set_tbs(parent->tbs());
  return code;
}

void CodeBreakGenerator::ReconstructStack(CodeGenerator *code_gen, int untracked_bci,
                                          int untracked_stack_effect) const {
  const auto &instr = GetCFG()->instr_pool()[break_bci_];
  if (break_bci_ == untracked_bci) {
    RestoreStack(code_gen);
    return;
  }
  if (instr->op() != CALL_FUNCTION && instr->op() != CALL_FUNCTION_KW) {
    RestoreStack(code_gen);
    code_gen->AddInstrs(CodeGenerator::CopyInstr(cfg_->instr_pool(), break_bci_, untracked_bci));
    return;
  }

  // (chaiyouheng): replace function call, mark function to compile ...
  RestoreStack(code_gen);
  code_gen->NewInstr(instr->op(), instr->arg(), instr->line());
}

void CodeBreakGenerator::BreakAtIf(CodeGenerator *code_gen) const {
  const auto &list = GetCFG()->instr_pool();
  int op = list[break_bci_]->op();
  int stack_effect = -1;
  int stack_count = SizeToInt(interpret_.outputs.size() - alive_locals_.size());
  int closures = PyTuple_GET_SIZE(co_->co_cellvars) + PyTuple_GET_SIZE(co_->co_freevars);
  py::object code;

  MS_EXCEPTION_IF_CHECK_FAIL(stack_count >= 1, "error stack");

  RestoreStack(code_gen);
  code_gen->NewInstr(op);
  Instr *if_instr = code_gen->GetCode().co_code.back().get();

  // fall-branch
  code = MakeUntrackedCode(break_bci_ + 1, stack_effect);
  code_gen->AddInstrs(MakeFunc(code, "<pijit.resume>", closures));
  code_gen->AddInstrs(CodeGenerator::RotStack(stack_count + stack_effect));
  RestoreLocals(code_gen, true);
  code_gen->NewInstr(CALL_FUNCTION, interpret_.outputs.size() + stack_effect);
  code_gen->NewInstr(RETURN_VALUE);

  // jump-branch
  stack_effect = (op == JUMP_IF_TRUE_OR_POP || op == JUMP_IF_FALSE_OR_POP) ? 0 : -1;
  code = MakeUntrackedCode(list[break_bci_]->extra_jump()->bci(), stack_effect);
  auto jump_branch = MakeFunc(code, "<pijit.resume>", closures);
  if_instr->set_extra_jump(jump_branch.begin()->get());
  code_gen->AddInstrs(std::move(jump_branch));
  code_gen->AddInstrs(CodeGenerator::RotStack(stack_count + stack_effect));
  RestoreLocals(code_gen, true);
  code_gen->NewInstr(CALL_FUNCTION, interpret_.outputs.size() + stack_effect);
  code_gen->NewInstr(RETURN_VALUE);
}

void CodeBreakGenerator::BreakAtBlock(CodeGenerator *code_gen, int untracked_bci, int untracked_stack_effect) {
  RestoreStack(code_gen);
  RestoreLocals(code_gen, false);
  const auto &instr_list = GetCFG()->instr_pool();
  code_gen->AddInstrs(CodeGenerator::CopyInstr(instr_list, break_bci_, untracked_bci));

  BitMap alive = GetCFG()->liveness()->CollectAlive(untracked_bci);
  BitMap defined(alive.size());
  for (int i = break_bci_; i < untracked_bci; ++i) {
    if (instr_list[i]->op() == STORE_FAST) {
      defined.Set(instr_list[i]->arg());
    }
  }
  std::for_each(alive_locals_.begin(), alive_locals_.end(), [&defined](int i) { defined.Set(i); });
  alive.And(defined);

  alive_locals_.clear();
  for (BitMap::Iter iter(&alive, true), end(&alive, false); iter != end; ++iter) {
    alive_locals_.push_back(*iter);
  }

  interpret_.outputs.resize(alive_locals_.size(), &ValueNode::kUnboundLocal);
  untracked_stack_effect = 0;

  py::object code = MakeUntrackedCode(untracked_bci, untracked_stack_effect);
  int closures = PyTuple_GET_SIZE(co_->co_cellvars) + PyTuple_GET_SIZE(co_->co_freevars);
  code_gen->AddInstrs(MakeFunc(code, "<pijit.resume>", closures));

  for (auto i : alive_locals_) {
    code_gen->NewInstr(LOAD_FAST, i);
  }
  code_gen->NewInstr(CALL_FUNCTION, interpret_.outputs.size() + untracked_stack_effect);
  code_gen->NewInstr(RETURN_VALUE);
}

void CodeBreakGenerator::CallUntrackedCode(CodeGenerator *code_gen) {
  if (break_bci_ == -1) {
    return;
  }
  const auto &list = GetCFG()->instr_pool();
  int start_bci = break_bci_;
  int start_op = list[start_bci]->op();

  int untracked_bci;
  int untracked_stack_effect;
  bool find_block = FindBlock(start_bci, GetCFG(), &untracked_bci, &untracked_stack_effect);
  untracked_bci++;
  if (IsNotNeedTrack(GetCFG()->instr_pool(), std::min(untracked_bci + 1, SizeToInt(list.size())))) {
    RestoreStack(code_gen);
    RestoreLocals(code_gen, false);
    code_gen->AddInstrs(CodeGenerator::CopyInstr(GetCFG()->instr_pool(), break_bci_));
    return;
  }
  if (find_block) {
    BreakAtBlock(code_gen, untracked_bci, untracked_stack_effect);
    return;
  }
  if (start_op == JUMP_IF_FALSE_OR_POP || start_op == JUMP_IF_TRUE_OR_POP || start_op == POP_JUMP_IF_FALSE ||
      start_op == POP_JUMP_IF_TRUE) {
    BreakAtIf(code_gen);
    return;
  }
  if (start_op != JUMP_ABSOLUTE && start_op != JUMP_FORWARD) {
    MS_EXCEPTION_IF_CHECK_FAIL(list[start_bci]->extra_jump() == nullptr, "unexpected jump instruction");
    // break at unsupported bytecode
    untracked_stack_effect = PyCompile_OpcodeStackEffect(start_op, list[start_bci]->arg());
    untracked_bci++;
  }

  py::object code = MakeUntrackedCode(untracked_bci, untracked_stack_effect);
  int closures = PyTuple_GET_SIZE(co_->co_cellvars) + PyTuple_GET_SIZE(co_->co_freevars);
  code_gen->AddInstrs(MakeFunc(code, "<pijit.resume>", closures));

  ReconstructStack(code_gen, untracked_bci, untracked_stack_effect);
  RestoreLocals(code_gen, true);

  code_gen->NewInstr(CALL_FUNCTION, interpret_.outputs.size() + untracked_stack_effect);
  code_gen->NewInstr(RETURN_VALUE);
}

py::object CodeBreakGenerator::MakeDispatchCode() {
  auto jcr = GetJitCompileResults(co_);

  CodeGenerator code_gen(&interpret_);
  code_gen.SetGlobals(GetGlobals());
  code_gen.Init();
  for (auto i : captured_.inputs) {
    code_gen.MarkAlive(i);
  }
  code_gen.Build();

  CallCapturedCode(&code_gen);
  FixInterpretOuput(&code_gen);

  side_effect_handler_->Restore(&code_gen);
  interpret_.outputs.resize(interpret_.outputs.size() - side_effect_handler_->GetRequiredNodes().size());

  CallUntrackedCode(&code_gen);
  MakeReturn(&code_gen);

  std::string co_name = PyUnicode_AsUTF8(co_->co_name);
  co_name = std::to_string(jcr->IncCodeCount()) + "R." + co_name;

  int nlocals = SizeToInt(code_gen.GetLocalsMap().size());
  nlocals = std::max(nlocals, co_->co_nlocals);
  nlocals = std::max(nlocals, cfg_->GetLocalCount());

  ExtendCodeInfo(&code_gen, false);
  code_gen.SetLocalsCount(nlocals);
  code_gen.SetCodeName(co_name);

  code_gen.EraseUnusedInstr();
  py::object result = CodeGenerator::Transform(code_gen.GetCode());
  return result;
}

void CodeBreakGenerator::MakeReturn(CodeGenerator *code_gen) const {
  if (break_bci_ != -1) {
    // call untracked nodes
    return;
  }
  if (captured_.operations.empty()) {
    // all values is interpret produce
    code_gen->GenReturn();
    return;
  }
  // not break graph, mix interpret and graph
  ValueNode *rv = interpret_.outputs[0];
  auto iter = code_gen->GetLocalsMap().find(rv);
  if (iter != code_gen->GetLocalsMap().end() || IsNonLocalValue(rv)) {
    code_gen->LoadValue(rv);
    code_gen->NewInstr(RETURN_VALUE);
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(captured_.outputs.size() == 1 && extra_local_ != -1,
                             "can't find return value from interpret locals and graph locals");
  code_gen->NewInstr(LOAD_FAST, extra_local_);
  code_gen->NewInstr(RETURN_VALUE);
}

py::object CodeBreakGenerator::MakeCapturedCode() const {
  auto jcr = GetJitCompileResults(co_);

  CodeGenerator code_gen(&interpret_);
  code_gen.SetGlobals(GetGlobals());
  code_gen.Init();
  code_gen.Build();
  code_gen.GenReturn();

  std::string co_name = MakeCompiledName(PyUnicode_AsUTF8(co_->co_name));
  co_name = std::to_string(jcr->IncCodeCount()) + "R." + co_name;

  int nlocals = SizeToInt(code_gen.GetLocalsMap().size());
  nlocals = std::max(nlocals, co_->co_nlocals);
  nlocals = std::max(nlocals, cfg_->GetLocalCount());

  ExtendCodeInfo(&code_gen, true);
  code_gen.SetLocalsCount(nlocals);
  code_gen.SetCodeName(co_name);

  code_gen.EraseUnusedInstr();
  py::object result = CodeGenerator::Transform(code_gen.GetCode());

  JitCompileResults *child = CreateJitCompileResults(result.ptr());
  child->set_stat(JitCompileResults::GRAPH_CAPTURED);
  child->set_conf(jcr->conf());
  child->set_tbs(jcr->tbs());
  return result;
}

void CodeBreakGenerator::ExtendCodeInfo(CodeGenerator *cg, bool merge_kw_only) const {
  int argc = merge_kw_only ? (co_->co_argcount) + co_->co_kwonlyargcount : co_->co_argcount;
  int kw_only = merge_kw_only ? 0 : co_->co_kwonlyargcount;

  cg->SetArgsInfo(argc, kw_only);
  cg->SetLocalsCount(co_->co_nlocals);
  cg->SetCodeFlags(co_->co_flags);
  cg->SetFirstLineNumber(co_->co_firstlineno);
  cg->SetVariableNames(py::cast<std::vector<std::string>>(co_->co_varnames));
  cg->SetCellVariableNames(py::cast<std::vector<std::string>>(co_->co_cellvars));
  cg->SetFreeVariableNames(py::cast<std::vector<std::string>>(co_->co_freevars));
  cg->SetFileName(py::reinterpret_borrow<py::object>(co_->co_filename));
}

void CodeBreakGenerator::Init(const Graph *graph, const GraphAnalyzer &analyzer) {
  alive_locals_ = analyzer.alive_locals();
  break_bci_ = graph->GetStopTraceBci();
  cfg_ = graph->GetCFG().get();
  const GraphAnalyzer::CapturedInfo &info = analyzer.GetCaptureInfo();
  interpret_.inputs = info.interpret_.inputs;
  interpret_.outputs = info.interpret_.outputs;
  interpret_.operations = info.interpret_.operations;
  captured_.inputs = info.captured_.inputs;
  captured_.outputs = info.captured_.outputs;
  captured_.operations = info.captured_.operations;
  graph_inputs_info_.args = info.graph_inputs_.args;
  graph_inputs_info_.vargs = info.graph_inputs_.vargs;
  graph_inputs_info_.kwargs = info.graph_inputs_.kwargs;
  graph_inputs_info_.globals = info.graph_inputs_.globals;
  side_effect_handler_ = graph->GetSideEffect();

  size_t alive_count;
  if (break_bci_ != -1) {
    size_t stack_count = graph->GetFrame(break_bci_).GetStacks().size();
    size_t alive_locals_count = alive_locals_.size();
    size_t side_effect_required = side_effect_handler_->GetRequiredNodes().size();
    alive_count = stack_count + alive_locals_count + side_effect_required;
  } else {
    alive_count = 1 + side_effect_handler_->GetRequiredNodes().size();
  }
  MS_EXCEPTION_IF_CHECK_FAIL(alive_count == interpret_.outputs.size(), "error alive count");

  if (analyzer.NeedInterpret()) {
    return;
  }
  // all parameters is graph support
  captured_.inputs.clear();
  captured_.outputs.clear();
  interpret_.operations = std::move(captured_.operations);
}

const CFG *CodeBreakGenerator::GetCFG() const { return cfg_; }

void CodeBreakGenerator::BuildGraphParameters(const std::unordered_map<ValueNode *, int> &locals,
                                              GraphParameterBuilder *builder) {
  // NOTE: if *vargs is cell variable, it is not parameter node
  MS_EXCEPTION_IF_CHECK_FAIL(co_->co_nlocals == SizeToInt(interpret_.inputs.size()),
                             "interpret inputs must be same as locals");

  builder->Init(graph_inputs_info_.args, graph_inputs_info_.globals, graph_inputs_info_.vargs,
                graph_inputs_info_.kwargs);
  builder->Build(locals);

  size_t inputs_count = captured_.inputs.size();
  captured_.inputs = builder->args_;
  if (builder->vargs_ != nullptr) {
    captured_.inputs.push_back(builder->vargs_);
  }
  if (builder->kwargs_ != nullptr) {
    captured_.inputs.push_back(builder->kwargs_);
  }
  captured_.inputs.insert(captured_.inputs.end(), builder->globals_.begin(), builder->globals_.end());
  MS_EXCEPTION_IF_CHECK_FAIL(inputs_count == captured_.inputs.size(), "error parameters");
}

std::string GraphParameterBuilder::Key(int index, ValueNode *n) {
  static uint64_t kId = 0;
  PyTypeObject *tp = n->GetVobj() ? n->GetVobj()->GetTypeObject() : nullptr;
  std::string descr = AObject::GetTypeDesc(n->GetVobj() ? n->GetVobj()->GetType() : AObject::kTypeAnyValue);
  std::stringstream s;
  s << "<" << index << ">" << (tp ? (tp->tp_name ? tp->tp_name : "<unnamed>") : descr) << "<" << (kId++) << ">";
  return s.str();
}

void GraphParameterBuilder::Init(const std::vector<ValueNode *> &args, const std::vector<ValueNode *> &globals,
                                 ValueNode *vargs, ValueNode *kwargs) {
  args_ = args;
  globals_ = globals;
  vargs_ = vargs;
  kwargs_ = kwargs;
}

void GraphParameterBuilder::Build(const std::unordered_map<ValueNode *, int> &locals) {
  auto Load = [&locals](ValueNode *param) {
    auto iter = locals.find(param);
    MS_EXCEPTION_IF_CHECK_FAIL(iter != locals.end(), "can't find graph parameters from interpret locals");
    return std::make_unique<Instr>(LOAD_FAST, iter->second);
  };

  /**
   * graph parameter treat tuple, list, dict as constant
   * must be unpack these parameters and pack it by graph
   * if param is tuple or param is list:
   *   TupleRebuild(param, &load_, &sort_, &args_)
   * if param is dict:
   *   DictRebuild(param, &load_, &sort_, &args_)
   **/
  std::transform(args_.begin(), args_.end(), std::back_inserter(load_), Load);

  const int argc = SizeToInt(args_.size()) + (vargs_ != nullptr) + (kwargs_ != nullptr);
  for (size_t i = 0; i < globals_.size(); ++i) {
    std::string name = GraphParameterBuilder::Key(i, globals_[i]);
    load_.emplace_back(Load(globals_[i]));
    load_.emplace_back(std::make_unique<Instr>(STORE_GLOBAL, 0, name));
    dele_.emplace_back(std::make_unique<Instr>(DELETE_GLOBAL, 0, name));
    sort_.emplace_back(std::make_unique<Instr>(LOAD_GLOBAL, 0, name));
    sort_.emplace_back(std::make_unique<Instr>(STORE_FAST, argc + i));
  }
  if (vargs_) {
    BuildVargs(locals);
  }
  if (kwargs_) {
    BuildKwVargs(locals);
  }
}

void GraphParameterBuilder::BuildVargs(const std::unordered_map<ValueNode *, int> &locals) {
  auto iter = locals.find(vargs_);
  MS_EXCEPTION_IF_CHECK_FAIL(iter != locals.end(), "can't find graph parameters from interpret locals");
  if (args_.size() == 0) {
    load_.push_back(std::make_unique<Instr>(LOAD_FAST, iter->second));
    return;
  }

  load_.push_back(std::make_unique<Instr>(BUILD_LIST, args_.size()));
  load_.push_back(std::make_unique<Instr>(LOAD_FAST, iter->second));
#if PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 9
  const int tuple_unpack_arg = 2;
  load_.push_back(std::make_unique<Instr>(BUILD_TUPLE_UNPACK, tuple_unpack_arg));
#else
  load_.push_back(std::make_unique<Instr>(LIST_EXTEND, 1));
  load_.push_back(std::make_unique<Instr>(LIST_TO_TUPLE, 0));
#endif
}

void GraphParameterBuilder::BuildKwVargs(const std::unordered_map<ValueNode *, int> &locals) {
  auto iter = locals.find(kwargs_);
  MS_EXCEPTION_IF_CHECK_FAIL(iter != locals.end(), "can't find graph parameters from interpret locals");

  if (vargs_ == nullptr) {
    // only kwargs
    load_.push_back(std::make_unique<Instr>(BUILD_TUPLE, args_.size()));
  }
  load_.push_back(std::make_unique<Instr>(LOAD_FAST, iter->second));
}

// e.g. while..., for..., while...else..., for...else...,
static int FindLoopEnd(int start, const CFG *cfg) {
  Block *loop_begin = cfg->GetBlockByBci(start);
  if (!loop_begin->is_loop_head()) {
    return start - 1;
  }

  const auto &instrs = cfg->instr_pool();
  int loop_exit = loop_begin->begin_ci();
  int target = loop_begin->GetJumpBB() ? loop_begin->GetJumpBB()->begin_ci() : loop_exit;
  // find loop last exit
  for (; loop_exit != target; ++loop_exit) {
    Instr *jump = instrs[loop_exit]->extra_jump();
    if (jump == nullptr) {
      continue;
    }
    if (target < jump->bci()) {
      // if jump forward out of loop branch target, reset target
      target = jump->bci();
    }
  }
  // find last backward edge, get next instruction
  int result = 0;
  for (auto i : loop_begin->pred_bbs()) {
    result = std::max(result, i->end_ci());
  }
  return std::max(result, target) - 1;
}

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION < 9)

static size_t FindTryBlockEnd(int start_bci, const CFG *cfg) {
  const auto &list = cfg->instr_pool();
  size_t block_end = list[start_bci]->extra_jump()->bci();
  for (; block_end < list.size() && list[block_end]->op() != END_FINALLY; ++block_end) {
  }
  if (list[block_end - 1]->extra_jump()) {
    size_t jump = list[block_end - 1]->extra_jump()->bci();
    block_end = std::max(block_end, jump);
  }
  return block_end;
}

static bool FindBlock(int start_bci, const CFG *cfg, int *end_bci, int *stack_effect) {
  const auto &list = cfg->instr_pool();
  size_t block_end = 0;
  *stack_effect = 0;
  int opcode = list[start_bci]->op();
  if (opcode == Opcode::k_ILLEGAL_OPCODE) {
    MS_LOG(INTERNAL_EXCEPTION) << "shouldn't reach here";

#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 7)
  } else if (opcode == SETUP_EXCEPT) {
    block_end = FindTryBlockEnd(start_bci, cfg);
  } else if (opcode == SETUP_LOOP) {
    block_end = list[start_bci]->extra_jump()->bci() - 1;
  } else if (opcode == FOR_ITER) {
    block_end = FindLoopEnd(start_bci, cfg);
    *stack_effect = -1;
#endif
#if (PY_MAJOR_VERSION == 3) && (PY_MINOR_VERSION == 8)
  } else if (opcode == BEGIN_FINALLY || opcode == CALL_FINALLY) {
    MS_EXCEPTION_IF_CHECK_FAIL(false, "shouldn't reach here, must be break at SETUP_FINALLY");
  } else if (opcode == FOR_ITER) {
    block_end = FindLoopEnd(start_bci, cfg);
    *stack_effect = -1;
#endif
  } else if (opcode == SETUP_WITH || opcode == SETUP_FINALLY) {
    if (opcode == SETUP_WITH) {
      *stack_effect = -1;
    }
    block_end = FindTryBlockEnd(start_bci, cfg);
  } else {
    block_end = FindLoopEnd(start_bci, cfg);
  }
  if (list[start_bci]->op() == FOR_ITER && SizeToInt(block_end) == start_bci - 1) {
    // break at FOR_ITER and it is not a loop
    block_end = list[start_bci]->extra_jump()->bci() - 1;
  }
  *end_bci = block_end;
  return SizeToInt(block_end) != start_bci - 1;
}

#else

static int FindWithBlockEnd(int start_bci, const CFG *cfg) {
  const auto &list = cfg->instr_pool();
  size_t tar = (size_t)list[start_bci]->extra_jump()->bci();
  bool validate = tar + 1 < list.size() && list[tar]->op() == WITH_EXCEPT_START && list[tar + 1]->extra_jump();
  MS_EXCEPTION_IF_CHECK_FAIL(validate, "can't find with block");
  return list[tar - 1]->extra_jump() ? list[tar - 1]->extra_jump()->bci() - 1 : list.back()->bci();
}

// finally block has two copies in bytecodes, only test for Python3.9
static int FindFinallyBlockEnd(int raise_block, int normal_block, const CFG *cfg) {
  const auto &list = cfg->instr_pool();
  MS_EXCEPTION_IF_CHECK_FAIL(normal_block < SizeToInt(list.size()) && list[normal_block]->op() == POP_BLOCK,
                             "can't find finally block");
  auto i = normal_block + 1;
  auto j = raise_block;
  for (; list[i]->op() == list[j]->op(); ++i, ++j) {
  }
  // only python3.9, list[i]->op() == JUMP_FORWARD && list[j]->op() == RERAISE;
  return j;
}

static int FindTryBlockEnd(int start, const CFG *cfg) {
  const auto &list = cfg->instr_pool();
  Instr *tar = list[start]->extra_jump();
  MS_EXCEPTION_IF_NULL(tar);

  size_t res = (size_t)tar->bci();
  if (tar->op() == DUP_TOP) {
    // try block without finally
    MS_EXCEPTION_IF_CHECK_FAIL(res + 2 < list.size(), "can't find try block");
    while (res < list.size() && list[res]->op() != RERAISE) {
      res = list[res + 2]->extra_jump()->bci();
    }
    if (list[res - 1]->op() == JUMP_FORWARD) {
      res = list[res - 1]->extra_jump()->bci();
    }
    return res;
  }
  // finally block has two copies in bytecodes, first is normally and end with JUMP_FORWARD, second is end with RERAISE
  int reraise_finally_block_start = tar->bci();
  if (start + 1 < SizeToInt(list.size()) && list[start + 1]->op() != SETUP_FINALLY) {
    // Handle try without exception scene.
    res = start;
  } else {
    MS_EXCEPTION_IF_CHECK_FAIL(start + 1 < SizeToInt(list.size()) && list[start + 1]->op() == SETUP_FINALLY,
                               "can't find finally block");
    res = IntToSize(list[start + 1]->extra_jump()->bci());
    while (res < list.size() && list[res]->op() != RERAISE) {
      res = list[res + 2]->extra_jump()->bci();
    }
  }
  /*
    In the current situation:
      try：
        ...
      else:
        ...
      finally:
        ...
      this codes have a else block, wo should find byteCode 'POP_BLOCK'
  */
  while (res < list.size() && list[res]->op() != POP_BLOCK) {
    res++;
  }
  return FindFinallyBlockEnd(reraise_finally_block_start, res, cfg);
}

static bool FindBlock(int start_bci, const CFG *cfg, int *end_bci, int *stack_effect) {
  const std::vector<std::unique_ptr<Instr>> &list = cfg->instr_pool();
  *stack_effect = 0;
  int opcode = list[start_bci]->op();
  if (opcode == SETUP_FINALLY) {
    *end_bci = FindTryBlockEnd(start_bci, cfg);
    return true;
  } else if (opcode == SETUP_WITH) {
    *end_bci = FindWithBlockEnd(start_bci, cfg);
    *stack_effect = -1;
    return true;
  } else if (opcode == FOR_ITER) {
    *stack_effect = -1;
  }
  *end_bci = FindLoopEnd(start_bci, cfg);
  if (list[start_bci]->op() == FOR_ITER && *end_bci == start_bci - 1) {
    // break at FOR_ITER and it is not a loop
    *end_bci = list[start_bci]->extra_jump()->bci() - 1;
  }
  return *end_bci != start_bci - 1;
}
#endif

py::object MakeCodeFromCodeGen(const GraphBuilderPtr &builder, const GraphAnalyzerPtr &analyzer, PyObject *globals) {
  TimeRecorder time_recorder(__FUNCTION__, kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogPerf));

  auto graph = builder->GetGraph();
  auto cg = CodeBreakGenerator::Creator(builder, graph->GetCodeObj());
  cg->Init(graph, *analyzer);
  cg->SetGlobals(py::cast<py::dict>(globals));
  py::object code = analyzer->NeedInterpret() ? cg->MakeDispatchCode() : cg->MakeCapturedCode();
  return code;
}

std::string PrintInstr(const std::vector<std::unique_ptr<Instr>> &list) {
  std::stringstream s;
  for (const auto &i : list) {
    s << i->ToString() << "\n";
  }
  return s.str();
}

std::string PrintNodeSet(const NodeSet &nodes) {
  std::stringstream s;
  s << "inputs: \n";
  for (auto i : nodes.inputs) {
    s << i->ToString() << "\n";
  }
  s << "outputs: \n";
  for (auto i : nodes.outputs) {
    s << i->ToString() << "\n";
  }
  s << "operations: \n";
  for (auto i : nodes.operations) {
    s << i->ToString() << "\n";
  }
  return s.str();
}

py::object MindCodeBreakGenerator::MakeCapturedCode(std::vector<std::unique_ptr<Instr>> &&load, int argc,
                                                    unsigned code_flag) const {
  // a stub to call graph
  py::object res = this->CodeBreakGenerator::MakeCapturedCode(std::move(load), argc, code_flag);
  auto jcr = GetJitCompileResults(co_);
  if (jcr->conf()->GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
    return res;
  }
  auto name = PyUnicode_AsUTF8(reinterpret_cast<PyCodeObject *>(res.ptr())->co_name);
  Compile(name, argc, 0, code_flag, res);
  return res;
}

py::object MindCodeBreakGenerator::MakeCapturedCode() const {
  auto jcr = GetJitCompileResults(co_);
  if (jcr->conf()->GetBoolConfig(GraphJitConfig::kInterpretCapturedCode)) {
    return this->CodeBreakGenerator::MakeCapturedCode();
  }
  auto name = std::to_string(jcr->IncCodeCount()) + "R." + MakeCompiledName(PyUnicode_AsUTF8(co_->co_name));
  Compile(name, co_->co_argcount, co_->co_kwonlyargcount, co_->co_flags, py::object());
  return py::object();
}

void MindCodeBreakGenerator::Compile(const std::string &co_name, int co_argcount, int co_kwonlyargcount, int co_flags,
                                     const py::object &stub) const {
  TimeRecorder compile_time("MindCodeCompile", kPIJitConfigDefault.GetBoolConfig(GraphJitConfig::kLogGuardPerf));

  // Compile graph.
  auto b = std::dynamic_pointer_cast<MindGraphBuilder>(builder_);
  MS_EXCEPTION_IF_NULL(b);
  FGBuilder()->ClearNodeAbstract();
  FGBuilder()->SetGraphName(co_name);
  auto func_graph = FGBuilder()->graph();
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Get function graph from function graph builder failed.";
  }
  std::string phase =
    py::cast<std::string>(co_->co_filename) + "_" + std::to_string(co_->co_firstlineno) + "_" + co_name;
  const auto &parameters = func_graph->parameters();
  py::tuple args(parameters.size() - func_graph->fv_param_count());
  for (size_t i = 0; i < parameters.size(); ++i) {
    auto para = parameters[i]->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      continue;
    }
    auto para_abstract = para->abstract();
    MS_EXCEPTION_IF_NULL(para_abstract);
    phase += "_" + para_abstract->ToString();
    auto input_obj = para->user_data<py::object>("pi_jit_py_obj");
    MS_EXCEPTION_IF_NULL(input_obj);
    args[i] = *input_obj;
  }
  phase += ".pi_jit";
  MindCompiler::CompileInfo compile_info{co_name, co_argcount, co_kwonlyargcount, co_flags};
  CallableGraph callable = mindspore::pijit::MindCompiler::Compile(func_graph, args, py::dict(), phase, compile_info);
  // Set NativeFunc.
  auto parent = GetJitCompileResults(co_);
  if (stub.ptr() == nullptr) {
    parent->code()->SetNativeFunc(phase, callable, nullptr);
    parent->set_stat(JitCompileResults::GRAPH_CALLABLE);
  } else {
    JitCompileResults *child = CreateJitCompileResults(stub.ptr());
    MS_EXCEPTION_IF_CHECK_FAIL(child->code() == nullptr, "must be a new stub code");
    child->set_code(child->codehub()->AddOptTarget(OptOption::CreateOptionByPoint(child)));
    child->code()->SetNativeFunc(phase, callable, nullptr);
    child->set_stat(JitCompileResults::GRAPH_CALLABLE);
    child->set_conf(parent->conf());
    child->set_tbs(parent->tbs());
  }
}

}  // namespace pijit
}  // namespace mindspore
