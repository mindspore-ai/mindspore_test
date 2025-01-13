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
#include "pipeline/jit/pi/graph_capture/graph_build.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include "pipeline/jit/pi/runtime.h"
#include "pipeline/jit/pi/graph_capture/special_func_infer.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/external.h"
#include "pipeline/jit/pi/graph_build/func_graph_builder.h"
#include "pipeline/jit/pi/graph_capture/abstract_object.h"
#include "pipeline/jit/pi/capture_context.h"
#include "include/common/debug/anf_ir_dump.h"
#include "pipeline/jit/pi/graph_compiler/utils.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "ir/cell.h"
#include "ir/func_graph_cloner.h"
#include "pybind_api/ir/primitive_py.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "pipeline/jit/pi/python_adapter/pydef.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "include/common/utils/tensor_py.h"
#include "frontend/operator/composite/composite.h"
#include "include/common/utils/hook.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace pijit {
extern TracePtr GetTrace(ValueNode *node, bool strict, bool print, int depth, int max_depth);

void LogGuardFailed(ValueNode *node, const GraphJitConfig &conf, const std::string &msg);
static bool GuardLoopSequence(Graph *graph, ValueNode *seq_node, Py_ssize_t seq_size = -1);

const char *GraphBuilder::ID___self__ = "__self__";
const char *GraphBuilder::ID___globals__ = "__globals__";
const char *GraphBuilder::ID___call__ = "__call__";
const char *GraphBuilder::ID_construct = "construct";

static const int infer_primitive_create = 1;
static const int infer_primitive_object = 2;
static const int infer_primitive_func = 4;
static int infer_func_count = 0;
static constexpr const char *kPIJitCopyFuncKey = ".<pijit.copy>.";
constexpr auto kRegisterHookKey = "backward_register_hook";

const std::unordered_map<int, bool (GraphBuilder::*)(const Instr &)> GraphBuilder::bytecode_meth_map_ = {
  {POP_TOP, &GraphBuilder::DoStackOp},
  {ROT_TWO, &GraphBuilder::DoStackOp},
  {ROT_THREE, &GraphBuilder::DoStackOp},
  {ROT_FOUR, &GraphBuilder::DoStackOp},
  {DUP_TOP, &GraphBuilder::DoStackOp},
  {DUP_TOP_TWO, &GraphBuilder::DoStackOp},
  {NOP, &GraphBuilder::DoNop},
  {EXTENDED_ARG, &GraphBuilder::DoNop},
  {GEN_START, &GraphBuilder::DoNop},
  {RETURN_VALUE, &GraphBuilder::DoReturn},
  {UNARY_POSITIVE, &GraphBuilder::DoUnary},
  {UNARY_NEGATIVE, &GraphBuilder::DoUnary},
  {UNARY_NOT, &GraphBuilder::DoUnary},
  {UNARY_INVERT, &GraphBuilder::DoUnary},
  {BINARY_MATRIX_MULTIPLY, &GraphBuilder::DoBinary},
  {BINARY_MULTIPLY, &GraphBuilder::DoBinaryMul},
  {BINARY_MODULO, &GraphBuilder::DoBinary},
  {BINARY_POWER, &GraphBuilder::DoBinary},
  {BINARY_ADD, &GraphBuilder::DoBinaryAdd},
  {BINARY_SUBTRACT, &GraphBuilder::DoBinary},
  {BINARY_FLOOR_DIVIDE, &GraphBuilder::DoBinary},
  {BINARY_TRUE_DIVIDE, &GraphBuilder::DoBinary},
  {BINARY_LSHIFT, &GraphBuilder::DoBinary},
  {BINARY_RSHIFT, &GraphBuilder::DoBinary},
  {BINARY_AND, &GraphBuilder::DoBinary},
  {BINARY_XOR, &GraphBuilder::DoBinary},
  {BINARY_OR, &GraphBuilder::DoBinary},
  {INPLACE_MATRIX_MULTIPLY, &GraphBuilder::DoBinary},
  {INPLACE_MULTIPLY, &GraphBuilder::DoBinary},
  {INPLACE_MODULO, &GraphBuilder::DoBinary},
  {INPLACE_POWER, &GraphBuilder::DoBinary},
  {INPLACE_ADD, &GraphBuilder::DoInplaceAdd},
  {INPLACE_SUBTRACT, &GraphBuilder::DoBinary},
  {INPLACE_FLOOR_DIVIDE, &GraphBuilder::DoBinary},
  {INPLACE_TRUE_DIVIDE, &GraphBuilder::DoBinary},
  {INPLACE_LSHIFT, &GraphBuilder::DoBinary},
  {INPLACE_RSHIFT, &GraphBuilder::DoBinary},
  {INPLACE_AND, &GraphBuilder::DoBinary},
  {INPLACE_XOR, &GraphBuilder::DoBinary},
  {INPLACE_OR, &GraphBuilder::DoBinary},
  {IS_OP, &GraphBuilder::DoIsOp},
  {CONTAINS_OP, &GraphBuilder::DoContainsOp},
  {BUILD_TUPLE, &GraphBuilder::DoBuildOp},
  {BUILD_LIST, &GraphBuilder::DoBuildOp},
  {BUILD_SET, &GraphBuilder::DoBuildOp},
  {BUILD_MAP, &GraphBuilder::DoBuildOp},
  {BUILD_SLICE, &GraphBuilder::DoBuildOp},
  {BUILD_CONST_KEY_MAP, &GraphBuilder::DoBuildOp},
  {BUILD_STRING, &GraphBuilder::DoBuildOp},
  {LIST_APPEND, &GraphBuilder::DoMergeOp},
  {LIST_EXTEND, &GraphBuilder::DoMergeOp},
  {DICT_MERGE, &GraphBuilder::DoMergeOp},
  {DICT_UPDATE, &GraphBuilder::DoMergeOp},
  {SET_UPDATE, &GraphBuilder::DoMergeOp},
  {SET_ADD, &GraphBuilder::DoMergeOp},
  {MAP_ADD, &GraphBuilder::DoMergeOp},
  {COMPARE_OP, &GraphBuilder::DoCompare},
  {MAKE_FUNCTION, &GraphBuilder::DoMakeFunction},
  {FORMAT_VALUE, &GraphBuilder::DoFormatValue},
  {LIST_TO_TUPLE, &GraphBuilder::DoListToTuple},
  {LOAD_CONST, &GraphBuilder::DoLoadConst},
  {IMPORT_STAR, &GraphBuilder::DoImport},
  {IMPORT_NAME, &GraphBuilder::DoImport},
  {IMPORT_FROM, &GraphBuilder::DoImport},
  {CALL_FUNCTION, &GraphBuilder::DoCall},
  {CALL_FUNCTION_KW, &GraphBuilder::DoCall},
  {CALL_FUNCTION_EX, &GraphBuilder::DoCall},
  {CALL_METHOD, &GraphBuilder::DoCall},
  {UNPACK_SEQUENCE, &GraphBuilder::DoUnpack},
  {UNPACK_EX, &GraphBuilder::DoUnpack},
  {BINARY_SUBSCR, &GraphBuilder::DoItemAccess},
  {STORE_SUBSCR, &GraphBuilder::DoItemAccess},
  {DELETE_SUBSCR, &GraphBuilder::DoItemAccess},
  {LOAD_GLOBAL, &GraphBuilder::DoGlobalAccess},
  {STORE_GLOBAL, &GraphBuilder::DoGlobalAccess},
  {DELETE_GLOBAL, &GraphBuilder::DoGlobalAccess},
  {LOAD_METHOD, &GraphBuilder::DoAttrAccess},
  {LOAD_ATTR, &GraphBuilder::DoAttrAccess},
  {STORE_ATTR, &GraphBuilder::DoAttrAccess},
  {DELETE_ATTR, &GraphBuilder::DoAttrAccess},
  {LOAD_CLOSURE, &GraphBuilder::DoCellAccess},
  {LOAD_DEREF, &GraphBuilder::DoCellAccess},
  {STORE_DEREF, &GraphBuilder::DoCellAccess},
  {DELETE_DEREF, &GraphBuilder::DoCellAccess},
  {LOAD_FAST, &GraphBuilder::DoLocalAccess},
  {STORE_FAST, &GraphBuilder::DoLocalAccess},
  {DELETE_FAST, &GraphBuilder::DoLocalAccess},
  {GET_ITER, &GraphBuilder::DoGetIter},
  {FOR_ITER, &GraphBuilder::TraceRunForIter},
  {POP_JUMP_IF_FALSE, &GraphBuilder::TraceRunControl},
  {POP_JUMP_IF_TRUE, &GraphBuilder::TraceRunControl},
  {JUMP_IF_FALSE_OR_POP, &GraphBuilder::TraceRunControl},
  {JUMP_IF_TRUE_OR_POP, &GraphBuilder::TraceRunControl},
  {JUMP_FORWARD, &GraphBuilder::TraceRunControl},
  {JUMP_ABSOLUTE, &GraphBuilder::TraceRunControl},
  {YIELD_VALUE, &GraphBuilder::DoYieldValue},
  {YIELD_FROM, &GraphBuilder::DoYieldFrom},
  {GET_YIELD_FROM_ITER, &GraphBuilder::DoGetYieldFromIter},
  {POP_BLOCK, &GraphBuilder::DoPopStack},
  {SETUP_WITH, &GraphBuilder::DoWith},
  {SETUP_FINALLY, &GraphBuilder::DoSetupFinally},
  {WITH_CLEANUP_START, &GraphBuilder::DoWithCleanUpStart},
  {WITH_CLEANUP_FINISH, &GraphBuilder::DoWithCleanUpFinish},
  {END_FINALLY, &GraphBuilder::DoEndFinally},
  {SETUP_EXCEPT, &GraphBuilder::DoSetupExc},
  {LOAD_ASSERTION_ERROR, &GraphBuilder::DoLoadAssertError},
  {POP_EXCEPT, &GraphBuilder::DoPopExc},
  {RERAISE, &GraphBuilder::DoRaise},
  {RAISE_VARARGS, &GraphBuilder::DoRaiseVarags},
  {JUMP_IF_NOT_EXC_MATCH, &GraphBuilder::DoExcMatch},
  {BEGIN_FINALLY, &GraphBuilder::DoBeginFinally},
  {POP_FINALLY, &GraphBuilder::DoPopFinally},
  {CALL_FINALLY, &GraphBuilder::DoCallFinally},
  {BUILD_TUPLE_UNPACK, &GraphBuilder::DoBuildWithUnpack},
  {BUILD_TUPLE_UNPACK_WITH_CALL, &GraphBuilder::DoBuildWithUnpack},
  {BUILD_LIST_UNPACK, &GraphBuilder::DoBuildWithUnpack},
  {BUILD_SET_UNPACK, &GraphBuilder::DoBuildWithUnpack},
  {BUILD_MAP_UNPACK, &GraphBuilder::DoBuildMapWithUnpack},
  {BUILD_MAP_UNPACK_WITH_CALL, &GraphBuilder::DoBuildMapWithUnpack},
  {LOAD_NAME, &GraphBuilder::DoLoadName},
};

bool GraphBuilder::DoOtherBytecode(const Instr &instr) {
  MS_LOG(ERROR) << "TODO: resolve for instruction " << instr.ToString();
  return false;
}

bool GraphBuilder::ReplaceAll(ValueNode *old_node, ValueNode *new_node, bool *is_referenced) {
  static const std::set<int> ref_op = {
    BUILD_TUPLE, BUILD_LIST, BUILD_SET, BUILD_MAP, BUILD_CONST_KEY_MAP,
  };

  /**
   * check reference relationship, find id_map, check them at reference graph...
   * remove this code after build a object reference graph, use function IsReferencedVariable of reference graph
   * to check it
   */
  const auto &nodes = graph_->GetTracedNodes();
  bool find = std::any_of(nodes.begin(), nodes.end(), [&old_node](ValueNode *node) {
    if (Opcode(node->GetOpcode()).MayDelete() && ref_op.find(node->GetOpcode()) == ref_op.end()) {
      return false;
    }
    const auto &args = node->getInputs();
    return std::any_of(args.begin(), args.end(), [&old_node](ValueNode *i) { return i == old_node; });
  });
  if (is_referenced != nullptr) {
    *is_referenced |= find;
  } else if (find) {
    return false;
  }

  if (parent_ != nullptr && !parent_->ReplaceAll(old_node, new_node, is_referenced)) {
    return false;
  }
  // find id_map, replace all nodes......
  const auto pred = [&old_node](ValueNode *i) { return i == old_node; };
  std::replace_if(frame_.GetLocals().begin(), frame_.GetLocals().end(), pred, new_node);
  std::replace_if(frame_.GetStacks().begin(), frame_.GetStacks().end(), pred, new_node);
  std::for_each(frame_.GetClosures().begin(), frame_.GetClosures().end(), [&old_node, &new_node](CellVarNode *i) {
    if (i->GetValue() == old_node) {
      i->SetValue(new_node);
    }
  });
  return true;
}

ValueNode *GraphBuilder::NewValueNode(AObject *o, int op, int arg, const std::vector<ValueNode *> &p,
                                      const std::string &name) {
  ValueNode *v;
  if (Opcode(op).IsCall()) {
    v = graph_->NewCallNode(op, arg, p);
    v->SetVobj(o);
  } else {
    v = graph_->NewValueNode(o, op, arg, p, name);
  }
  v->set_bci(cur_bci_);
  return v;
}

ValueNode *GraphBuilder::NewValueNode(AObject *o, const Instr &i, const std::vector<ValueNode *> &p) {
  ValueNode *v = NewValueNode(o, i.op(), i.arg(), p, i.name());
  v->SetLineNo(i.line());
  graph_->GetTracedNodes().push_back(v);
  return v;
}

Graph *GraphBuilder::NewGraph(PyCodeObject *co, PyObject *globals) {
  std::vector<Graph *> &graphs = (root_ != nullptr) ? root_->graph_pool_ : this->graph_pool_;
  if ((root_ == nullptr || root_ == this) && graph_ == nullptr) {
    JitCompileResults *jcr = GetJitCompileResults(co);
    MS_EXCEPTION_IF_CHECK_FAIL(jcr && jcr->code() != nullptr, "must be create guard code before trace start");
    graphs.push_back(new Graph(co, globals, *jcr->conf()));
    graphs.back()->SetGuard(jcr->code());
    // initialize side-effect handler, set unique data
    graphs.back()->SetSideEffect(std::make_shared<SideEffect>());
    graphs.back()->GetSideEffect()->set_data(std::make_shared<SideEffectData>());
  } else {
    graphs.push_back(new Graph(co, globals, root_->GetGraph()->Config()));
    graphs.back()->SetGuard(root_->GetGraph()->GetGuard());
    graphs.back()->SetSideEffect(root_->GetGraph()->GetSideEffect());
  }
  return graphs.back();
}

static bool CheckValueValid(AObject *obj) {
  if (obj->GetType() == AObject::kTypeTensor) {
    AbstractTensor *tensor = static_cast<AbstractTensor *>(obj);
    return tensor->IsStubTensor() || CheckTensorDataInitialized(obj->GetPyObject());
  } else {
    return true;
  }
}

int CondIsTrue(ValueNode *cond) {
  // if cond is tensor attrs, infer tensor attrs
  // if tensor is return node of cell, if tensor is return node of primitive
  // if tensor is result of math operation(+-*/...)
  AObject *cond_value = cond->GetVobj();
  int ret = -1;
  if (cond_value == nullptr || cond_value->GetPyObject().ptr() == nullptr) {
    return ret;
  }
  py::object value = cond_value->GetPyObject();
  if (CheckValueValid(cond_value)) {
    ret = PyObject_IsTrue(value.ptr());
    PyErr_Clear();
  }
  return ret;
}

static std::vector<AObject *> CollectObjects(const std::vector<ValueNode *> &nodes) {
  std::vector<AObject *> res;
  std::transform(nodes.begin(), nodes.end(), std::back_inserter(res),
                 [](const ValueNode *node) { return node->GetVobj(); });
  return res;
}

std::vector<ValueNode *> GraphBuilder::UnpackConstObject(const py::object &iterable) {
  std::vector<ValueNode *> outputs;
  std::transform(iterable.begin(), iterable.end(), std::back_inserter(outputs), [this](const py::handle &item) {
    this->DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(item)});
    return pop();
  });
  return outputs;
}

bool GraphBuilder::UnpackSequenceElements(ValueNode *node) {
  py::object seq = node->GetVobj()->GetPyObject();
  if (seq.ptr() == nullptr || !PySequence_Check(seq.ptr()) || !GuardLoopSequence(this->graph_, node)) {
    return false;
  }

  Py_ssize_t size = PySequence_Size(seq.ptr());
  for (Py_ssize_t index = 0; index < size; ++index) {
    push(node);
    DoLoadConst({LOAD_CONST, -1, py::object(py::int_(index))});
    DoItemAccess({BINARY_SUBSCR, 0});
  }
  return true;
}

bool GraphBuilder::UnpackElements(ValueNode *node) {
  int opcode = node->GetOpcode();
  if (opcode == BUILD_LIST || opcode == BUILD_TUPLE) {
    std::for_each(node->getInputs().begin(), node->getInputs().end(), [this](ValueNode *i) { this->push(i); });
  } else if (node->IsConstantValue()) {
    std::vector<ValueNode *> nodes = UnpackConstObject(node->GetVobj()->GetPyObject());
    std::for_each(nodes.begin(), nodes.end(), [this](ValueNode *i) { this->push(i); });
  } else {
    return UnpackSequenceElements(node);
  }
  return true;
}

bool GraphBuilder::UnpackDict(ValueNode *map) {
  PyObject *map_object = map->GetVobj() ? map->GetVobj()->GetPyObject().ptr() : nullptr;
  if (map->GetOpcode() == BUILD_MAP) {
    std::for_each(map->getInputs().begin(), map->getInputs().end(), [this](ValueNode *n) { this->push(n); });
  } else if (map_object != nullptr && PyDict_Check(map_object)) {
    auto keys = py::reinterpret_steal<py::object>(PyDict_Keys(map_object));
    // guard dict keys, transform to const key map......
    Py_ssize_t size = PyList_GET_SIZE(keys.ptr());
    for (Py_ssize_t i = 0; i < size; ++i) {
      Instr instr(LOAD_CONST, 0, py::reinterpret_borrow<py::object>(PyList_GET_ITEM(keys.ptr(), i)));
      this->DoLoadConst(instr);
      this->push(map);
      this->DoLoadConst(instr);
      this->DoGetItem({BINARY_SUBSCR, 0});
    }
  } else {
    return false;
  }
  return true;
}

static void GenUnpackValue(const std::function<void(int, int)> &gen_item, int cnt, int cnt_after, Py_ssize_t size) {
  if (cnt_after != -1) {
    const int end_pos = size - cnt_after;
    for (int i = size; i > end_pos; --i) {
      gen_item(i - 1, -1);
    }
    gen_item(cnt, end_pos);
  }
  for (; cnt > 0; --cnt) {
    gen_item(cnt - 1, -1);
  }
}

Py_ssize_t GetIterableSize(const ValueNode *iterable) {
  if (iterable->has_abstract_wrapper()) {
    MS_LOG(DEBUG) << "Get iterable size from abstract wrapper.";
    return iterable->abstract_wrapper()->TryToGetSize();
  }

  MS_LOG(DEBUG) << "Get iterable size from python object.";
  int op = iterable->GetOpcode();
  if (op == BUILD_LIST || op == BUILD_TUPLE || op == BUILD_MAP) {
    return iterable->getInputs().size();
  }

  AObject *seq = iterable->GetVobj();
  if (seq == nullptr) {
    return -1;
  }
  if (seq->GetType() == AObject::kTypeTuple || seq->GetType() == AObject::kTypeList) {
    return static_cast<AbstractTuple *>(seq)->size();
  }
  return PyObject_Size(seq->GetPyObject().ptr());
}

Py_ssize_t GetUnpackSize(ValueNode *iterable, int cnt, int cnt_after) {
  Py_ssize_t total_args = cnt + cnt_after + 1;
  Py_ssize_t size = GetIterableSize(iterable);
  if (size == -1 || (cnt_after == -1 && cnt != size) || total_args > size + 1) {
    PyErr_Clear();
    return -1;
  }
  return size;
}

bool GraphBuilder::DoUnpack(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  int cnt = (opcode == UNPACK_EX) ? (oparg & 0xFF) : oparg;
  int cnt_after = (opcode == UNPACK_EX) ? (oparg >> 8) : -1;
  Py_ssize_t size = GetUnpackSize(seek(0), cnt, cnt_after);
  if (size == -1) {
    return false;
  }
  ValueNode *iterable = pop();

  size_t elements_size = frame_.GetStacks().size();
  int iterable_opcode = iterable->GetOpcode();
  if (iterable_opcode == BUILD_LIST || iterable_opcode == BUILD_TUPLE) {
    std::for_each(iterable->getInputs().begin(), iterable->getInputs().end(), [this](ValueNode *i) { this->push(i); });
  } else if (iterable->IsConstantValue()) {
    std::vector<ValueNode *> nodes = UnpackConstObject(iterable->GetVobj()->GetPyObject());
    std::for_each(nodes.begin(), nodes.end(), [this](ValueNode *i) { this->push(i); });
  } else if (iterable->has_abstract_wrapper() && iterable->abstract_wrapper()->IsDict()) {
    const auto &keys = iterable->abstract_wrapper()->GetDictKeysObject();
    for (const auto &key : keys) {
      if (key.ptr() == nullptr) {
        MS_LOG(INFO) << "Failed to build key object for unpack.";
        return false;
      }
      if (!py::isinstance<py::str>(key)) {
        MS_LOG(WARNING) << "Unpack dictionary key " << py::str(key) << ", not string";
      }
      DoLoadConst({LOAD_CONST, -1, py::object(key)});
    }
  } else {
    for (Py_ssize_t index = 0; index < size; ++index) {
      push(iterable);
      DoLoadConst({LOAD_CONST, -1, py::object(py::int_(index))});
      DoItemAccess({BINARY_SUBSCR, 0});
    }
  }
  elements_size = frame_.GetStacks().size() - elements_size;
  std::vector<ValueNode *> elements(frame_.GetStacks().end() - elements_size, frame_.GetStacks().end());
  popn(elements_size);

  auto gen_item = [this, &elements](int i, int j) {
    if (j == -1) {
      this->push(elements[i]);
      return;
    }
    MS_EXCEPTION_IF_CHECK_FAIL(j >= i, "check UNPACK_EX oparg");
    auto in_iter = elements.begin();
    std::for_each(in_iter + i, in_iter + j, [this](ValueNode *i) { this->push(i); });
    DoBuildOp({BUILD_LIST, j - i});
  };
  GenUnpackValue(gen_item, cnt, cnt_after, size);
  return true;
}

bool GraphBuilder::DoBuildWithUnpack(const Instr &instr) {
  static const std::map<int, int> build_op_map = {{BUILD_LIST_UNPACK, BUILD_LIST},
                                                  {BUILD_SET_UNPACK, BUILD_SET},
                                                  {BUILD_TUPLE_UNPACK, BUILD_TUPLE},
                                                  {BUILD_TUPLE_UNPACK_WITH_CALL, BUILD_TUPLE}};
  int opcode = instr.op();
  MS_EXCEPTION_IF_CHECK_FAIL(build_op_map.find(opcode) != build_op_map.end(), "Invalid opcode for DoBuildWithUnpack.");
  const std::vector<ValueNode *> iterables(frame_.GetStacks().end() - instr.arg(), frame_.GetStacks().end());
  popn(instr.arg());
  int elements_cnt = 0;
  for (auto iter = iterables.rbegin(); iter != iterables.rend(); iter++) {
    int size = GetIterableSize(*iter);
    if (size < 0) {
      MS_LOG(ERROR) << "Invalid unpack object. error : " << py::error_already_set().what();
      return false;
    }
    // maybe there is empty tuple/list in BUILD_*_UNPACK*
    if (size == 0) {
      continue;
    }
    push(*iter);
    DoUnpack({UNPACK_SEQUENCE, size});
    elements_cnt += size;
  }
  const std::vector<ValueNode *> elements(frame_.GetStacks().end() - elements_cnt, frame_.GetStacks().end());
  popn(elements_cnt);
  std::for_each(elements.rbegin(), elements.rend(), [this](auto node) { push(node); });
  DoBuildOp({build_op_map.at(opcode), elements_cnt});
  return true;
}

bool GraphBuilder::DoBuildMapWithUnpack(const Instr &instr) {
  const std::vector<ValueNode *> iterables(frame_.GetStacks().end() - instr.arg(), frame_.GetStacks().end());
  popn(instr.arg());
  std::vector<ValueNode *> keys_values;
  std::for_each(iterables.begin(), iterables.end(), [this, &keys_values](auto node) {
    if (node->GetOpcode() == BUILD_MAP) {
      std::for_each(node->getInputs().begin(), node->getInputs().end(),
                    [&keys_values](ValueNode *input) { keys_values.push_back(input); });
    } else {
      PyObject *key = nullptr;
      PyObject *value = nullptr;
      Py_ssize_t pos = 0;
      while (PyDict_Next(node->GetVobj()->GetPyObject().ptr(), &pos, &key, &value)) {
        DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(key)});
        keys_values.push_back(pop());
        DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(value)});
        keys_values.push_back(pop());
      }
    }
  });
  std::for_each(keys_values.begin(), keys_values.end(), [this](auto node) { push(node); });
  DoBuildOp({BUILD_MAP, SizeToInt(keys_values.size() / 2)});
  return true;
}

bool GraphBuilder::DoCall(const Instr &instr) {
  Opcode opcode(instr.op());
  int oparg = instr.arg();
  int tmp_arg = oparg;
  std::vector<ValueNode *> params;
  if (opcode == CALL_FUNCTION_EX) {
    tmp_arg = (tmp_arg & 0x01) + 1;
  } else if (opcode == CALL_FUNCTION_KW) {
    tmp_arg += 1;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(opcode.IsCall(), "must be call");
  params = {frame_.GetStacks().end() - tmp_arg - 1, frame_.GetStacks().end()};
  opcode = (opcode == CALL_METHOD) ? CALL_FUNCTION : opcode;
  popn(tmp_arg + 1);
  if (IsPartialFunc(params[0]->GetVobj()->GetPyObject())) {
    push(params[0]);
    DoAttrAccess({LOAD_ATTR, 0, "func"});
    push(params[0]);
    DoAttrAccess({LOAD_ATTR, 0, "args"});
    ValueNode *args = pop();
    size_t args_size = PyTuple_GET_SIZE(args->GetVobj()->GetPyObject().ptr());
    UnpackElements(args);
    for (size_t i = 1; i < params.size(); ++i) {
      push(params[i]);
    }
    DoBuildOp({BUILD_TUPLE, SizeToInt(args_size + params.size()) - 1});
    auto kwargs = PyObject_GetAttrString(params[0]->GetVobj()->GetPyObject().ptr(), "keywords");
    if (kwargs != nullptr && kwargs != Py_None && PyDict_Size(kwargs) > 0) {
      push(params[0]);
      DoAttrAccess({LOAD_ATTR, 0, "keywords"});
      DoCall({CALL_FUNCTION_EX, 1});
    } else {
      DoCall({CALL_FUNCTION_EX, 0});
    }
    Py_XDECREF(kwargs);
    return true;
  } else {
    push(NewValueNode(nullptr, opcode, oparg, params));
  }

  CallNode *call_node = static_cast<CallNode *>(seek(0));
  call_node->SetVobj(AObject::MakeAObject(AObject::kTypeAnyValue));
  call_node->SetLineNo(instr.line());
  call_node->set_bci(instr.bci());
  this->graph_->GetTracedNodes().push_back(call_node);

  StopTraceReason r = HandleCall(0);
  if (r != StopTraceReason::kNonStopTrace) {
    graph_->StopTraceAt(cur_bci_, r);
    return false;
  }
  return true;
}

bool GraphBuilder::DoNop(const Instr &instr) { return true; }
bool GraphBuilder::NotImplementBytecode(const Instr &instr) { return false; }

bool GraphBuilder::DoGetYieldFromIter(const Instr &instr) {
  auto iterable = seek(0)->GetVobj()->GetPyObject().ptr();
  if (iterable != nullptr && !PyIter_Check(iterable)) {
    DoGetIter(instr);
  } else {
    MS_LOG(WARNING) << "not support yield iterator yet!";
    return false;
  }
  return true;
}

bool GraphBuilder::DoYieldFrom(const Instr &instr) {
  auto iter_node = dynamic_cast<IterNode *>(seek(1));
  MS_EXCEPTION_IF_NULL(iter_node);
  size_t size = frame_.GetStacks().size();
  if (!UnpackElements(iter_node->iterable())) {
    return false;
  }
  size = frame_.GetStacks().size() - size;
  std::vector<ValueNode *> elements(frame_.GetStacks().end() - size, frame_.GetStacks().end());
  popn(size);
  for (auto n : elements) {
    push(n);
    DoYieldValue(instr);
    pop();
  }
  pop();  // None
  return true;
}

bool GraphBuilder::DoYieldValue(const Instr &instr) {
  ValueNode *result = graph_->GetGeneratorResult();
  if (result == nullptr) {
    result = NewValueNode(nullptr, BUILD_TUPLE, 0);
    graph_->SetGeneratorResult(result);
  }
  ValueNode *value = seek(0);
  result->AddInput(value);
  const int YIELD_COUNT_THRESHOLD = 1000;
  if (result->getInputs().size() % YIELD_COUNT_THRESHOLD == 0) {
    MS_LOG(WARNING) << "yield too many value: " << result->getInputs().size();
  }
  return true;
}

bool GraphBuilder::DoReturn(const Instr &instr) {
  graph_->SetRetVal(pop());
  if (graph_->GetGeneratorResult() == nullptr) {
    return true;
  }
  const auto &inputs = graph_->GetGeneratorResult()->getInputs();
  std::for_each(inputs.begin(), inputs.end(), [this](ValueNode *i) { this->push(i); });
  DoBuildOp({BUILD_TUPLE, SizeToInt(inputs.size())});
  ValueNode *new_node = pop();
  graph_->SetGeneratorResult(new_node);
  graph_->SetRetVal(new_node);
  return true;
}

ValueNode *GraphBuilder::MakePrimCastNode(ValueNode *node, const py::handle &dst_dtype) {
  py::object prim_cast = Utils::GetModuleAttr("mindspore.ops.functional", "_cast", false, true);
  DoLoadConst({LOAD_CONST, -1, prim_cast});
  push(node);
  DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(dst_dtype)});
  DoCall({CALL_FUNCTION, 2});
  return pop();
}

bool GraphBuilder::DoMixedPrecisionLocalAccess(const Instr &instr, ValueNode *node) {
  auto param_node = static_cast<ParamNode *>(node);
  auto dst_dtype = param_node->GetMixedPrecisionType();
  ValueNode *call_node = MakePrimCastNode(node, dst_dtype);
  push(call_node);
  auto *call = static_cast<CallNode *>(call_node);
  call->SetVobj(AObject::MakeAObject(AObject::kTypeAnyValue));
  call->SetLineNo(instr.line());
  call->set_bci(instr.bci());
  StopTraceReason r = HandleCall(0);
  if (r != StopTraceReason::kNonStopTrace) {
    graph_->StopTraceAt(cur_bci_, r);
    return false;
  }
  this->graph_->GetTracedNodes().push_back(call_node);
  return true;
}

bool GraphBuilder::DoLocalAccess(const Instr &instr) {
  if (instr.op() == LOAD_FAST) {
    auto local = getLocal(instr.arg());
    if (local->GetType() == AbstractNode::Param && reinterpret_cast<ParamNode *>(local)->IsMixedPrecisionType()) {
      // TODO(lvxudong): fix multi cast
      DoMixedPrecisionLocalAccess(instr, local);
    } else {
      push(local);
    }
  } else if (instr.op() == STORE_FAST) {
    setLocal(instr.arg(), pop());
  } else if (instr.op() == DELETE_FAST) {
    setLocal(instr.arg(), &ValueNode::kUnboundLocal);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "parser got an error instruction " << instr.ToString();
  }
  return true;
}

namespace {
bool IsFreeVar(PyCodeObject *code, int closure_index) {
#if IS_PYTHON_3_11_PLUS
  MS_LOG(INFO) << "Python 3.11 is not supported";
  throw py::error_already_set();
#endif
  return closure_index >= PyCodeWrapper(code).CellVarsSize();
}
}  // namespace

bool GraphBuilder::DoCellAccess(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  ValueNode *node;
  ValueNode *value;
  PyObject *cell = frame_.Closure(oparg)->GetVobj()->GetPyObject().ptr();
  MS_EXCEPTION_IF_CHECK_FAIL(cell && PyCell_Check(cell), "must be a cell object");
  if (opcode == LOAD_CLOSURE) {
    push(frame_.Closure(oparg));
  } else if (opcode == LOAD_DEREF) {
    MS_EXCEPTION_IF_NULL(frame_.Closure(oparg)->GetValue());
    push(frame_.Closure(oparg)->GetValue());
  } else if (opcode == STORE_DEREF) {
    if (IsFreeVar(graph_->GetCodeObj(), oparg) && !IsTopGraph()) {
      // The side-effect of free-variable STORE_DEREF in subgraph will be supported later.
      graph_->StopTraceAt(cur_bci_, kStopTraceByteCode_Unsupported);
      return false;
    }
    value = pop();
    bool is_same = value->GetOpcode() == LOAD_DEREF && frame_.Closure(oparg) == frame_.Closure(value->GetOparg());
    if (!is_same) {
      node = NewValueNode(nullptr, instr, {value});
      frame_.Closure(oparg)->SetValue(value);
      frame_.Closure(oparg)->AddCellOper(node);
    }
  } else if (opcode == DELETE_DEREF) {
    if (IsFreeVar(graph_->GetCodeObj(), oparg) && !IsTopGraph()) {
      graph_->StopTraceAt(cur_bci_, kStopTraceByteCode_Unsupported);
      return false;
    }
    node = NewValueNode(nullptr, instr, {});
    frame_.Closure(oparg)->SetValue(&ValueNode::kUnboundLocal);
    frame_.Closure(oparg)->AddCellOper(node);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "parser got an error instruction " << instr.ToString();
  }
  return true;
}

// Parse byteCode -- SETUP_WITH
bool GraphBuilder::DoWith(const Instr &instr) {
  if (graph_->Config().GetBoolConfig(GraphJitConfig::kSkipException) || PyErr_Occurred()) {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceSkip_Exception);
    return false;
  }
  auto node = pop();
  push(node);
  DoAttrAccess({LOAD_ATTR, 0, "__exit__"});

  push(node);
  DoAttrAccess({LOAD_ATTR, 0, "__enter__"});

  if (!DoCall({CALL_FUNCTION, 0})) {
    MS_LOG(ERROR) << "function '__enter__' runs failed here, it should be successful!";
    return false;
  }
  PushStack(TryBlock{SETUP_WITH, instr.extra_jump()->bci(), instr.name(), instr.bci(), false});
  cur_bci_++;
  return true;
}

bool GraphBuilder::DoPopFinally(const mindspore::pijit::Instr &instr) {
  auto preserveTOS = instr.arg();
  if (preserveTOS) {
    auto res = pop();
    pop();
    push(res);
  }
  return true;
}

bool GraphBuilder::DoRaiseVarags(const mindspore::pijit::Instr &instr) {
  int oparg = instr.arg();
  if (oparg != 1) {
    return false;
  }
  auto exc = pop();
  pushExc(exc);
  return DoRaise(instr);
}

bool GraphBuilder::DoRaise(const mindspore::pijit::Instr &instr) {
  auto exc = peekExc(0);

  if (StackSize() < 1) {
    return false;
  }

  auto currentBlock = PopStack();
  while (currentBlock.name == "EXCEPT_HANDLER") {
    popn(3);
    if (StackSize() < 1) {
      return false;
    }
    currentBlock = PopStack();
  }

  if (currentBlock.type != SETUP_FINALLY && currentBlock.type != SETUP_EXCEPT) {
    PushStack(currentBlock);
    return false;
  }

  // Push a dummy block stack entry of EXCEPT_HANDLER
  // https://github.com/python/cpython/blob/3.10/Python/ceval.c#L1456
  PushStack(TryBlock{0, 0, "EXCEPT_HANDLER", 0, false});
  DoLoadConst({LOAD_CONST, -1, py::none()});
  auto noneNode = pop();
  if (excStackSize() >= 2) {
    auto old_exc = peekExc(1);
    push(noneNode);
    push(old_exc);
    push(old_exc);
  } else {
    push(noneNode);
    push(noneNode);
    push(noneNode);
  }

  push(noneNode);  // traceback
  push(exc);       // value
  push(exc);       // type

  cur_bci_ = currentBlock.bci - 1;
  return true;
}

bool GraphBuilder::DoSetupExc(const mindspore::pijit::Instr &instr) {
  if (graph_->Config().GetBoolConfig(GraphJitConfig::kSkipException)) {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceSkip_Exception);
    return false;
  }
  PushStack(TryBlock{SETUP_EXCEPT, instr.extra_jump()->bci(), instr.name(), instr.bci(), false});
  cur_bci_++;
  return true;
}

bool GraphBuilder::DoPopExc(const mindspore::pijit::Instr &instr) {
  if (StackSize() < 1) {
    MS_LOG(ERROR) << "try block stack size is 0.";
    return false;
  }
  if (PeekStack(0).name != "EXCEPT_HANDLER") {
    MS_LOG(ERROR) << "Top of the try block stack is not EXCEPT_HANDLER.";
    return false;
  }
  PopStack();
  popn(3);
  popExc();
  return true;
}

bool GraphBuilder::DoExcMatch(const mindspore::pijit::Instr &instr) {
  auto expectedExcType = pop();
  auto gotExcInstance = pop();

  auto expectedErrs = expectedExcType->GetVobj()->GetPyObject().ptr();
  auto gotErr = gotExcInstance->GetVobj()->GetPyObject().ptr();
  if (!PyTuple_Check(expectedErrs) && !PyExceptionClass_Check(expectedErrs)) {
    MS_LOG(ERROR) << "unsupported except types: " << Py_TYPE(expectedErrs);
    return false;
  }

  auto res = PyErr_GivenExceptionMatches(gotErr, expectedErrs);
  if (res == 0) {
    cur_bci_ = instr.extra_jump()->bci();  // 没有匹配上，跳到目标bci
  } else {
    // 匹配到对应类型，fallthrough
    cur_bci_++;
  }
  return true;
}

bool GraphBuilder::DoPopStack(const mindspore::pijit::Instr &instr) {
  PopStack();
  return true;
}

bool GraphBuilder::DoSetupFinally(const mindspore::pijit::Instr &instr) {
  /*
        ByteCode like this in python3.9
        0 SETUP_FINALLY    xxx
        1 SETUP_FINALLY    xxx
        the first SETUP_FINALLY points to finally block, the second points to exception block
      */
  if (graph_->Config().GetBoolConfig(GraphJitConfig::kSkipException)) {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceSkip_Exception);
    return false;
  }
  if (StackSize() == 0 || GetTryBlockStacks().back().type != SETUP_FINALLY) {
    PushStack(TryBlock{SETUP_FINALLY, instr.extra_jump()->bci(), instr.name(), instr.bci(), true});
  } else {
    PushStack(TryBlock{SETUP_FINALLY, instr.extra_jump()->bci(), instr.name(), instr.bci(), false});
  }
  cur_bci_++;
  return true;
}

bool GraphBuilder::DoWithCleanUpFinish(const mindspore::pijit::Instr &instr) {
  auto exc = pop();
  (void)pop();
  push(exc);
  return true;
}

bool GraphBuilder::DoWithCleanUpStart(const mindspore::pijit::Instr &instr) {
  /* python3.7 only */
  ValueNode *exc = seek(0);
  ValueNode *exit_func = seek(1);
  if (exc->GetVobj()->GetType() != AObject::kTypeNone) {
    return false;
  }
  if (exit_func->GetName() != "__exit__") {
    MS_LOG(ERROR) << "it should call function '__exit__' here!";
    return false;
  }
  // run exit func
  push(exc);
  push(exc);
  if (!DoCall({CALL_FUNCTION, 3})) {
    MS_LOG(ERROR) << "function '__exit__' runs failed here, it should be successful!";
    return false;
  }
  push(exc);
  return true;
}

bool GraphBuilder::DoBeginFinally(const mindspore::pijit::Instr &instr) {
  DoLoadConst({LOAD_CONST, -1, py::none()});
  return true;
}

bool GraphBuilder::DoEndFinally(const mindspore::pijit::Instr &instr) {
  auto tos = pop();
  if (PyLong_Check(tos->GetVobj()->GetPyObject().ptr())) {
    cur_bci_ = tos->bci();
  }

  if (PyExceptionInstance_Check(tos->GetVobj()->GetPyObject().ptr()) ||
      PyExceptionClass_Check(tos->GetVobj()->GetPyObject().ptr())) {
    push(tos);
    return DoRaise(instr);
  }
  return true;
}

bool GraphBuilder::DoLoadAssertError(const mindspore::pijit::Instr &instr) {
  DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(PyExc_AssertionError)});
  return true;
}

bool GraphBuilder::DoCallFinally(const mindspore::pijit::Instr &instr) {
  DoLoadConst({LOAD_CONST, -1, py::int_(0)});
  cur_bci_ += instr.arg() / 2;
  return true;
}

TryBlock &GraphBuilder::PeekStack(int p) {
  MS_ASSERT(tryBlockStacks_.size() > p);
  return tryBlockStacks_[tryBlockStacks_.size() - p - 1];
}

TryBlock GraphBuilder::PopStack() {
  MS_ASSERT(tryBlockStacks_.size() > 0);
  auto tb = tryBlockStacks_[tryBlockStacks_.size() - 1];
  tryBlockStacks_.pop_back();
  return tb;
}

void GraphBuilder::DoLoadGlobal(const Instr &instr) {
  py::object globals = graph_->GetGlobals();
  py::str key = instr.name();

  PyObject *obj = PyObject_GetItem(globals.ptr(), key.ptr());
  if (obj == nullptr) {
    PyErr_Clear();
    obj = PyObject_GetItem(PyEval_GetBuiltins(), key.ptr());
    if (obj == nullptr) {
      PyErr_Clear();
    }
  }
  py::object pyobj = py::reinterpret_steal<py::object>(obj);
  auto n = NewValueNode(AObject::Convert(pyobj), instr, {});
  n->SetName(instr.name());
  push(n);
}

bool GraphBuilder::DoGlobalAccess(const Instr &instr) {
  int opcode = instr.op();
  if (opcode == LOAD_GLOBAL) {
    auto cache_result = graph_->GetSideEffect()->LoadGlobal(graph_->GetModuleName(), instr.name());
    if (cache_result.is_deleted_value_) {
      return false;  // name error
    } else if (cache_result.cache_value_ != nullptr) {
      push(cache_result.cache_value_);
    } else {
      DoLoadGlobal(instr);
    }
  } else if (opcode == STORE_GLOBAL) {
    auto global_node = pop();
    auto node = NewValueNode(nullptr, instr, {global_node});
    graph_->GetSideEffect()->Record(node);
  } else if (opcode == DELETE_GLOBAL) {
    auto node = NewValueNode(nullptr, instr, {});
    graph_->GetSideEffect()->Record(node);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "parser got an error instruction " << instr.ToString();
  }
  return true;
}

bool GraphBuilder::HandleSuper(const Instr &instr, AObject *super) {
  if (super == nullptr || super->GetTypeObject() != &PySuper_Type) {
    return false;
  }
  ValueNode *self_super = SearchSelfPyObject(graph_->GetCodeObj()).second;
  if (self_super == nullptr) {
    return false;
  }
  py::object method = super->GetPyObject().attr(instr.name().c_str());
  if (!PyMethod_Check(method.ptr())) {
    return false;
  }

  // method type object
  auto mtype_obj = reinterpret_cast<PyObject *>(&PyMethod_Type);
  DoLoadConst({LOAD_CONST, -1, py::cast<py::object>(mtype_obj)});

  // function object
  PyObject *m = PyMethod_GET_FUNCTION(method.ptr());
  DoLoadConst({LOAD_CONST, -1, py::cast<py::object>(m)});

  push(self_super);

  // call method type
  return DoCall({CALL_FUNCTION, 2});
}

PyObject *SetLocalPyObject(ValueNode *node) {
  if (node == nullptr || node->GetVobj() == nullptr) {
    return NULL;
  } else {
    return node->GetVobj()->GetPyObject().ptr();
  }
}

std::pair<PyObject *, ValueNode *> GraphBuilder::SearchSelfPyObject(PyCodeObject *co) {
#if IS_PYTHON_3_11_PLUS
  MS_LOG(ERROR) << "not implement in python3.11";
  return {nullptr, nullptr};
#else
  if (co->co_argcount < 1) {
    return {nullptr, nullptr};
  }
  std::pair<PyObject *, ValueNode *> obj_value;
  ValueNode *value = frame_.Local(0);
  // get self or son class, eg.super(Son, self)
  PyObject *obj = SetLocalPyObject(frame_.Local(0));
  Py_ssize_t i, n;
  if (obj == NULL && co->co_cell2arg) {
    // the first argument might be a cell
    n = PyTuple_GET_SIZE(co->co_cellvars);
    for (i = 0; i < n; i++) {
      if (co->co_cell2arg[i] == 0) {
        value = frame_.Closure(i)->GetValue();
        obj = SetLocalPyObject(frame_.Closure(i));
        break;
      }
    }
  }
  obj_value = std::make_pair(obj, value);
  return obj_value;
#endif
}

ValueNode *GraphBuilder::HandleGetattr(ValueNode *target_node, const Instr &instr) {
  return NewValueNode(target_node->get_attr(instr.name()), instr, {target_node});
}

ValueNode *GraphBuilder::DoMixedPrecisionAttrAccess(const Instr &instr, ValueNode *node, ValueNode *attr) {
  if (node->GetVobj() == nullptr || node->GetVobj()->GetPyObject().ptr() == nullptr ||
      node->GetVobj()->GetType() != AbstractObjectBase::kTypeCell) {
    return nullptr;
  }
  auto cell = py::cast<CellPtr>(node->GetVobj()->GetPyObject());
  auto mixed_type = cell->GetMixedPrecisionType();
  if (mixed_type == kNotSet) {
    return nullptr;
  }
  if (attr->GetVobj() == nullptr || attr->GetVobj()->GetPyObject().ptr() == nullptr) {
    return nullptr;
  }
  if (attr->GetVobj()->GetType() == AObject::kTypeTensor && !attr->GetVobj()->GetPyObject().attr("dtype").is_none()) {
    auto src_dtype = attr->GetVobj()->GetPyObject().attr("dtype");
    bool is_cast = false;
    if (py::isinstance<Float>(src_dtype)) {
      auto float_nbits = py::cast<Float>(src_dtype).nbits();
      if (float_nbits == 64 || (float_nbits == 32 && mixed_type != kFP32) ||
          (float_nbits == 16 && mixed_type != kFP16)) {
        is_cast = true;
      }
    }
    if (py::isinstance<BFloat>(src_dtype) && mixed_type != kBF16) {
      is_cast = true;
    }
    if (is_cast) {
      auto dst_dtype = Utils::MixedPrecisionTypeToDType(mixed_type);
      ValueNode *call_node = MakePrimCastNode(attr, dst_dtype);
      CallNode *call = static_cast<CallNode *>(call_node);
      call->SetVobj(AObject::MakeAObject(AObject::kTypeAnyValue));
      call->SetLineNo(instr.line());
      call->set_bci(instr.bci());
      push(call_node);
      StopTraceReason r = HandleCall(0);
      if (r != StopTraceReason::kNonStopTrace) {
        graph_->StopTraceAt(cur_bci_, r);
        return nullptr;
      }
      this->graph_->GetTracedNodes().push_back(call_node);
      return pop();
    }
  }
  return nullptr;
}

bool GraphBuilder::DoLoadName(const mindspore::pijit::Instr &instr) {
  if (instr.name() == "__name__") {
    this->graph_->FoundInnerClass();
  }

  return false;
}

bool GraphBuilder::DoAttrAccess(const Instr &instr) {
  int opcode = instr.op();
  if (opcode == LOAD_METHOD || opcode == LOAD_ATTR) {
    auto o = pop();
    if (HandleSuper(instr, o->GetVobj())) {
      return true;
    }
    auto cache_result = graph_->GetSideEffect()->LoadAttr(o, instr.name());
    if (cache_result.is_deleted_value_) {  // attribute error
      return false;
    } else if (cache_result.cache_value_ != nullptr) {
      push(cache_result.cache_value_);
    } else {
      ValueNode *node = HandleGetattr(o, instr);
      if (node == nullptr) {
        graph_->StopTraceAt(cur_bci_, StopTraceReason::kTrace_Fail);
        return false;
      }
      push(node);
      auto attr = DoMixedPrecisionAttrAccess(instr, o, seek(0));
      if (attr) {
        seek(0) = attr;
      }
    }
  } else if (opcode == STORE_ATTR) {
    auto o = pop();
    auto v = pop();
    auto node = NewValueNode(nullptr, instr, {v, o});
    graph_->GetSideEffect()->Record(node);
  } else if (opcode == DELETE_ATTR) {
    auto o = pop();
    auto node = NewValueNode(nullptr, instr, {o});
    graph_->GetSideEffect()->Record(node);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "parser got an error instruction " << instr.ToString();
  }
  return true;
}

// for unpack call optimize
static ValueNode *TupleDictGetItem(ValueNode *container, ValueNode *index_node) {
  if (!index_node->IsConstantValue()) {
    return nullptr;
  }
  PyObject *index_object = index_node->GetVobj()->GetPyObject().ptr();
  int opcode = container->GetOpcode();
  if ((opcode == BUILD_TUPLE || opcode == BUILD_LIST) && PyLong_Check(index_object)) {
    Py_ssize_t index = PyLong_AsSsize_t(index_object);
    Py_ssize_t size = container->getInputs().size();
    if (index < -size || index >= size) {
      return nullptr;
    }
    index = index < 0 ? (size + index) : index;
    return container->input(index);
  }
  if (container->GetOpcode() == BUILD_MAP && PyUnicode_Check(index_object)) {
    std::string k = PyUnicode_AsUTF8(index_object);
    size_t element_count = container->GetOparg() << 1;
    MS_EXCEPTION_IF_CHECK_FAIL(element_count == container->getInputs().size(), "check BUILD_MAP oparg");
    for (int i = 0; i < container->GetOparg(); ++i) {
      AObject *tmp = container->input(i * 2)->GetVobj();
      PyObject *str = tmp ? tmp->GetPyObject().ptr() : nullptr;
      if (str == nullptr || !PyUnicode_Check(str) || k != PyUnicode_AsUTF8(str)) {
        continue;
      }
      return container->input((i << 1) + 1);
    }
  }
  return nullptr;
}

bool GraphBuilder::DoGetItem(const Instr &instr) {
  constexpr const char *kNameGetItem = "__getitem__";
  auto r = pop();
  auto l = pop();
  ValueNode *v = TupleDictGetItem(l, r);
  if (v != nullptr) {
    push(v);
    return true;
  }

  AObject *container = l->GetVobj();
  PyObject *op = container ? container->GetPyObject().ptr() : nullptr;
  AObject *meth = nullptr;

  bool call_getitem = op == nullptr || container->GetType() != AObject::kTypeAnyValue;
  if (!call_getitem) {
    call_getitem = PyDict_Check(op) || PyTuple_Check(op) || PyList_Check(op);
  }
  if (!call_getitem) {
    meth = container->GetAttr(kNameGetItem);
    PyObject *m = meth ? meth->GetPyObject().ptr() : nullptr;
    call_getitem = m == nullptr || !PyMethod_Check(m) || !PyFunction_Check(PyMethod_GET_FUNCTION(m));
  }
  if (call_getitem) {
    /**
     * check safe callable of __getitem__ if user defined.
     */
    AObject *vo = l->binary_subscr(r);
    v = NewValueNode(vo, instr, {l, r});
    push(v);
    return true;
  }

  push(l);
  DoAttrAccess({LOAD_ATTR, 0, kNameGetItem});
  push(r);
  return DoCall({CALL_FUNCTION, 1});
}

ValueNode *GraphBuilder::TransformDictSetItem(ValueNode *map, ValueNode *key, ValueNode *value, bool ignore_key_error) {
  PyObject *index_object = key->GetVobj()->GetPyObject().ptr();
  if (index_object == nullptr || !key->IsConstantValue()) {
    return nullptr;  // only supported constant key
  }
  constexpr const int kNumberTwo = 2;
  PyObject *map_object = map->GetVobj()->GetPyObject().ptr();
  std::vector<ValueNode *> elements;
  if (map->GetOpcode() == BUILD_MAP) {
    elements = map->getInputs();
  } else if (map_object != nullptr) {
    auto keys = py::reinterpret_steal<py::object>(PyDict_Keys(map_object));
    // guard dict keys, transform to const key map......
    Py_ssize_t size = PyList_GET_SIZE(keys.ptr());
    for (Py_ssize_t i = 0; i < size; ++i) {
      Instr instr(LOAD_CONST, 0, py::reinterpret_borrow<py::object>(PyList_GET_ITEM(keys.ptr(), i)));
      this->DoLoadConst(instr);
      this->push(map);
      this->DoLoadConst(instr);
      this->DoGetItem({BINARY_SUBSCR, 0});
    }
    elements = {frame_.GetStacks().end() - size * kNumberTwo, frame_.GetStacks().end()};
    popn(size * kNumberTwo);
  } else {
    return nullptr;
  }

  // set(delete) element
  if (value != nullptr) {
    elements.push_back(key);
    elements.push_back(value);
  } else {
    int index_of_key = -1;
    for (int i = elements.size() - kNumberTwo; i >= 0 && index_of_key == -1; i -= kNumberTwo) {
      bool find = elements[i]->GetVobj()->GetPyObject().equal(py::handle(index_object));
      index_of_key = find ? i : -1;
    }
    if (index_of_key != -1) {
      elements.erase(elements.begin() + index_of_key, elements.begin() + index_of_key + kNumberTwo);
    } else if (!ignore_key_error) {
      return nullptr;  // maybe key error
    }
  }

  // rebuild map
  int size = elements.size() / kNumberTwo;
  std::for_each(elements.begin(), elements.end(), [this](ValueNode *i) { this->push(i); });
  DoBuildOp({BUILD_MAP, size});
  return pop();
}

std::vector<Py_ssize_t> ListIndexCompute(PyObject *index_object, Py_ssize_t size) {
  if (PyIndex_Check(index_object)) {
    Py_ssize_t index = PyNumber_AsSsize_t(index_object, PyExc_IndexError);
    if (!PyErr_Occurred() && index > -size && index < size) {
      index = index < 0 ? (index + size) : index;
      return {index, index + 1, 1, 1};
    }
  } else if (PySlice_Check(index_object)) {
    Py_ssize_t start;
    Py_ssize_t stop;
    Py_ssize_t step;
    Py_ssize_t slice_length;
    constexpr Py_ssize_t zero = 0;
    if (0 == PySlice_GetIndicesEx(index_object, size, &start, &stop, &step, &slice_length)) {
      slice_length = (start < 0 || stop < 0 || slice_length < 0) ? 0 : slice_length;
      return {std::max(start, zero), std::max(stop, zero), step, slice_length};
    }
  }
  if (!PyErr_Occurred()) {
    return {};
  }
  throw py::error_already_set();
}

template <typename T>
static bool SetSlice(std::vector<T> *elements, const std::vector<Py_ssize_t> &computed_slice,
                     std::vector<T> *new_elements = nullptr) {
  constexpr int start = 0;
  constexpr int stop = 1;
  constexpr int step = 2;
  constexpr int slice_length = 3;

  const auto &slice = computed_slice;
  if (slice[step] == 1) {
    elements->erase(elements->begin() + slice[start], elements->begin() + slice[stop]);
    if (new_elements != nullptr) {
      elements->insert(elements->begin() + slice[start], new_elements->begin(), new_elements->end());
    }
    return true;
  }
  if (new_elements != nullptr && new_elements->size() != static_cast<size_t>(slice[slice_length])) {
    return false;
  }
  for (Py_ssize_t cur = slice[start], i = 0; i < slice[slice_length]; cur += slice[step], ++i) {
    (*elements)[cur] = new_elements == nullptr ? nullptr : (*new_elements)[i];
  }
  if (new_elements == nullptr) {
    elements->erase(std::remove(elements->begin(), elements->end(), nullptr), elements->end());
  }
  return true;
}

ValueNode *GraphBuilder::TransformListSetItem(ValueNode *map, ValueNode *key, ValueNode *value) {
  PyObject *index_object = key->GetVobj()->GetPyObject().ptr();
  if (index_object == nullptr || !key->IsConstantValue()) {
    return nullptr;  // only supported constant key
  }
  PyObject *map_object = map->GetVobj()->GetPyObject().ptr();
  std::vector<ValueNode *> elements;
  if (map->GetOpcode() == BUILD_LIST) {
    elements = map->getInputs();
  } else if (UnpackElements(map)) {
    Py_ssize_t size = PyList_GET_SIZE(map_object);
    elements = {frame().GetStacks().end() - size, frame().GetStacks().end()};
    popn(size);
  } else {
    return nullptr;
  }

  // compute slice
  auto slice = ListIndexCompute(index_object, elements.size());
  if (slice.empty()) {
    return nullptr;
  }
  // set(delete) elements
  size_t stack_size = frame_.GetStacks().size();
  if (!PySlice_Check(index_object)) {
    auto iter = elements.begin() + slice[0];
    (void)(value == nullptr ? elements.erase(iter) : (*iter = value, iter));
  } else if (value == nullptr && SetSlice(&elements, slice)) {
    // delete success
  } else if (value != nullptr && UnpackElements(value)) {
    // unpack success
    stack_size = frame_.GetStacks().size() - stack_size;
    std::vector<ValueNode *> new_elements = {frame_.GetStacks().end() - stack_size, frame_.GetStacks().end()};
    popn(stack_size);
    if (!SetSlice(&elements, slice, &new_elements)) {
      return nullptr;
    }
    // set succuss
  } else {
    return nullptr;
  }

  std::for_each(elements.begin(), elements.end(), [this](ValueNode *i) { this->push(i); });
  DoBuildOp({BUILD_LIST, SizeToInt(elements.size())});
  return pop();
}

ValueNode *GraphBuilder::MakeTensorCopy(ValueNode *tensor) {
  py::object prim_cast = Utils::GetModuleAttr("mindspore.ops.functional", "_cast", false, true);
  DoLoadConst({LOAD_CONST, -1, prim_cast});
  push(tensor);
  push(tensor);
  DoAttrAccess({LOAD_ATTR, 0, "dtype"});
  DoCall({CALL_FUNCTION, 2});
  ValueNode *node = pop();
  return node;
}

bool GraphBuilder::DoSetItem(ValueNode *map, ValueNode *key, ValueNode *value) {
  // only support constant key
  if (!this->graph_->GuardValueNode(key)) {
    return false;
  }
  // erase side-effect
  ValueNode *side_effect_node = graph_->GetTracedNodes().back();
  graph_->GetTracedNodes().pop_back();

  // try to transform
  const auto &replace_map = graph_->GetSideEffect()->data()->modified_and_replaced_map();
  bool is_new_var = false;
  ValueNode *old_node = map;
  ValueNode *new_node = nullptr;
  AObject::Type type = map->GetVobj()->GetType();
  if (type == AObject::kTypeList) {
    is_new_var = map->GetOpcode() == BUILD_LIST && replace_map.find(map) == replace_map.end();
    new_node = TransformListSetItem(map, key, value);
  } else if (type == AObject::kTypeDict) {
    is_new_var = map->GetOpcode() == BUILD_MAP && replace_map.find(map) == replace_map.end();
    new_node = TransformDictSetItem(map, key, value, false);
  } else if (type == AObject::kTypeTensor) {
    push(map);
    DoAttrAccess({LOAD_ATTR, 0, "__setitem__"});
    push(key);
    push(value);
    bool success = DoCall({CALL_FUNCTION, 2});
    pop();
    return success;
  }
  // failed transform, restore side-effect
  if (new_node == nullptr) {
    graph_->GetTracedNodes().push_back(side_effect_node);
    return false;
  }
  bool is_referenced = false;
  ReplaceAll(old_node, new_node, &is_referenced);
  // check it is new variable and not escaped
  if (is_new_var && !is_referenced && map != value) {
    return true;
  }
  // restore and record
  this->graph_->GetTracedNodes().push_back(side_effect_node);
  this->graph_->GetSideEffect()->data()->RecordModifiedAndReplacedNode(old_node, new_node);
  this->graph_->GetSideEffect()->Record(side_effect_node);
  return true;
}

bool GraphBuilder::DoItemAccess(const Instr &instr) {
  int opcode = instr.op();
  if (opcode == BINARY_SUBSCR) {
    DoGetItem(instr);
  } else if (opcode == STORE_SUBSCR) {
    auto key = pop();
    auto map = pop();
    auto value = pop();
    NewValueNode(nullptr, instr, {value, map, key});
    DoSetItem(map, key, value);
  } else if (opcode == DELETE_SUBSCR) {
    auto key = pop();
    auto map = pop();
    NewValueNode(nullptr, instr, {map, key});
    DoSetItem(map, key, nullptr);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "parser got an error instruction " << instr.ToString();
  }
  return true;
}

bool GraphBuilder::DoStackOp(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  if (opcode == POP_TOP) {
    pop();
  } else if (opcode == ROT_TWO) {
    frame_.Rot(1);
  } else if (opcode == ROT_THREE) {
    frame_.Rot(2);
  } else if (opcode == ROT_FOUR) {
    frame_.Rot(3);
  } else if (opcode == ROT_N) {
    frame_.Rot(oparg - 1);
  } else if (opcode == DUP_TOP_TWO) {
    push(seek(1));
    push(seek(1));
  } else if (opcode == DUP_TOP) {
    push(seek(0));
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "parser got an error instruction " << instr.ToString();
  }
  return true;
}

bool GraphBuilder::DoLoadConst(const Instr &instr) {
  auto n = NewValueNode(AObject::Convert(instr.cnst()), instr, {});
  push(n);
  return true;
}

bool GraphBuilder::DoListToTuple(const Instr &instr) {
  ValueNode *list = pop();
  if (list->GetOpcode() == BUILD_LIST) {
    std::for_each(list->getInputs().begin(), list->getInputs().end(), [this](ValueNode *i) { this->push(i); });
    return DoBuildOp({BUILD_TUPLE, SizeToInt(list->getInputs().size())});
  }
  AObject *vo = list->GetVobj();
  if (vo && vo->GetType() == AObject::kTypeList) {
    vo = static_cast<AbstractList *>(vo)->ListToTuple();
  } else {
    vo = AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  ValueNode *tuple = NewValueNode(vo, instr, {list});
  push(tuple);
  return true;
}

bool GraphBuilder::DoGetIter(const Instr &instr) {
  ValueNode *iterable = pop();
  AObject *iterable_vobj = iterable->GetVobj();
  AObject *iterator_vobj = iterable_vobj ? iterable_vobj->GetIter() : AObject::MakeAObject(AObject::kTypeAnyValue);

  std::vector<ValueNode *> inputs{iterable};
  auto *iter_node = graph_->allocator().NewNode<IterNode>(iterable, iterator_vobj, instr.op(), instr.arg(), inputs);
  iter_node->SetGraph(graph_);
  iter_node->set_bci(cur_bci_);
  iter_node->SetLineNo(instr.line());
  graph_->GetTracedNodes().push_back(iter_node);

  push(iter_node);
  iter_node->marker_ = 0;
  return true;
}

bool GraphBuilder::DoMakeFunction(const Instr &instr) {
  int oparg = instr.arg();
  // int cnt = __builtin_popcount(oparg & 0xf) + 2;
  int cnt = !!(oparg & 0x08) + !!(oparg & 0x04) + !!(oparg & 0x02) + !!(oparg & 0x01) + 2;
  std::vector<ValueNode *> p(frame_.GetStacks().end() - cnt, frame_.GetStacks().end());
  popn(cnt);
  AObject *f = AObject::MakeFunction(CollectObjects(p), graph_->GetGlobals(), oparg);
  ValueNode *func = NewValueNode(f, instr, p);
  push(func);
  current_block_->SetTrackResult(Block::kHasGlobalSideEffect);
  return true;
}

AObject *GraphBuilder::InferUnary(ValueNode *node, const Instr &instr) { return node->GetVobj()->Unary(instr.op()); }

bool GraphBuilder::DoUnary(const Instr &instr) {
  ValueNode *node = pop();

  AObject *object_info = InferUnary(node, instr);
  ValueNode *new_node = NewValueNode(object_info, instr, {node});
  push(new_node);
  return true;
}

bool GraphBuilder::DoIsOp(const Instr &instr) { return DoBinary(instr); }

bool GraphBuilder::DoContainsOp(const Instr &instr) { return DoBinary(instr); }

AObject *GraphBuilder::InferBinary(ValueNode *left, ValueNode *right, const Instr &instr) {
  AObject *object_info;
  if (instr.op() == IS_OP || instr.op() == CONTAINS_OP) {
    object_info = left->GetVobj()->Binary(right->GetVobj(), instr.op());
    PyObject *object = object_info != nullptr ? object_info->GetPyObject().ptr() : nullptr;
    if (object != nullptr) {
      object_info = AObject::Convert(py::bool_((object == Py_True) ^ instr.arg()));
    }
  } else if (Opcode(instr.op()).IsBinaryMath()) {
    if (left->IsConstantValue() && right->IsConstantValue()) {
      // compute real tensor value, not infer fake value
      AbstractObject *tensor = static_cast<AbstractObject *>(left->GetVobj());
      object_info = tensor->AbstractObject::Binary(right->GetVobj(), instr.op());
    } else {
      object_info = left->GetVobj()->Binary(right->GetVobj(), instr.op());
    }
  } else {
    return AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  return object_info;
}

bool GraphBuilder::DoBinary(const Instr &instr) {
  ValueNode *right = pop();
  ValueNode *left = pop();

  AObject *object_info = InferBinary(left, right, instr);
  ValueNode *new_node = NewValueNode(object_info, instr, {left, right});
  push(new_node);
  return true;
}

static bool CheckTupleListMul(ValueNode *left, ValueNode *right) {
  bool special = left->GetOpcode() == BUILD_LIST || left->GetOpcode() == BUILD_TUPLE;
  if (!special && left->IsConstantValue()) {
    AObject::Type l_type = left->GetVobj()->GetType();
    special = l_type == AObject::kTypeTuple || l_type == AObject::kTypeList;
  }
  if (special && right->IsConstantValue()) {
    PyObject *mul = right->GetVobj()->GetPyObject().ptr();
    const int max = 2;
    return PyLong_Check(mul) && Py_ABS(Py_SIZE(mul)) < max;
  }
  return false;
}

bool GraphBuilder::DoBinaryMul(const Instr &instr) {
  if (!CheckTupleListMul(seek(1), seek(0))) {
    return DoBinary(instr);
  }

  ValueNode *right = pop();
  ValueNode *left = pop();
  int l_op = left->GetVobj()->GetType() == AObject::kTypeTuple ? BUILD_TUPLE : BUILD_LIST;

  Py_ssize_t mul = PyLong_AsSsize_t(right->GetVobj()->GetPyObject().ptr());
  for (auto i = mul; i > 0; --i) {
    UnpackElements(left);
  }
  int oparg = left->getInputs().size() * (mul < 0 ? 0 : size_t(mul));
  DoBuildOp({l_op, oparg});
  return true;
}

static bool CheckTupleListAdd(ValueNode *left, ValueNode *right) {
  // type must be same
  AObject::Type l_type = left->GetVobj()->GetType();
  AObject::Type r_type = right->GetVobj()->GetType();
  bool support = l_type == AObject::kTypeTuple || l_type == AObject::kTypeList;
  if (!support || l_type != r_type) {
    return false;
  }
  // only handle BUILD_TUPLE and BUILD_LIST
  int l_op = left->GetOpcode();
  int r_op = right->GetOpcode();
  bool special = l_op == BUILD_TUPLE || l_op == BUILD_LIST || l_op == LOAD_CONST;
  bool accept = r_op == BUILD_TUPLE || r_op == BUILD_LIST || r_op == LOAD_CONST;
  if (!special || !accept) {
    return false;
  }
  return true;
}

bool GraphBuilder::DoInplaceAdd(const Instr &instr) {
  AObject::Type l_type = seek(1)->GetVobj()->GetType();
  if (l_type == AObject::kTypeTuple) {
    return DoBinaryAdd(instr);
  }
  if (!CheckTupleListAdd(seek(1), seek(0))) {
    return DoBinary(instr);
  }

  ValueNode *right = pop();
  ValueNode *left = pop();
  int l_op = BUILD_LIST;

  int size = this->frame_.GetStacks().size();
  UnpackElements(left);
  UnpackElements(right);
  size = this->frame_.GetStacks().size() - size;
  DoBuildOp({l_op, size});

  ValueNode *new_node = pop();
  if (ReplaceAll(left, new_node)) {
    push(new_node);
    return true;
  }
  graph_->GetTracedNodes().pop_back();
  push(left);
  push(right);
  return DoBinary(instr);
}

bool GraphBuilder::DoBinaryAdd(const Instr &instr) {
  if (!CheckTupleListAdd(seek(1), seek(0))) {
    return DoBinary(instr);
  }

  ValueNode *right = pop();
  ValueNode *left = pop();
  int l_op = left->GetVobj()->GetType() == AObject::kTypeTuple ? BUILD_TUPLE : BUILD_LIST;

  int size = this->frame_.GetStacks().size();
  UnpackElements(left);
  UnpackElements(right);
  size = this->frame_.GetStacks().size() - size;
  DoBuildOp({l_op, size});
  return true;
}

bool GraphBuilder::DoCompare(const Instr &instr) {
  Opcode opcode(instr.op());
  int oparg = instr.arg();
  auto r = pop();
  auto l = pop();

  bool invert;
  AObject *o;
  if (oparg >= Py_LT && oparg <= Py_GE) {
    PyObject *left = l->GetVobj() ? l->GetVobj()->GetPyObject().ptr() : nullptr;
    PyObject *right = r->GetVobj() ? r->GetVobj()->GetPyObject().ptr() : nullptr;
    if (left && right) {
      if (CheckValueValid(l->GetVobj()) && CheckValueValid(r->GetVobj())) {
        o = AObject::Convert(PyObject_RichCompare(left, right, oparg));
        PyErr_Clear();
      } else if (l->GetVobj()->GetType() == AObject::kTypeTensor || r->GetVobj()->GetType() == AObject::kTypeTensor) {
        o = l->GetVobj()->GetType() == AObject::kTypeTensor ? l->GetVobj() : r->GetVobj();
        auto tensor_type = py::reinterpret_borrow<py::object>(GetMsTensorType());
        py::object dtype_bool = Utils::GetModuleAttr("mindspore.common.dtype", "bool_");
        auto result_tensor = tensor_type(o->GetPyObject(), dtype_bool);
        o = AObject::Convert(result_tensor);
      } else {
        o = AObject::MakeAObject(AObject::kTypeBool);
      }
    } else {
      o = AObject::MakeAObject(AObject::kTypeBool);
    }
  } else if (opcode.CheckIsOp(oparg, &invert)) {
    int res = AObject::BinaryIs(l->GetVobj(), r->GetVobj());
    o = res == -1 ? AObject::MakeAObject(AObject::kTypeBool) : AObject::Convert((res ^ invert) ? Py_True : Py_False);
  } else if (opcode.CheckContainsOp(oparg, &invert)) {
    int res = AObject::BinaryContains(l->GetVobj(), r->GetVobj());
    o = res == -1 ? AObject::MakeAObject(AObject::kTypeBool) : AObject::Convert((res ^ invert) ? Py_True : Py_False);
  } else {
    return false;
  }

  auto v = NewValueNode(o, instr, {l, r});
  push(v);
  return true;
}

bool GraphBuilder::DoBuildOp(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  int tmp_arg = oparg;
  tmp_arg += opcode == BUILD_CONST_KEY_MAP;
  tmp_arg += opcode == BUILD_MAP ? tmp_arg : 0;
  std::vector<ValueNode *> p(frame_.GetStacks().end() - tmp_arg, frame_.GetStacks().end());
  popn(tmp_arg);

  ValueNode *v;
  if (opcode == BUILD_CONST_KEY_MAP) {
    PyObject *keys = p.back()->GetVobj()->GetPyObject().ptr();
    MS_EXCEPTION_IF_CHECK_FAIL(keys && PyTuple_CheckExact(keys), "error bytecode BUILD_CONST_KEY_MAP");
    Py_ssize_t size = PyTuple_GET_SIZE(keys);
    MS_EXCEPTION_IF_CHECK_FAIL(size_t(size) + 1 == p.size(), "error args BUILD_CONST_KEY_MAP");
    std::vector<ValueNode *> build_inputs;
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject *item = PyTuple_GET_ITEM(keys, i);
      DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(item)});
      build_inputs.push_back(pop());
      build_inputs.push_back(p[i]);
    }
    AObject *vo = AObject::BuildOperations(CollectObjects(build_inputs), BUILD_MAP);
    v = NewValueNode(vo, instr, build_inputs);
    v->SetOpcode(BUILD_MAP);
    v->SetOparg(size);
  } else {
    AObject *vo = AObject::BuildOperations(CollectObjects(p), opcode);
    v = NewValueNode(vo, instr, p);
  }
  push(v);
  return true;
}

ValueNode *GraphBuilder::ReplaceMergeOp(int opcode, const std::vector<ValueNode *> &inputs) {
  ValueNode *origin = inputs[0];
  ValueNode *arg = inputs[1];
  ValueNode *arg2 = inputs.size() > 2 ? inputs[2] : nullptr;
  if (origin->GetOpcode() != BUILD_LIST && origin->GetOpcode() != BUILD_MAP) {
    MS_LOG(INFO) << "Stack node should be BUILD_LIST or BUILD_MAP, but actual is: " << origin->ToString();
    return nullptr;
  }
  std::vector<ValueNode *> build_inputs = origin->getInputs();
  int div = 2;
  if (opcode == LIST_APPEND) {
    build_inputs.push_back(arg);
    opcode = BUILD_LIST;
    div = 1;
  } else if (opcode == LIST_EXTEND) {
    if (arg->IsConstantValue()) {
      build_inputs = UnpackConstObject(arg->GetConstantInfo()->value());
    } else if (arg->GetOpcode() == BUILD_LIST || arg->GetOpcode() == BUILD_TUPLE) {
      build_inputs.insert(build_inputs.end(), arg->getInputs().begin(), arg->getInputs().end());
    } else {
      int size = GetIterableSize(arg);
      if (size < 0) {
        MS_LOG(ERROR) << "Invalid iterable object:" << arg->ToString();
        return nullptr;
      }
      if (size > 0) {
        push(arg);
        DoUnpack({UNPACK_SEQUENCE, size});
        std::vector<ValueNode *> res = {frame_.GetStacks().end() - size, frame_.GetStacks().end()};
        popn(size);
        build_inputs.insert(build_inputs.end(), res.rbegin(), res.rend());
      }
    }
    opcode = BUILD_LIST;
    div = 1;
  } else if (opcode == DICT_MERGE || opcode == DICT_UPDATE) {
    if (arg->GetOpcode() != BUILD_MAP) {
      return nullptr;
    }
    build_inputs.insert(build_inputs.end(), arg->getInputs().begin(), arg->getInputs().end());
    opcode = BUILD_MAP;
  } else if (opcode == MAP_ADD) {
    build_inputs.push_back(arg);
    build_inputs.push_back(arg2);
    opcode = BUILD_MAP;
  } else {
    MS_LOG(INFO) << "Unsupported bytecode: " << Opcode(opcode).name();
    return nullptr;
  }
  std::for_each(build_inputs.begin(), build_inputs.end(), [this](ValueNode *i) { this->push(i); });
  int oparg = SizeToInt(build_inputs.size()) / div;
  if (!DoBuildOp({opcode, oparg})) {
    return nullptr;
  }
  return pop();
}

bool GraphBuilder::DoMergeOp(const Instr &instr) {
  int opcode = instr.op();
  int oparg = instr.arg();
  int pos = oparg + (opcode == MAP_ADD);

  int index = this->frame_.GetStacks().size() - 1 - pos;
  ValueNode *container = seek(pos);
  std::vector<ValueNode *> inputs = {container, pop()};
  if (opcode == MAP_ADD) {
    inputs.insert(inputs.begin() + 1, pop());
  }

  // DICT_MERGE only generated when unpack-call in python3.9, all keys must be string
  // NOTE: DICT_MERGE opcode requires that *(stack_pointer - oparg - 2) is a function if has duplicate key
  // ...
  ValueNode *new_node = ReplaceMergeOp(opcode, inputs);
  if (new_node != nullptr) {
    this->frame_.GetStacks()[index] = new_node;
    return true;
  }

  return false;
}

bool GraphBuilder::DoFormatValue(const Instr &instr) {
  int oparg = instr.arg();
  std::vector<ValueNode *> arg;
  if ((oparg & FVS_MASK) == FVS_HAVE_SPEC) {
    arg.push_back(pop());
  }
  arg.insert(arg.begin(), pop());
  constexpr unaryfunc conv_fn[] = {nullptr, PyObject_Str, PyObject_Repr, PyObject_ASCII};
  constexpr size_t size = sizeof(conv_fn) / sizeof(conv_fn[0]);
  size_t which_conversion = oparg & FVC_MASK;
  bool have_fmt_spec = (oparg & FVS_MASK) == FVS_HAVE_SPEC;

  ValueNode *fmt_spec_node = have_fmt_spec ? arg.back() : nullptr;
  ValueNode *value_node = *arg.begin();
  py::object value = value_node->GetVobj()->GetPyObject();
  bool not_constant = std::any_of(arg.begin(), arg.end(), [](ValueNode *i) { return !i->IsConstantValue(); });

  if (0 < which_conversion && which_conversion < size) {
    if (not_constant) {
      value = py::object();
      value_node = nullptr;
    } else {
      value = py::reinterpret_steal<py::object>(conv_fn[which_conversion](value.ptr()));
      DoLoadConst({LOAD_CONST, -1, value});
      value_node = pop();
    }
  }
  ValueNode *result_node = nullptr;
  if (value.ptr() != nullptr) {
    if (PyUnicode_CheckExact(value.ptr()) && fmt_spec_node == nullptr) {
      result_node = value_node;
    } else if (not_constant) {
      /* Actually call format(). */
      result_node = nullptr;
    } else {
      /* Actually call format(). */
      PyObject *po = NULL;
      if (fmt_spec_node != nullptr) po = fmt_spec_node->GetVobj()->GetPyObject().ptr();
      py::object result = py::reinterpret_steal<py::object>(PyObject_Format(value.ptr(), po));
      DoLoadConst({LOAD_CONST, -1, result});
      result_node = pop();
    }
  }
  if (result_node != nullptr) {
    push(result_node);
    return true;
  }

  auto vo = AObject::MakeAObject(AObject::kTypeString);
  auto v = NewValueNode(vo, instr, arg);
  push(v);
  return true;
}

bool GraphBuilder::DoImport(const Instr &instr) {
  int opcode = instr.op();
  if (opcode == IMPORT_FROM) {
    // any object
    push(NewValueNode(AObject::MakeAObject(AObject::kTypeAnyValue), instr, {seek(0)}));
  } else if (opcode == IMPORT_STAR) {
    auto from = pop();
    NewValueNode(AObject::MakeAObject(AObject::kTypeAnyValue), instr, {from});
  } else if (opcode == IMPORT_NAME) {
    auto from_list = pop();
    auto level = pop();
    auto vo = AObject::MakeAObject(AObject::kTypeModule);
    auto v = NewValueNode(vo, instr, {level, from_list});
    push(v);
  } else {
    return false;
  }
  return true;
}

bool GraphBuilder::DoByteCode(const Instr &instr) {
  if (current_block_->is_loop_head() && !graph_->Config().GetBoolConfig(GraphJitConfig::kLoopUnrolling)) {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceLoop_Unsupported);
    return false;
  }

  auto func_iter = bytecode_meth_map_.find(instr.op());
  if (func_iter == bytecode_meth_map_.end()) {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceByteCode_Unsupported);
    MS_LOG(WARNING) << "ByteCode " << Opcode(instr.op()).name() << " is not supported yet.";
    return false;
  }
  MS_LOG(DEBUG) << "Do bytecode " << instr.ToString();
  bool infer_succ = false;
  {
    TraceGuard trace_guard(GetLocation(instr));
    infer_succ = (this->*(func_iter->second))(instr);
  }

  const auto &nodes = graph_->GetTracedNodes();
  for (auto i = nodes.rbegin(); i != nodes.rend() && (*i)->GetBlock() == nullptr; ++i) {
    (*i)->SetBlock(current_block_);
  }

  if (!infer_succ && graph_->GetStopTraceBci() == -1) {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceReasonUnknown);
    MS_LOG(WARNING) << "Set Unknown Reason to " << instr.ToString() << " at bci " << cur_bci_;
  }

  if (instr.op() == RETURN_VALUE) {
    return false;
  }
  if (!infer_succ) {
    MS_LOG(INFO) << "Function " << graph_->GetCodeName() << " break graph at: " << instr.ToString();
    return false;
  }

  if (instr.extra_jump() == nullptr) {
    ++cur_bci_;
  } else {
    bool valid = (cur_bci_ == instr.bci() + 1) || cur_bci_ == instr.extra_jump()->bci();
    MS_EXCEPTION_IF_CHECK_FAIL(valid, "error jump target");
  }
  if (cur_bci_ < current_block_->begin_ci() || cur_bci_ >= current_block_->end_ci()) {
    current_block_ = graph_->GetCFG()->GetBlockByBci(cur_bci_);
  }
  if (no_grad_) {
    Opcode opcode(instr.op());
    if (opcode == STORE_FAST) {
      py::object func = Utils::GetModuleAttr("mindspore.ops.operations", "StopGradient")();
      DoLoadConst({LOAD_CONST, -1, func});
      DoLocalAccess({LOAD_FAST, instr.arg()});
      DoCall({CALL_FUNCTION, 1});
      DoLocalAccess({STORE_FAST, instr.arg()});
    }
  }
  return true;
}

GraphBuilder::GraphBuilder(const PyFrameWrapper &f)
    : root_(this), parent_(nullptr), graph_(nullptr), current_block_(nullptr), no_grad_(false) {
#if IS_PYTHON_3_11_PLUS
  MS_LOG(ERROR) << "not implement in python3.11";
#else
  auto co_wrapper = f.GetCode();
  PyCodeObject *co = co_wrapper.ptr();
  int argc = co_wrapper.ArgCount();
  int nfrees = co_wrapper.FreeVarsSize();
  int nlocals = co->co_nlocals;
  int fast_local_size = co_wrapper.FastLocalSize();
  int free_offset = fast_local_size - nfrees;
  auto varnames_release_handle = co_wrapper.VarNames();
  PyObject *names = varnames_release_handle.ptr();
  PyObject **fast_locals = f.FastLocal();
  graph_ = NewGraph(co, f.Globals().ptr());
  frame_.ResizeLocal(nlocals);
  for (int i = 0; i < argc; i++) {
    if (fast_locals[i] == nullptr) {
      continue;
    }
    auto vo = AObject::Convert(fast_locals[i]);
    ParamNode *n = graph_->NewParamNode(vo, i, PyUnicode_AsUTF8(PyTuple_GET_ITEM(names, i)));
    frame_.SetLocal(i, n);
    graph_->GetSideEffect()->data()->Track(fast_locals[i], n);
  }
  frame_.ResizeClosure(fast_local_size - nlocals);
  for (int fast_index = nlocals; fast_index < fast_local_size; fast_index++) {
    PyObject *cell = fast_locals[fast_index];
    if (cell == nullptr) {
      continue;
    }
    int offset = fast_index - nlocals;
    int oparg = IS_PYTHON_3_11_PLUS ? fast_index : offset;
    AbstractNode::Type t = fast_index < free_offset ? AbstractNode::CellVar : AbstractNode::FreeVar;
    PyObject *cell_contents = PyCell_GET(cell);
    CellVarNode *n = graph_->NewCellNode(AObject::Convert(cell), LOAD_CLOSURE, oparg);
    graph_->GetTracedNodes().push_back(n);
    frame_.SetClosure(offset, n);
    if (t == AbstractNode::CellVar && co->co_cell2arg != nullptr && co->co_cell2arg[offset] != CO_CELL_NOT_AN_ARG) {
      MS_EXCEPTION_IF_NULL(cell_contents);
      n->SetFromParam(co->co_cell2arg[offset]);
    }
    if (cell_contents == nullptr) {
      n->SetValue(&ValueNode::kUnboundLocal);
    } else {
      // The bci of this ValueNode is set to 0.
      ValueNode *param = NewValueNode(AObject::Convert(cell_contents), LOAD_DEREF, oparg);
      graph_->GetTracedNodes().push_back(param);
      param->SetGraph(graph_);
      n->AddCellOper(param);
      n->SetValue(param);
    }
  }
#endif
}

void GraphBuilder::CollectInlineInfo(CallNode *node, int depth) {
  Graph *sub_graph = node->GetSubGraph();
  if (!sub_graph) {
    return;
  }
  std::string inline_name = "";
  int code_size = 0;
  if (sub_graph != nullptr && sub_graph->GetCodeObj() != nullptr) {
    inline_name = py::str(reinterpret_cast<PyObject *>(sub_graph->GetCodeObj())).cast<std::string>();
    code_size = _PyCode_NBYTES(sub_graph->GetCodeObj());
  }
  std::string func_name = graph_->GetCodeName();
  std::string root_name = root_->GetGraph()->GetCodeName();
  JitCompileResults *jcr = GetJitCompileResults(root_->GetGraph()->GetCodeObj());
  if (jcr && jcr->tbs() && !func_name.empty()) {
    jcr->tbs()->PushInlineInfo(
      {func_name, inline_name, root_name, node->GetInlineReason(), code_size, depth, node->GetLineNo()});
  }
}

void GraphBuilder::HandleLoop() {
  Block *loop_head = graph_->GetCFG()->GetBlockByBci(cur_bci_);
  if (!loop_head->is_loop_head()) {
    return;
  }
  /**
   * (chaiyouheng): before trace start, unrolling loop. avoid graph status is changed while trace loop
   *       just unrolling a small loop that call nn.CellList.
   *
   * LoopUnrolling loopUnrollingExe = LoopUnrolling(*graph_);
   * (void)loopUnrollingExe.ExecuteLoopUnroll(loop_head);
   */
}

py::object GraphBuilder::FindPyFunc(AObject *vobj) {
  if (!vobj) {
    return py::cast<py::object>(nullptr);
  }

  switch (vobj->GetType()) {
    case AObject::kTypeCell:
      vobj = vobj->GetAttr(ID_construct);
      break;
    case AObject::kTypeAnyValue:
      vobj = vobj->GetAttr(ID___call__);
      break;
    case AObject::kTypeType:
      vobj = vobj->GetAttr("__init__");
      break;
    case AObject::kTypeBoundMethod:
      vobj = vobj->GetAttr("__func__");
    default:
      break;
  }
  py::object func = vobj ? vobj->GetPyObject() : py::object();

  if (func.ptr() == nullptr) {
    PyErr_Clear();
    return py::cast<py::object>(nullptr);
  }

  if (PyMethod_Check(func.ptr())) {
    func = py::reinterpret_borrow<py::object>(PyMethod_GET_FUNCTION(func.ptr()));
  }

  if (PyFunction_Check(func.ptr())) {
    if (PyFunction_GET_CODE(func.ptr()) == CaptureContext::GetInstance()->wrapper_code()) {
      return func.attr("__wrapped__");
    }
    return func;
  }
  return py::cast<py::object>(nullptr);
}

py::object GraphBuilder::GetFuncInfo(ValueNode *func_node) {
  AObject *vobj = func_node->GetVobj();
  if (vobj->GetType() == AObject::kTypeCFunction) {
    return py::object();
  }
  if (func_node->GetOpcode() == MAKE_FUNCTION) {
    return func_node->GetVobj()->GetPyObject();
  }
  return FindPyFunc(vobj);
}

bool GraphBuilder::WhiteListFuncCheckAndInfer(CallNode *call_node, const py::object &callable) {
  const auto &conf = call_node->GetGraph()->Config();

  AObject::Type vobj_type = call_node->input(0)->GetVobj()->GetType();
  if (vobj_type == AObject::kTypeCell) {
    current_block_->SetTrackResult(Block::kTrackHasOpsPrimitive);
    std::string module_name = GetTopModule(callable);
    if (!module_name.empty()) {
      kPIJitConfigDefault.AddAllowedInlineModules(module_name);
    }
  }

  bool infer_primitive = conf.GetBoolConfig(GraphJitConfig::kInferPrimitive);
  int max_infer = conf.getIntConfig(GraphJitConfig::kInferPrimitiveMax);
  if (max_infer != 0 && infer_func_count >= max_infer) {
    infer_primitive = false;
  } else {
    infer_func_count++;
  }
  infer_primitive &= (conf.getIntConfig(GraphJitConfig::kInferPrimitiveMask) & infer_primitive_func) != 0;
  if (!infer_primitive && vobj_type == AObject::kTypePrimitive) {
    call_node->SetVobj(AObject::MakeAObject(AObject::kTypeTensor));
    call_node->SetInlineReason(InlineReason::kInlineGraphSupportedByMS);
    current_block_->SetTrackResult(Block::kTrackHasOpsPrimitive);
    return true;
  }

  InferFunc infer_func = FindInferFunc(callable);
  if (infer_func == nullptr) {
    return false;
  }

  call_node->SetInlineReason(InlineReason::kInlineUnknown);
  call_node->SetSubGraph(NewGraph(nullptr, nullptr));
  call_node->GetSubGraph()->SetGuard(root_->GetGraph()->GetGuard());
  infer_func(call_node, this);

  InlineReason r;
  if (call_node->GetSubGraph() == nullptr) {
    r = InlineReason::kInlineFuncSpecialize;
  } else {
    MS_EXCEPTION_IF_NULL(call_node->GetSubGraph()->GetRetVal());
    r = InlineReason::kInline;
    seek(0) = call_node->GetSubGraph()->GetRetVal();
  }
  if (call_node->GetInlineReason() == InlineReason::kInlineUnknown) {
    call_node->SetInlineReason(r);
  }
  return true;
}

bool UnsupportedCodeTypeCheck(PyCodeObject *co) {
  if (co->co_flags & (CO_GENERATOR | CO_COROUTINE | CO_ASYNC_GENERATOR)) {
    MS_LOG(DEBUG) << "generator is unsupported";
    return true;
  }
  /**
   * skip super call
   * >>>def super_wrapper(self):
   * ...    __class__=type(self)
   * ...    def super_init(self):
   * ...        return super()
   * ...    return super_init(self)
   * >>>assert super(int, 1).__hash__() == super_wrapper(1).__hash__()
   */
  return false;
}

bool ApplyInlinePolicy(CallNode *call_node) {
  Graph *g = call_node->GetSubGraph();
  if (g == nullptr || g->GetRetVal() == nullptr) {
    return false;
  }
  if (g->GetRetVal()->GetOpcode() == MAKE_FUNCTION) {
    return false;
  }
  for (auto i : g->GetTracedNodes()) {
    // check MAKE_FUNCTION is alive, it is incorrect that inline the function of different module with MAKE_FUNCTION
    auto begin = i->getInputs().begin();
    if (Opcode(i->GetOpcode()).IsCall() && static_cast<CallNode *>(i)->GetInlineReason() == InlineReason::kInline) {
      begin++;
    }
    if (std::any_of(begin, i->getInputs().end(), [](ValueNode *n) { return n->GetOpcode() == MAKE_FUNCTION; })) {
      return false;
    }
  }
  return true;
}

bool CheckSupportCreateInstance(CallNode *call_node) {
  /**
   * only support exactly type, sub-class not create
   */
  static const std::set<PyTypeObject *> support_create_instance_type = {
    &PyComplex_Type, &PyMap_Type,   &PyBaseObject_Type, &PyRange_Type, &PyZip_Type,    &PySlice_Type,
    &PyBool_Type,    &PyFloat_Type, &PyLong_Type,       &PyType_Type,  &PyMethod_Type,
  };

  AObject *cls_info = call_node->input(0)->GetVobj();
  PyTypeObject *tp = reinterpret_cast<PyTypeObject *>(static_cast<AbstractType *>(cls_info)->GetPyObject().ptr());
  if (tp == nullptr) {
    return false;
  }
  if (PyExceptionClass_Check(tp)) {
    return true;
  }
  if (support_create_instance_type.find(tp) != support_create_instance_type.end()) {
    return true;
  }

  // create empty container(list(),tuple(),dict(),zip())
  static const std::set<PyTypeObject *> support_empty_create_instance_type = {
    &PyList_Type, &PyTuple_Type, &PySet_Type, &PyFrozenSet_Type, &PyDict_Type, &PyZip_Type};
  if (call_node->getInputs().size() == 1 &&
      support_empty_create_instance_type.find(tp) != support_empty_create_instance_type.end()) {
    return true;
  }

  /**
   * maybe has sideeffect, limit create
   */
  static const std::set<PyTypeObject *> limit_create_instance_type = {
    &PyList_Type, &PyTuple_Type, &PySet_Type, &PyFrozenSet_Type, &PyDict_Type, &PyUnicode_Type, &PyEnum_Type,
  };
  if (call_node->getInputs().size() != 2) {
    return false;
  }
  ValueNode *iterable_node = call_node->input(1);
  AObject *first_param = iterable_node->GetVobj();
  if (first_param == nullptr) {
    return false;
  }

  if (first_param->GetType() == AObject::kTypeAnyValue) {
    if (iterable_node->GetOpcode() != CALL_FUNCTION || call_node->bci() - 1 != iterable_node->bci()) {
      return false;
    }
    /**
     * just process this case:
     *    z = list(zip(list(x), list(y)))
     *    z = list(enumerate(x))
     */
    // this case, zip object and enumerate object is dead variable
  }
  return limit_create_instance_type.find(tp) != limit_create_instance_type.end();
}

AObject *GraphBuilder::BuildSuperObject(PyCodeObject *co) {
#if IS_PYTHON_3_11_PLUS
  MS_LOG(ERROR) << "not implement in python3.11";
  return nullptr;
#else
  AObject *super_obj = nullptr;
  if (co->co_argcount == 0) {
    PyErr_SetString(PyExc_RuntimeError, "super(): no arguments");
    return nullptr;
  }

  Py_ssize_t i, n;
  // search self object
  PyObject *obj = SearchSelfPyObject(co).first;
  if (obj == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "super(): arg[0] deleted");
    return nullptr;
  }

  if (co->co_freevars == NULL) {
    n = 0;
  } else {
    assert(PyTuple_Check(co->co_freevars));
    n = PyTuple_GET_SIZE(co->co_freevars);
  }

  PyTypeObject *type = NULL;
  for (i = 0; i < n; i++) {
    PyObject *name = PyTuple_GET_ITEM(co->co_freevars, i);
    assert(PyUnicode_Check(name));
    // check class id
    if (!strcmp("__class__", PyUnicode_AsUTF8(name))) {
      Py_ssize_t index = PyTuple_GET_SIZE(co->co_cellvars) + i;
      PyObject *cell = SetLocalPyObject(frame_.Closure(index));
      if (cell == NULL || !PyCell_Check(cell)) {
        PyErr_SetString(PyExc_RuntimeError, "super(): bad __class__ cell");
        return nullptr;
      }
      type = reinterpret_cast<PyTypeObject *>(PyCell_GET(cell));
      if (type == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "super(): empty __class__ cell");
        return nullptr;
      }
      if (!PyType_Check(type)) {
        PyErr_Format(PyExc_RuntimeError, "super(): __class__ is not a tyep (%s)", Py_TYPE(type)->tp_name);
        return nullptr;
      }
      break;
    }
  }
  if (type == NULL) {
    PyErr_SetString(PyExc_RuntimeError, "super(): __class__ cell not found");
    return nullptr;
  }

  py::object py_obj = py::reinterpret_borrow<py::object>(obj);
  py::object py_type = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(type));
  py::tuple tuple_obj(2);
  tuple_obj[0] = py_type;
  tuple_obj[1] = py_obj;
  PyObject *ret = PyObject_Call(reinterpret_cast<PyObject *>(&PySuper_Type), tuple_obj.ptr(), nullptr);
  super_obj = AObject::Convert(ret);
  Py_DECREF(ret);
  return super_obj;
#endif
}

bool GraphBuilder::ClassInstantiationFold(CallNode *call_node, AObject::Type type) {
  const auto &params = call_node->getInputs();
  int call_op = call_node->GetOpcode();

  // list, tuple, dict fold
  std::vector<ValueNode *> inputs;
  int new_op;
  int new_arg;
  if (type == AObject::kTypeTuple || type == AObject::kTypeList) {
    if (params.size() > 1) {
      int arg_op = params[1]->GetOpcode();
      if (call_op == CALL_FUNCTION && (arg_op == BUILD_TUPLE || arg_op == BUILD_LIST)) {
        inputs = params[1]->getInputs();
      } else {
        return false;
      }
    }
    new_op = type == AObject::kTypeTuple ? BUILD_TUPLE : BUILD_LIST;
    new_arg = inputs.size();
  } else if (type == AObject::kTypeDict) {
    if (params.size() > 1) {
      ValueNode *map_node;
      if (call_op == CALL_FUNCTION && params[1]->GetOpcode() == BUILD_MAP) {
        map_node = params[1];
      } else if (call_op == CALL_FUNCTION_EX && params.size() > 2 && params[2]->GetOpcode() == BUILD_MAP) {
        map_node = params[2];
      } else {
        return false;
      }
      inputs = map_node->getInputs();
    }
    new_op = BUILD_MAP;
    new_arg = inputs.size() / 2;
  } else {
    return false;
  }
  pop();
  std::for_each(inputs.begin(), inputs.end(), [this](auto n) { push(n); });
  return DoBuildOp({new_op, new_arg});
}

void LogGuardFailed(ValueNode *node, const GraphJitConfig &conf, const std::string &msg) {
  if (!conf.GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    return;
  }
  auto tr = GetTrace(node, false, true, 0, -1);
  std::stringstream s;
  if (node->GetVobj() == nullptr || node->GetVobj()->GetPyObject().ptr() == nullptr) {
    s << "infer failed\n";
  } else {
    std::map<Trace *, size_t> cache;
    s << "trace:\n" << (tr ? tr->FormatString(&cache).c_str() : "trace failed") << "\n";
  }
  s << msg << " [" << node->ToString() << "]";
  GRAPH_JIT_LOG_F("%s", s.str().c_str());
}

static py::object FilterCTensorInZip(AObject *vobj, py::object obj) {
  if (IsZipPyObject(reinterpret_cast<PyTypeObject *>(static_cast<AbstractType *>(vobj)->GetPyObject().ptr())) &&
      IsCTensorPyObject(obj.ptr())) {
    return ConvertCppTensorToMsTensor(obj);
  } else {
    return obj;
  }
}

ValueNode *GraphBuilder::HandleCallClass(CallNode *call_node) {
  AObject *vobj = call_node->input(0)->GetVobj();
  if (!vobj || vobj->GetType() != AObject::kTypeType) {
    return nullptr;
  }
  auto *t = static_cast<AbstractType *>(vobj);
  AObject::Type type = t->GetTypeType();
  if (type == AObject::kTypeTensor && HandleCallTensorClass(call_node)) {
    return call_node;
  }

  const auto &params = call_node->getInputs();
  AObject *instance = nullptr;
  bool support_create_instance = CheckSupportCreateInstance(call_node);
  bool constant = type == AObject::kTypePrimitive || type == AObject::kTypeTensor || type == AObject::kTypeStubTensor ||
                  IsMsClass(t->GetPyObject().ptr());
  // create instance
  if (support_create_instance || constant) {
    MS_LOG(INFO) << "Build instance, support_create_instance=" << support_create_instance << ", constant=" << constant;
    std::vector<py::object> args;
    std::transform(params.begin() + 1, params.end(), std::back_inserter(args), [vobj](ValueNode *n) {
      AObject *i = n->GetVobj();
      return i ? FilterCTensorInZip(vobj, i->GetPyObject()) : py::object();
    });
    py::object res = t->BuildInstance(args, call_node->GetOpcode());
    instance = res.ptr() ? AObject::Convert(res) : nullptr;
  } else if (reinterpret_cast<PyTypeObject *>(vobj->GetPyObject().ptr()) == &PySuper_Type) {
    // take super ptr and compare with PySuper_Type
    MS_LOG(INFO) << "Build super object";
    instance = BuildSuperObject(graph_->GetCodeObj());
    this->graph_->GetTracedNodes().pop_back();
    if (PyErr_Occurred()) {
      throw py::error_already_set();
    }
  }

  if (!instance) {
    // create abstract instance
    MS_LOG(INFO) << "Build abstract instance";
    instance = t->BuildAbstractInstance(CollectObjects({params.begin() + 1, params.end()}), call_node->GetOpcode());
  }
  call_node->SetVobj(instance);
  return call_node;
}

// NOTE: must be copy __code__, copy.deepcopy do nothing for code object
static py::object CopyPyFunc(const py::object &o) {
  MS_EXCEPTION_IF_CHECK_FAIL(PyFunction_Check(o.ptr()), "must be function");
  PyFunctionObject *func = reinterpret_cast<PyFunctionObject *>(o.ptr());
  PyCodeObject *code = reinterpret_cast<PyCodeObject *>(func->func_code);
  PyObject *new_name_object = PyUnicode_FromFormat("%s%U", kPIJitCopyFuncKey, code->co_name);
  if (new_name_object == nullptr) {
    throw py::error_already_set();
  }
  py::object new_name = py::reinterpret_steal<py::object>(new_name_object);
  py::object new_code = PyCodeWrapper(code).DeepCopy();
  PyObject *new_func = PyFunction_NewWithQualName(new_code.ptr(), func->func_globals, new_name.ptr());
  PyFunctionObject *new_ff = reinterpret_cast<PyFunctionObject *>(new_func);
  REPLACE_PY_MEMBER(new_ff->func_closure, func->func_closure);
  REPLACE_PY_MEMBER(new_ff->func_defaults, func->func_defaults);
  REPLACE_PY_MEMBER(new_ff->func_kwdefaults, func->func_kwdefaults);
  REPLACE_PY_MEMBER(new_ff->func_annotations, func->func_annotations);

  return py::reinterpret_steal<py::object>(new_func);
}

py::object GetPIJitCopiedFunc(const py::object &func) {
  PyObject *res = PyObject_GetAttrString(func.ptr(), kPIJitCopyFuncKey);
  if (res != nullptr) {
    return py::reinterpret_steal<py::object>(res);
  }
  PyErr_Clear();
  py::object copy = CopyPyFunc(func);
  PyObject_SetAttrString(func.ptr(), kPIJitCopyFuncKey, copy.ptr());
  (void)pi_jit_should_compile(copy, py::dict(), py::none());
  return copy;
}

ValueNode *GetSelfFromMethod(ValueNode *method) {
  Opcode opcode(method->GetOpcode());
  if (opcode.IsCall() && opcode != CALL_FUNCTION_EX) {
    // method from the CALL_FUNCTION
    py::object tp = method->input(0)->GetVobj() ? method->input(0)->GetVobj()->GetPyObject() : py::object();
    if (tp.ptr() == reinterpret_cast<PyObject *>(&PyMethod_Type)) {
      return method->input(2);
    }
  }
  if (method->GetOpcode() != LOAD_ATTR) {
    return nullptr;
  }
  ValueNode *self = method->input(0);
  PyTypeObject *tp = self->GetVobj() ? self->GetVobj()->GetTypeObject() : nullptr;
  PyTypeObject *real_tp = method->GetVobj()->GetAttr("__self__")->GetTypeObject();
  return tp == real_tp ? self : nullptr;
}

bool GraphBuilder::ReplaceCall(CallNode *call_node, const py::object &old_func) {
  if (call_node->GetOpcode() == CALL_FUNCTION_EX && call_node->input(1)->GetOpcode() != BUILD_TUPLE) {
    // dynamic length variable arguments, user-defined unpack sequence
    return false;
  }
  if (!graph_->GuardInlinedFunc(call_node)) {
    return false;
  }
  auto jcr = GetJitCompileResults(old_func.ptr());
  if (jcr != nullptr && jcr->stat() != JitCompileResults::NEVER_COMPILE) {
    return true;
  }

  py::object new_func = GetPIJitCopiedFunc(old_func);

  auto &nodes = graph_->GetTracedNodes();
  size_t index = std::distance(nodes.rbegin(), std::find(nodes.rbegin(), nodes.rend(), call_node));
  MS_EXCEPTION_IF_CHECK_FAIL(index < nodes.size(), "can't find call node");
  ++index;

  ValueNode *self = nullptr;
  AObject::Type func_type = call_node->input(0)->GetVobj()->GetType();
  if (func_type == AObject::kTypeBoundMethod) {
    ValueNode *func_val = call_node->input(0);
    self = GetSelfFromMethod(func_val);
    if (self == nullptr) {
      ValueNode *node = NewValueNode(func_val->get_attr(GraphBuilder::ID___self__), LOAD_ATTR, -1, {func_val},
                                     GraphBuilder::ID___self__);
      node->SetGraph(call_node->GetGraph());
      nodes.insert(nodes.end() - index, node);
      self = node;
    }
  } else if (func_type == AObject::kTypeCell || AObject::kTypeAnyValue) {
    self = call_node->input(0);
  } else if (func_type != AObject::kTypeFunction) {
    return false;
  }

  std::stringstream key;
  PyObject *func_name = reinterpret_cast<PyFunctionObject *>(new_func.ptr())->func_qualname;
  key << std::string(py::str(func_name)) << "." << new_func.ptr();

  // new func node
  DoLoadConst({LOAD_CONST, -1, new_func});
  ValueNode *func_node = pop();
  nodes.insert(nodes.end() - index, func_node);

  // replace node
  call_node->getInputs()[0] = func_node;
  if (self == nullptr) {
    return true;
  }

  // append self to args
  if (call_node->GetOpcode() != CALL_FUNCTION_EX) {
    call_node->getInputs().insert(call_node->getInputs().begin() + 1, self);
    call_node->SetOparg(call_node->GetOparg() + 1);
    return true;
  }

  // append self to variable arguments
  ValueNode *args_node = call_node->input(1);
  std::vector<ValueNode *> inputs = args_node->getInputs();
  inputs.insert(inputs.begin(), self);
  AObject *args_info = AObject::BuildOperations(CollectObjects(inputs), BUILD_TUPLE);

  ValueNode *tuple = this->NewValueNode(args_info, BUILD_TUPLE, inputs.size(), inputs);
  tuple->set_bci(call_node->bci());
  tuple->SetLineNo(call_node->GetLineNo());
  nodes.insert(nodes.end() - index, tuple);
  call_node->getInputs()[1] = tuple;
  return true;
}

MindGraphBuilder::MindGraphBuilder(const PyFrameWrapper &f) : GraphBuilder(f), side_effect_outputs_() {
  auto co = f.GetCode();
  const char *name = co.Name();
  int first_line = co.FirstLine();
  auto fg_builder = std::make_shared<FuncGraphBuilder>(true);
  fg_builder->SetGraphName(std::string() + name + "_" + std::to_string(first_line));
  co_name_ = name;

  graph_->set_func_graph_builder(fg_builder);
  this->FGAddTopInputs();
  auto add_local = [this](ValueNode *node) {
    if (node != &ValueNode::kUnboundLocal && node->abstract_wrapper() == nullptr) {
      MS_LOG(INFO) << "The python argument is Parameter, recompile if it's changed [" << node->ToString();
      node->set_abstract_wrapper(FGBuilder()->AddLocalVariable(node->GetVobj()->GetPyObject()));
    }
  };
  std::for_each(frame_.GetLocals().begin(), frame_.GetLocals().end(), add_local);
  for (auto n : frame_.GetClosures()) {
    add_local(n);
    add_local(n->GetValue());
  }
}

MindGraphBuilder::MindGraphBuilder(GraphBuilder *r, GraphBuilder *p, PyCodeObject *co, PyObject *globals)
    : GraphBuilder(r, p, co, globals), side_effect_outputs_() {
  auto fg_builder = std::make_shared<FuncGraphBuilder>();
  graph_->set_func_graph_builder(fg_builder);
}

namespace {
std::string GetFuncGraphName(const py::object &func, const MindGraphBuilderPtr &subgraph) {
  auto func_str = py::cast<std::string>(py::str(func));
  std::vector<std::string> vec;
  std::istringstream iss(func_str);
  std::string str;
  while (iss >> str) {
    (void)vec.emplace_back(str);
  }
  if (vec.size() <= 1) {
    return "";
  }
  auto func_name = vec[1];
  std::replace(func_name.begin(), func_name.end(), '.', '_');
  return func_name + "_" + std::to_string(subgraph->GetGraph()->GetCodeObj()->co_firstlineno);
}

bool CheckBuildSubGraph(const py::object &ret) {
  if (ret.ptr() == nullptr) {
    // ValueAny
    return true;
  }
  if (py::isinstance<py::str>(ret)) {
    std::string ret_str = ret.cast<std::string>();
    const std::string fake_grad_prefix = "FakeNodeKey MetaFuncGraph-grad";
    if (ret_str.substr(0, fake_grad_prefix.size()) == fake_grad_prefix) {
      return true;
    }
  }
  if (ret.ptr() == Py_None) {
    // Function return None, or has no return statement.
    return true;
  }
  return !CheckConstPyObject(ret.ptr());
}

std::string GetModuleName(const py::object &object) {
  PyObject *mod = PyObject_GetAttrString(object.ptr(), "__module__");
  const char *module_name = "";
  if (mod == nullptr) {
    PyErr_Clear();
  } else if (PyModule_Check(mod)) {
    module_name = PyModule_GetName(mod);
  } else if (PyUnicode_Check(mod)) {
    module_name = PyUnicode_AsUTF8(mod);
  }
  return std::string(module_name);
}

bool HasPyObj(const ValueNode *node) {
  if (node == nullptr) {
    MS_LOG(DEBUG) << "ValueNode is null";
    return false;
  }
  if (node->GetVobj() != nullptr && node->GetVobj()->GetPyObject().ptr() != nullptr) {
    return true;
  }
  MS_LOG(DEBUG) << "ValueNode's python object is null, node: " << node->ToString();
  return false;
}

void UpdateNodeInfo(const AbstractWrapperPtr &res, CallNode *call_node, StopTraceReason *stop_reason) {
  if (res == nullptr) {
    MS_LOG(INFO) << "Add node fail";
    *stop_reason = StopTraceReason::kTrace_Fail;
  } else {
    MS_LOG(INFO) << "Add node succ";
    auto node = AObject::Convert(res);
    MS_LOG(INFO) << node->ToString();
    call_node->SetVobj(node);
    call_node->set_abstract_wrapper(res);
    *stop_reason = StopTraceReason::kNonStopTrace;
  }
}
}  // namespace

AbstractWrapperPtrList MindGraphBuilder::HandleInputArgs(const std::vector<ValueNode *> args) {
  AbstractWrapperPtrList ret;
  for (auto arg : args) {
    MS_EXCEPTION_IF_NULL(arg);
    auto wrapper = arg->abstract_wrapper();
    if (wrapper == nullptr) {
      MS_LOG(INFO) << "infer failed with nullptr abstract wrapper";
      return {nullptr};
    }
    if (FGBuilder()->FindNodeByWrapper(wrapper) == nullptr) {
      std::stringstream s;
      s << std::endl << "the node can't be found " << arg->ToString() << std::endl;
      s << "the abstract is " << (wrapper == nullptr ? "<nullptr>" : wrapper->ToString()) << std::endl;
      s << "the inputs is: " << std::endl;
      for (const auto &n : args) {
        s << n->ToString() << std::endl;
      }
      s << std::endl << root_->GetGraph()->ToString() << std::endl;
      std::string debug_string = s.str();
      if (graph_->Config().GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
        GRAPH_JIT_LOG_F("%s", debug_string.c_str());
      }
      MS_LOG(INTERNAL_EXCEPTION) << "the node can't be found " << arg->ToString() << std::endl;
    }
    ret.push_back(wrapper);
  }
  return ret;
}

void MindGraphBuilder::HandleCustomBProp(const FuncGraphPtr &graph, const py::object &obj) const {
  if (graph == nullptr || obj.ptr() == nullptr) {
    return;
  }
  if (!py::hasattr(obj, parse::CUSTOM_BPROP_NAME)) {
    return;
  }
  bool enable_bprop_debug = py::cast<bool>(py::getattr(obj, "bprop_debug"));
  FuncGraphPtr bprop_graph = enable_bprop_debug
                               ? parse::ConvertToBpropCut(obj)
                               : parse::ConvertToFuncGraph(obj, {}, parse::PYTHON_MOD_GET_BPROP_METHOD);
  if (bprop_graph != nullptr) {
    (void)graph->transforms().emplace(parse::CUSTOM_BPROP_NAME, FuncGraphTransform(bprop_graph));
    (void)bprop_graph->transforms().emplace("primal", FuncGraphTransform(graph));
    graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
    graph->set_flag(FUNC_GRAPH_FLAG_PRIMAL_OF_BPROP, true);
    MS_LOG(INFO) << "Add custom bprop to graph.";
  }
  return;
}

StopTraceReason MindGraphBuilder::BuildSubGraph(CallNode *call_node, int depth, const py::object &func,
                                                const GraphBuilderPtr &subgraph) {
  auto sg = std::dynamic_pointer_cast<MindGraphBuilder>(subgraph);
  sg->FGBuilder()->AddPrevBuilder(FGBuilder());
  sg->FGBuilder()->set_manager(FGBuilder()->manager());

  auto code = sg->GetGraph()->GetGuard();
  MS_EXCEPTION_IF_NULL(code);
  code->GetGuard()->Backup();

  std::vector<ValueNode *> args;
  if (PyFunction_Check(func.ptr())) {
    args = GetNewArgs(call_node, AObject::Convert(func.ptr()), subgraph);
  } else {
    const auto &call_node_inputs = call_node->getInputs();
    (void)std::copy(call_node_inputs.begin() + 1, call_node_inputs.end(), std::back_inserter(args));
  }

  MS_LOG(INFO) << "new subgraph->TraceRun: " << py::str(func);
  bool succ = sg->FGAddInputs(args);
  if (!succ) {
    MS_LOG(INFO) << "Add input fail for new subgraph->TraceRun: " << py::str(func);
    return StopTraceReason::kStopTraceFunc_ArgHandle_Unsupported;
  }
  call_node->SetSubGraph(sg->GetGraph());
  auto reason = sg->TraceRun();
  MS_LOG(INFO) << "new subgraph->TraceRun end: " << py::str(func);

  sg->CollectSideEffectOutputs();
  auto sub_ret = sg->GetGraph()->GetRetVal();
  if (sub_ret == nullptr || !sub_ret->has_abstract_wrapper()) {
    MS_LOG(INFO) << "Failed to build subgraph for call node " << call_node->ToString();
    sg->RollbackSideEffectRecords();
  } else if (sg->side_effect_outputs_.empty() && sub_ret->abstract_wrapper()->IsConstant() &&
             !CheckBuildSubGraph(sub_ret->GetVobj()->GetPyObject())) {
    // If there are side effect outputs, we need to build sub-graph and add these nodes as graph outputs.
    MS_LOG(INFO) << "Subgraph ret value is const type, will not build subgraph, the node is " << call_node->ToString();
    call_node->SetVobj(sub_ret->GetVobj());
    auto ret_wrapper = FGBuilder()->AddLocalVariable(sub_ret->GetVobj()->GetPyObject());
    if (ret_wrapper != nullptr) {
      call_node->set_abstract_wrapper(ret_wrapper);
      MS_LOG(INFO) << "Constant fold call node " << call_node->ToString() << " to wrapper " << ret_wrapper->ToString();
    }
  } else {
    sg->FGBuilder()->SetGraphName(GetFuncGraphName(func, sg));

    const FuncGraphPtr &sub_graph = BuildSubFuncGraph(sg, args, call_node);
    if (sub_graph == nullptr) {
      sg->RollbackSideEffectRecords();
      return StopTraceReason::kTrace_Fail;
    }
    auto callable_obj = GetPyObject(call_node->input(0));
    if (py::isinstance<Cell>(callable_obj)) {
      HandleCustomBProp(sub_graph, callable_obj);
    }
  }
  if (call_node->input(0)->GetOpcode() != MAKE_FUNCTION) {
    graph_->GuardInlinedFunc(call_node);
  }
  return reason;
}

void MindGraphBuilder::CollectSideEffectOutputs() {
  const auto &side_effect_nodes = graph_->GetSideEffect()->GetRequiredNodes();
  std::copy_if(side_effect_nodes.begin(), side_effect_nodes.end(), std::back_inserter(side_effect_outputs_),
               [this](ValueNode *node) { return node->GetGraph() == graph_; });
}

namespace {
void CollectAllSubGraphs(Graph *cur_graph, std::unordered_set<Graph *> *graphs) {
  for (auto node : cur_graph->GetTracedNodes()) {
    if (node->GetType() == ValueNode::Call) {
      auto call_node = static_cast<CallNode *>(node);
      if (call_node->GetSubGraph() != nullptr) {
        graphs->insert(call_node->GetSubGraph());
        CollectAllSubGraphs(call_node->GetSubGraph(), graphs);
      }
    }
  }
}
}  // namespace

void MindGraphBuilder::RollbackSideEffectRecords() {
  if (side_effect_outputs_.empty()) {
    return;
  }
  std::unordered_set<Graph *> graphs{graph_};
  CollectAllSubGraphs(graph_, &graphs);

  std::set<ValueNode *> new_side_effect_nodes;
  for (const auto &pair : graph_->GetSideEffect()->nodes()) {
    ValueNode *node = pair.first;
    // Exclude the side-effect nodes generated in current graph (and all its subgraphs).
    if (graphs.find(node->GetGraph()) == graphs.end()) {
      new_side_effect_nodes.insert(node);
    }
  }
  graph_->GetSideEffect()->ResetRecord(new_side_effect_nodes);
}

FuncGraphPtr MindGraphBuilder::BuildSubFuncGraph(const MindGraphBuilderPtr &subgraph_builder,
                                                 const std::vector<ValueNode *> &args, CallNode *call_node) {
  bool succ = subgraph_builder->FGAddOutput();
  if (!succ) {
    return nullptr;
  }
  const FuncGraphPtr &sub_graph = subgraph_builder->FGBuilder()->graph();
  if (sub_graph == nullptr) {
    MS_LOG(INFO) << "Subgraph's func graph is NULL";
    return nullptr;
  }
  const AbstractWrapperPtr &res = FGBuilder()->AddNode(sub_graph, HandleInputArgs(args));
  if (res == nullptr) {
    MS_LOG(INFO) << "Add fg node failed: " << sub_graph->ToString();
    return nullptr;
  }
  MS_LOG(INFO) << "Add fg node success: " << res->ToString();

  succ = HandleSubGraphOutput(res, subgraph_builder, call_node);
  return succ ? sub_graph : nullptr;
}

bool MindGraphBuilder::HandleSubGraphOutput(const AbstractWrapperPtr &output,
                                            const MindGraphBuilderPtr &subgraph_builder, CallNode *call_node) {
  if (subgraph_builder->FGBuilder()->GetOutputSize() == 1) {
    // Only the output of function call, no side effect output.
    call_node->SetVobj(AObject::Convert(output));
    call_node->set_abstract_wrapper(output);
  } else {
    // output[0] is function call output, and output[1:] are side effect outputs.
    AbstractWrapperPtr call_node_output = FGTupleGetItem(output, 0);
    if (call_node_output == nullptr) {
      return false;
    }
    call_node->SetVobj(AObject::Convert(call_node_output));
    call_node->set_abstract_wrapper(call_node_output);
    // Add sub-graph's side effect outputs to parent-graph.
    for (size_t i = 0; i < subgraph_builder->side_effect_outputs_.size(); ++i) {
      AbstractWrapperPtr side_effect_output = FGTupleGetItem(output, SizeToInt(i + 1));
      if (side_effect_output == nullptr) {
        return false;
      }
      // Build the mapping from sub-graph's ValueNode to parent-graph's AnfNode.
      AnfNodePtr side_effect_anf_node = FGBuilder()->ReadLocalVariable(side_effect_output);
      MS_EXCEPTION_IF_NULL(side_effect_anf_node);
      AbstractWrapperPtr side_effect_abs = subgraph_builder->side_effect_outputs_[i]->abstract_wrapper();
      FGBuilder()->UpdateNodesMap(side_effect_abs, side_effect_anf_node);
    }
    (void)std::copy(subgraph_builder->side_effect_outputs_.begin(), subgraph_builder->side_effect_outputs_.end(),
                    std::back_inserter(side_effect_outputs_));
  }
  return true;
}

AbstractWrapperPtr MindGraphBuilder::FGTupleGetItem(const AbstractWrapperPtr &tuple, int index) {
  MS_EXCEPTION_IF_NULL(tuple);
  AbstractWrapperPtr idx = FGBuilder()->AddLocalVariable(py::int_(index));
  AbstractWrapperPtr ret = FGBuilder()->AddNode(prim::kPrimTupleGetItem, {tuple, idx});
  if (ret == nullptr || ret->abstract() == nullptr) {
    MS_LOG(INFO) << "Failed to do tuple getitem, index: " << index << ", tuple: " << tuple->ToString();
    return nullptr;
  }
  return ret;
}

// build sub-graph
StopTraceReason GraphBuilder::BuildSubGraph(CallNode *call_node, int depth, const py::object &func,
                                            const GraphBuilderPtr &subgraph) {
  InlineReason stat = InlineReason::kInline;
  bool is_make_func = call_node->input(0)->GetOpcode() == MAKE_FUNCTION;

  auto code = subgraph->GetGraph()->GetGuard();
  MS_EXCEPTION_IF_NULL(code);
  code->GetGuard()->Backup();

  MS_LOG(INFO) << "old subgraph->TraceRun";
  subgraph->TraceRun();

  call_node->SetSubGraph(subgraph->GetGraph());
  if (subgraph->GetGraph()->GetRetVal() != nullptr) {
    call_node->SetVobj(subgraph->GetGraph()->GetRetVal()->GetVobj());
  }
  bool gen_to_tuple = subgraph->GetGraph()->Config().GetBoolConfig(GraphJitConfig::kEnableGeneratorExpressionToTuple);
  if (!gen_to_tuple && subgraph->GetGraph()->GetGeneratorResult() != nullptr) {
    subgraph->GetGraph()->SetRetVal(nullptr);
  }

  stat = ApplyInlinePolicy(call_node) ? stat : InlineReason::kInlinePolicyDisabled;
  if (stat != InlineReason::kInline) {
    code->GetGuard()->Rollback();
    if (!is_make_func) {
      /**
       * replace function call, inline or resume capture after break graph
       * exclude make function, because of function always a new function but code is constant
       **/
      stat = ReplaceCall(call_node, func) ? stat : InlineReason::kInlinePolicyDisabled;
    }
  } else {
    if (!is_make_func) {
      // exclude make function, because of function always a new function but code is constant
      stat = graph_->GuardInlinedFunc(call_node) ? stat : InlineReason::kInlinePolicyDisabled;
    }
    if (stat != InlineReason::kInline) {
      code->GetGuard()->Rollback();
    } else {
      code->GetGuard()->Pop();
    }
  }

  // if stat == InlineReason::kInline, guard free variable
  call_node->SetInlineReason(stat);
  return StopTraceReason::kNonStopTrace;
}

bool GraphBuilder::UnpackDynamicLengthDictByBytecode(std::vector<ValueNode *> *params, CallNode *call_node,
                                                     ValueNode *dict_node) {
  // user defined mappings, dynamic length dictionary unpack
  if (dict_node->GetVobj()->GetType() != AObject::kTypeDict) {
    return false;
  }
  auto dict = static_cast<AbstractDict *>(dict_node->GetVobj());
  if (!dict->IsElementValid()) {
    return false;
  }
  /**
   * must be guard this dict length
   */
  py::dict py_dict = dict->GetPyObject();
  py::tuple keys(py_dict.size());
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  Py_ssize_t cnt = 0;
  while (PyDict_Next(py_dict.ptr(), &pos, &key, &value)) {
    PyObject *py_key = key;
    MS_EXCEPTION_IF_CHECK_FAIL(PyUnicode_CheckExact(py_key), "key must be string");
    PyObject *py_value = value;
    ValueNode *index = NewValueNode(AObject::Convert(py_key), LOAD_CONST, -1, {});
    ValueNode *val = NewValueNode(AObject::Convert(py_value), BINARY_SUBSCR, 0, {dict_node, index});
    keys[cnt++] = py_key;
    params->push_back(val);
    call_node->AddParam(val);
  }
  ValueNode *const_keys = NewValueNode(AObject::Convert(keys), LOAD_CONST, -1, {});
  params->push_back(const_keys);
  return true;
}

bool GraphBuilder::UnpackCallExDict(std::vector<ValueNode *> *params, CallNode *call_node) {
  ValueNode *dict_node = params->back();
  params->clear();
  if (dict_node->GetOpcode() != BUILD_MAP) {
    return UnpackDynamicLengthDictByBytecode(params, call_node, dict_node);
  }
  if (dict_node->GetOparg() == 0) {
    return true;
  }
  py::tuple keys(dict_node->GetOparg());
  for (int i = 0; i < dict_node->GetOparg(); ++i) {
    AObject *k = dict_node->input(i * 2)->GetVobj();
    if (k->GetType() != AObject::kTypeString) {
      MS_LOG(DEBUG) << "for unpack-call, dict keys must be string";
      return false;
    }
    keys[i] = k->GetPyObject();
    params->push_back(dict_node->input((i << 1) + 1));
    MS_EXCEPTION_IF_CHECK_FAIL(keys[i].ptr(), "the keys of unpack-call must be a const string");
  }
  ValueNode *const_keys = this->NewValueNode(AObject::Convert(keys), LOAD_CONST, -1, {});
  params->push_back(const_keys);
  return true;
}

bool GraphBuilder::UnpackDynamicLengthTupleByBytecode(std::vector<ValueNode *> *params, ValueNode *args_node,
                                                      CallNode *call_node) {
  // user-defined sequence, dynamic length tuple unpack
  if (args_node->GetVobj() && args_node->GetVobj()->GetType() != AObject::kTypeTuple &&
      args_node->GetVobj()->GetType() != AObject::kTypeList) {
    return false;
  }
  AbstractTuple *tuple = static_cast<AbstractTuple *>(args_node->GetVobj());
  if (!tuple->IsElementValid()) {
    return false;
  }
  /**
   * must be guard this tuple length
   */
  auto items = tuple->items();
  std::vector<ValueNode *> args;
  for (size_t i = 0; i < items.size(); i++) {
    ValueNode *idx_node = this->NewValueNode(AObject::Convert(py::int_(i)), LOAD_CONST, -1, {});
    auto value = this->NewValueNode(items[i], BINARY_SUBSCR, 0, {args_node, idx_node});
    args.push_back(value);

    call_node->AddParam(value);
  }
  params->insert(params->begin(), args.begin(), args.end());
  return true;
}

// unpack CALL_FUNCTION_EX parameters
// should do this when bytecode analyze ? replace origin opcode
bool GraphBuilder::UnpackCallExParams(std::vector<ValueNode *> *params, int extra_local, bool *has_kw,
                                      CallNode *call_node) {
  bool has_dict = params->size() > 1;
  ValueNode *args_node = params->operator[](0);
  if (!has_dict) {
    params->clear();
  } else if (!UnpackCallExDict(params, call_node)) {
    return false;
  }
  *has_kw = params->size();
  if (args_node->GetOpcode() != BUILD_TUPLE && args_node->GetOpcode() != BUILD_LIST) {
    return UnpackDynamicLengthTupleByBytecode(params, args_node, call_node);
  }
  params->insert(params->begin(), args_node->getInputs().begin(), args_node->getInputs().end());
  return true;
}

bool GraphBuilder::PackKwParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame,
                                std::vector<ValueNode *> *kwvargs) {
  PyCodeWrapper co(PyFunction_GET_CODE(func.ptr()));
  AObject *keys_info = params->back()->GetVobj();
  if (params->back()->GetOpcode() != LOAD_CONST || keys_info->GetType() != AObject::kTypeTuple) {
    return false;  // other case
  }

  const int posonlyargcount = co.PositionOnlyArgCount();
  py::object varnames = co.VarNames();

  PyObject **vars = &PyTuple_GET_ITEM(varnames.ptr(), 0);
  bool has_va;
  bool has_kw_va;
  int argc = co.ArgCount(&has_va, &has_kw_va);
  argc = argc - has_va - has_kw_va;
  PyObject **kwnames = &PyTuple_GET_ITEM(keys_info->GetPyObject().ptr(), 0);
  const int k_cnt = PyTuple_GET_SIZE(keys_info->GetPyObject().ptr());
  // kwnames must be string
  MS_ASSERT(static_cast<AbstractTuple *>(keys_info)->GetElementType() == AObject::kTypeString);
  MS_EXCEPTION_IF_CHECK_FAIL(SizeToInt(params->size()) > k_cnt, "check param");

  int kw_2_p_cnt = 0;

  // for each kw argument
  for (int i = k_cnt - 1; i >= 0; --i) {
    PyObject *key = kwnames[i];
    // find position and kwonly argument for key
    int pos = std::find_if(vars, vars + argc, [&key](PyObject *k) { return !PyUnicode_Compare(key, k); }) - vars;
    if (pos < posonlyargcount) {
      MS_LOG(DEBUG) << "position only argument specified by key-word";
      return false;
    }

    ValueNode *v = *(params->end() - 1 - k_cnt + i);
    // if key is position arg, store it
    if (pos < argc) {
      frame->SetLocal(pos, v);
      kw_2_p_cnt++;
      continue;
    }
    DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(key)});
    ValueNode *k = pop();

    kwvargs->push_back(k);
    kwvargs->push_back(v);
  }

  params->resize(params->size() - 1 - k_cnt);
  if (!has_kw_va) {
    return kw_2_p_cnt == k_cnt;  // if not equal, too many key-word arguments
  }
  return true;
}

bool GraphBuilder::HandleKWParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame) {
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  std::vector<ValueNode *> kwvargs;
  if (!PackKwParams(func, params, frame, &kwvargs)) {
    // illegal arguments
    return false;
  }

  const int argc = co->co_argcount + co->co_kwonlyargcount;
  if (!(co->co_flags & CO_VARKEYWORDS)) {
    // kw_2_p_cnt == k_cnt, all kw arguments is positions arguments
    return true;
  }

  int kwvarg_loc = argc + ((co->co_flags & CO_VARARGS) ? 1 : 0);
  AObject *dict = AObject::BuildOperations(CollectObjects(kwvargs), BUILD_MAP);
  frame->SetLocal(kwvarg_loc, NewValueNode(dict, BUILD_MAP, kwvargs.size() / 2, kwvargs));

  static_cast<CallNode *>(seek(0))->AddParam(frame->Local(kwvarg_loc));
  return true;
}

bool GraphBuilder::CheckAndSetDefaultParams(const py::object &func, FrameStates *frame, int position_argc) {
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  PyObject *defs = PyFunction_GET_DEFAULTS(func.ptr());
  PyObject *kwdefs = PyFunction_GET_KW_DEFAULTS(func.ptr());

  const int argc = co->co_argcount + co->co_kwonlyargcount;
  py::object varnames_release_handle = PyCodeWrapper(co).VarNames();
  PyObject *vars = varnames_release_handle.ptr();

  int defs_off = defs ? co->co_argcount - PyTuple_GET_SIZE(defs) : INT_MAX;
  for (int i = position_argc; i < argc; ++i) {
    if (frame->Local(i) != &ValueNode::kUnboundLocal) {
      continue;
    }
    PyObject *val;
    if (i < co->co_argcount) {
      val = i < defs_off ? nullptr : PyTuple_GET_ITEM(defs, i - defs_off);
    } else {
      val = kwdefs == nullptr ? nullptr : PyDict_GetItem(kwdefs, PyTuple_GET_ITEM(vars, i));
    }
    if (val == nullptr) {
      MS_LOG(DEBUG) << "no " << (i < defs_off ? "" : "kw-") << "default parameter error";
      return false;
    }
    DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(val)});
    frame->SetLocal(i, pop());
  }
  return true;
}

ValueNode *GetBoundSelfHelper(CallNode *call_node, bool *check_method) {
  ValueNode *func_val = call_node->input(0);
  AObject *vo = func_val->GetVobj();
  PyObject *method_object = vo->GetPyObject().ptr();

  ValueNode *self = nullptr;
  *check_method = false;
  bool &is_method = *check_method;
  switch (vo->GetType()) {
    case AObject::kTypeBoundMethod:
      self = GetSelfFromMethod(func_val);
      is_method = true;
      break;
    case AObject::kTypeCell:
      self = func_val;
      break;
    case AObject::kTypeAnyValue:
      if (method_object != nullptr &&
          (Py_IS_TYPE(method_object, &PyMethodDescr_Type) || Py_IS_TYPE(method_object, &_PyMethodWrapper_Type))) {
        // no bound self
        break;
      }
      self = func_val;
      break;
    case AObject::kTypeCFunction:
      self = GetSelfFromKnownMethod(call_node);
      if (PyCFunction_GET_SELF(method_object) != nullptr && !PyModule_Check(PyCFunction_GET_SELF(method_object))) {
        is_method = true;
        if (func_val->GetOpcode() == LOAD_ATTR) {
          self = func_val->input(0);
          // check method is a generic attribute
        }
      }
      break;
    case AObject::kTypeFunction:
      break;
    default:
      MS_LOG(INTERNAL_EXCEPTION) << "unimplemented type " << vo->ToString();
      break;
  }
  return self;
}

ValueNode *GraphBuilder::GetBoundSelf(CallNode *call_node) {
  ValueNode *func_node = call_node->input(0);
  bool is_method = false;
  ValueNode *self = GetBoundSelfHelper(call_node, &is_method);
  if (is_method && self == nullptr) {
    push(func_node);
    DoAttrAccess({LOAD_ATTR, 0, std::string(GraphBuilder::ID___self__)});
    ValueNode *node = pop();
    call_node->AddParam(node);
    self = node;
  }
  return self;
}

bool GraphBuilder::HandlePositionParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame) {
  CallNode *call_node = reinterpret_cast<CallNode *>(seek(0));
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  auto vobj = AObject::Convert(func.ptr());
  AObject::Type callable_type = vobj->GetType();

  ValueNode *self = GetBoundSelf(call_node);
  if (self != nullptr) {
    params->insert(params->begin(), self);
  }

  const int argc = co->co_argcount;
  const int has_varg = (co->co_flags & CO_VARARGS) ? 1 : 0;
  const int has_kwvarg = (co->co_flags & CO_VARKEYWORDS) ? 1 : 0;
  const int varg_loc = argc + co->co_kwonlyargcount;
  const int kwvarg_loc = argc + co->co_kwonlyargcount + has_varg;
  int pargc = params->size();
  if (pargc > argc && !has_varg) {
    MS_LOG(DEBUG) << "too many parameters";
    return false;
  }
  bool append_self_to_varg = has_varg && self && callable_type == AObject::kTypeBoundMethod && argc == 0;
  if (append_self_to_varg) {  // self is in variable arguments
    MS_LOG(INFO) << "not implement append self to variable arguments, inline failed";
    return false;
  }

  if (has_kwvarg && frame->Local(kwvarg_loc) == &ValueNode::kUnboundLocal) {
    auto vo = AObject::Convert(py::dict());
    auto m = NewValueNode(vo, BUILD_MAP, 0, {});
    call_node->AddParam(m);
    frame->SetLocal(kwvarg_loc, m);
  }

  if (has_varg) {
    int vargc = pargc > argc ? pargc - argc : 0;
    std::vector<ValueNode *> vargs(params->end() - vargc, params->end());
    params->resize(params->size() - vargc);

    auto vo = AObject::BuildOperations(CollectObjects(vargs), BUILD_TUPLE);
    ValueNode *build_tuple = NewValueNode(vo, BUILD_TUPLE, vargc, vargs);
    call_node->AddParam(build_tuple);
    frame->SetLocal(varg_loc, build_tuple);
  }

  pargc = params->size();
  for (int i = pargc - 1; i >= 0; --i) {
    if (frame->Local(i) != &ValueNode::kUnboundLocal) {
      MS_LOG(DEBUG) << "duplicate key-word parameter error";
      return false;
    }
    frame->SetLocal(i, params->back());
    params->pop_back();
  }
  return CheckAndSetDefaultParams(func, frame, pargc);
}

bool GraphBuilder::HandleCallParameters(const py::object &func_info, CallNode *call_node, FrameStates *frame) {
#if IS_PYTHON_3_11_PLUS
  MS_LOG(ERROR) << "not implement in python3.11";
  return false;
#else
  if (func_info.ptr() == nullptr) {
    MS_LOG(EXCEPTION) << "HandleCallParameters with empty func_info input.";
  }
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func_info.ptr()));
  frame->ResizeLocal(co->co_nlocals);

  std::vector<ValueNode *> params(call_node->getInputs().begin() + 1, call_node->getInputs().end());
  int op = call_node->GetOpcode();
  bool has_kw = (op == CALL_FUNCTION_KW);
  if (op == CALL_FUNCTION_EX && !UnpackCallExParams(&params, co->co_nlocals, &has_kw, call_node)) {
    return false;  // ex_dict infer failed or user-defined sequence and map arguments
  }
  if (has_kw && !HandleKWParams(func_info, &params, frame)) {
    return false;
  }
  if (!HandlePositionParams(func_info, &params, frame)) {
    return false;
  }

  MS_EXCEPTION_IF_CHECK_FAIL(params.size() == 0, "check parameters handle");

  // python3.10 and lower only
  // after store all params
  // cell2arg
  const Py_ssize_t ncells = PyCodeWrapper(co).CellVarsSize();
  const Py_ssize_t *c2a_arr = co->co_cell2arg;
  for (int i = 0; c2a_arr != nullptr && i < ncells; ++i) {
    if (c2a_arr[i] != CO_CELL_NOT_AN_ARG) {
      Py_ssize_t arg_index = c2a_arr[i];
      CellVarNode *cell_node = frame->Closure(i);
      ValueNode *arg_node = frame->Local(arg_index);
      /**
       * here not delete the local, continue with local same as closure
       * frame->SetLocal(arg_index, &ValueNode::kUnboundLocal);
       */

      PyObject *cell = cell_node->GetVobj()->GetPyObject().ptr();
      PyObject *cell_contents = arg_node->GetVobj() ? arg_node->GetVobj()->GetPyObject().inc_ref().ptr() : nullptr;
      MS_EXCEPTION_IF_CHECK_FAIL(cell && PyCell_Check(cell) && PyCell_GET(cell) == nullptr, "must be a empty closure");

      ValueNode *n = NewValueNode(nullptr, STORE_DEREF, i, {arg_node});

      cell_node->AddCellOper(n);
      cell_node->SetValue(arg_node);
      Py_XSETREF(PyCell_GET(cell), cell_contents);
      // cell variable is eliminate
      // call_node->AddParam(n);
    }
  }
  return true;
#endif
}

static void SetGradFuncInfo(mindspore::pijit::CallNode *call_node);

bool GraphBuilder::ResolveNoGrad(CallNode *call_node, StopTraceReason *stop_reason) {
  AObject *callable = call_node->input(0)->GetVobj();
  py::object callable_info = callable->GetPyObject();
  bool is_nograd_enter = IsNoGradEnterFunc(callable_info);
  bool is_nograd_exit = IsNoGradExitFunc(callable_info);
  if (is_nograd_enter || is_nograd_exit) {
    call_node->SetVobj(AObject::Convert(Py_True));
    call_node->SetSubGraph(nullptr);
    *stop_reason = StopTraceReason::kNonStopTrace;
    no_grad_ = is_nograd_enter;
    return true;
  }
  return false;
}

py::object GraphBuilder::ResolveCallable(CallNode *call_node, StopTraceReason *stop_reason) {
  AObject *callable = call_node->input(0)->GetVobj();
  py::object callable_info;
  *stop_reason = StopTraceReason::kStopTraceInfer_Fail;
  call_node->SetInlineReason(InlineReason::kInlineInfer_Fail);
  if (!callable) {
    return callable_info;
  }
  callable_info = callable->GetPyObject();
  if (callable_info.ptr() == nullptr) {
    callable_info = py::cast<py::object>(reinterpret_cast<PyObject *>(callable->GetTypeObject()));
  }

  AObject::Type callable_type = callable->GetType();
  if (callable_info.ptr() == nullptr) {
    if (callable->TestMsFlag(AObject::kMsFlagGradFunc | AObject::kMsFlagShardFunc | AObject::kMsFlagVmapFunc)) {
      SetGradFuncInfo(call_node);
      *stop_reason = StopTraceReason::kNonStopTrace;
    }
    return py::object();
  }

  if (ResolveNoGrad(call_node, stop_reason)) {
    return py::object();
  }

  *stop_reason = StopTraceReason::kNonStopTrace;
  if (callable_type == AObject::kTypeType) {
    call_node->SetInlineReason(InlineReason::kInlineFunc_ArgType_IsClass);
    HandleCallClass(call_node);
    if (static_cast<AbstractType *>(callable)->GetTypeType() == AObject::kTypeCell) {
      *stop_reason = StopTraceReason::kStopTraceInfer_Fail;
    }
    return py::object();
  }

  if (WhiteListFuncCheckAndInfer(call_node, callable_info)) {
    if (call_node->GetInlineReason() == InlineReason::kInlineFunc_Type_Unsupported) {
      *stop_reason = StopTraceReason::kStopTraceFunc_Type_Unsupported;
    } else {
      graph_->GuardInlinedFunc(call_node);
    }
    return py::object();
  }

  // find code object
  callable_info = GetFuncInfo(call_node->input(0));
  if (callable_info.ptr() == nullptr) {
    *stop_reason = StopTraceReason::kStopTraceFunc_Type_Unsupported;
    call_node->SetInlineReason(InlineReason::kInlineCFunction_Unsupported);
  }
  return callable_info;
}

void GraphBuilder::ResolveClosure(const py::object &func_info, CallNode *call_node, FrameStates *frame) {
  if (func_info.ptr() == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "When resolving closure, get func_info failed.";
  }
  ValueNode *callable_node = call_node->input(0);
  PyCodeWrapper co(PyFunction_GET_CODE(func_info.ptr()));
  PyObject *closure = PyFunction_GET_CLOSURE(func_info.ptr());

  int ncells = co.CellVarsSize();
  int nfrees = co.FreeVarsSize();
  frame->ResizeClosure(ncells + nfrees);

  auto TrackExtraAttrArgs = [this, &call_node](ValueNode *src, const std::string &name) {
    push(src);
    DoAttrAccess({LOAD_ATTR, 0, name});
    ValueNode *attr_node = pop();
    call_node->AddParam(attr_node);
    return attr_node;
  };

  if (ncells > 0) {
    // track cell variable
    DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(&PyCell_Type))});
    ValueNode *type_node = pop();
    call_node->AddParam(type_node);
    for (int i = 0; i < ncells; i++) {
      auto obj_info = AObject::Convert(py::reinterpret_steal<py::object>(PyCell_New(nullptr)));
      CellVarNode *cell_node = graph_->NewCellNode(obj_info, CALL_FUNCTION, 0, {type_node});
      auto abs = static_cast<MindGraphBuilder *>(this)->FGBuilder()->AddLocalVariable(obj_info->GetPyObject());
      cell_node->set_abstract_wrapper(abs);
      call_node->AddParam(cell_node);
      frame->SetClosure(i, cell_node);
    }
  }
  if (nfrees > 0) {
    // track free variable
    ValueNode *func_node = nullptr;
    switch (callable_node->GetVobj()->GetType()) {
      case AObject::kTypeCell:
        func_node = TrackExtraAttrArgs(callable_node, "construct");
        break;
      case AObject::kTypeAnyValue:
        func_node = TrackExtraAttrArgs(callable_node, "__call__");
        break;
      case AObject::kTypeFunction:
      case AObject::kTypeBoundMethod:
        func_node = callable_node;
        break;
      default:
        MS_LOG(INTERNAL_EXCEPTION) << "can't find the function of object";
        break;
    }

    // track free variable
    bool make_func = func_node->GetOpcode() == MAKE_FUNCTION;
    ValueNode *closures_node = nullptr;
    if (make_func) {
      closures_node = *(func_node->getInputs().end() - 3);
    } else if (closure) {
      closures_node = TrackExtraAttrArgs(func_node, "__closure__");
    } else {
      MS_LOG(EXCEPTION) << "error no closure";
    }
    for (int i = 0; i < nfrees; ++i) {
      CellVarNode *freevar = nullptr;
      if (make_func) {
        MS_EXCEPTION_IF_CHECK_FAIL(closures_node->GetOpcode() == BUILD_TUPLE, "unknown closure source");
        freevar = reinterpret_cast<CellVarNode *>(closures_node->input(i));
      } else {
        // track the node
        push(closures_node);
        DoLoadConst({LOAD_CONST, -1, py::int_(i)});
        DoItemAccess({BINARY_SUBSCR, 0});
        auto tmp = pop();
        // replaced
        MS_EXCEPTION_IF_CHECK_FAIL(graph_->GetTracedNodes().back() == tmp, "can't find the node");
        freevar = graph_->NewCellNode(tmp->GetVobj(), BINARY_SUBSCR, 0, tmp->getInputs());
        graph_->GetTracedNodes().back() = freevar;

        auto infer_result = tmp->GetVobj()->GetPyObject().ptr();
        auto actually_result = PyTuple_GET_ITEM(closure, i);
        // tuple and list will copy it, so only log it not error it
        if (infer_result != actually_result) {
          MS_LOG(INFO) << "LOAD_ATTR cell_contents of an cell object maybe failed, cell object is " << py::str(closure)
                       << " but infer result is  " << py::str(infer_result);
        }
        // must be equal
        MS_ASSERT(py::handle(infer_result).equal(actually_result));

        call_node->AddParam(freevar);
        auto cell_contents_node = TrackExtraAttrArgs(freevar, "cell_contents");
        MS_EXCEPTION_IF_NULL(cell_contents_node->GetVobj()->GetPyObject().ptr());
        freevar->SetValue(cell_contents_node);
        freevar->set_abstract_wrapper(tmp->abstract_wrapper());
      }
      frame->SetClosure(ncells + i, freevar);
    }
  }
}

void SetMixedPrecisionType(CallNode *call_node, FrameStates *frame) {
  auto func_node = call_node->input(0);
  if (func_node->GetVobj() && func_node->GetVobj()->GetType() == AbstractObjectBase::kTypeCell) {
    auto cell = py::cast<CellPtr>(func_node->GetVobj()->GetPyObject());
    auto mixed_type = cell->GetMixedPrecisionType();
    if (mixed_type != MixedPrecisionType::kNotSet) {
      for (size_t i = 0; i < frame->GetLocals().size(); i++) {
        if (frame->Local(i)->GetType() == AbstractNode::Param) {
          auto paramNode = reinterpret_cast<ParamNode *>(frame->Local(i));
          if (paramNode->GetVobj()->GetType() == AObject::kTypeTensor &&
              !paramNode->GetVobj()->GetPyObject().attr("dtype").is_none()) {
            auto src_dtype = paramNode->GetVobj()->GetPyObject().attr("dtype");
            bool is_cast = false;
            if (py::isinstance<Float>(src_dtype)) {
              auto float_nbits = py::cast<Float>(src_dtype).nbits();
              if (float_nbits == 64 || (float_nbits == 32 && mixed_type != kFP32) ||
                  (float_nbits == 16 && mixed_type != kFP16)) {
                is_cast = true;
              }
            }
            if (py::isinstance<BFloat>(src_dtype) && mixed_type != kBF16) {
              is_cast = true;
            }
            if (!is_cast) {
              continue;
            }
            auto dst_dtype = Utils::MixedPrecisionTypeToDType(mixed_type);
            paramNode->SetMixedPrecisionType(dst_dtype);
          }
        }
      }
    }
  }
}

StopTraceReason GraphBuilder::HandleCall(int depth) {
  MS_EXCEPTION_IF_CHECK_FAIL(seek(0)->GetType() == ValueNode::Call, "must be call node");
  CallNode *call_node = reinterpret_cast<CallNode *>(seek(0));
  if (depth > root_->graph_->Config().getIntConfig(GraphJitConfig::kMaxInlineDepth)) {
    call_node->SetInlineReason(InlineReason::kInlineTooDeep);
    return StopTraceReason::kNonStopTrace;
  }
  StopTraceReason stop_reason = StopTraceReason::kNonStopTrace;

  py::object callable_info = ResolveCallable(call_node, &stop_reason);
  if (callable_info.ptr() == nullptr) {
    if (stop_reason != StopTraceReason::kNonStopTrace) {
      MS_LOG(INFO) << "Handle call for node " << call_node->ToString()
                   << " failed. Stop reason: " << GetStopTraceReasonDesc(stop_reason);
    }
    return stop_reason;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(PyFunction_Check(callable_info.ptr()), "'ResolveCallable' must be return a function");

  // unsupported check
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(callable_info.ptr()));
  PyObject *globals = PyFunction_GET_GLOBALS(callable_info.ptr());
  auto subgraph = GraphBuilder::Creator(this->root_ ? this->root_ : this, this, co, globals);
  this->sub_graph = subgraph;

  // frame build
  FrameStates *frame = &(subgraph->frame_);
  ResolveClosure(callable_info, call_node, frame);
  if (!HandleCallParameters(callable_info, call_node, frame)) {
    call_node->SetInlineReason(InlineReason::kInlineFunc_ArgHandle_Unsupported);
    return StopTraceReason::kStopTraceFunc_ArgHandle_Unsupported;
  }

  SetMixedPrecisionType(call_node, frame);
  // build sub-graph
  stop_reason = BuildSubGraph(call_node, depth, callable_info, subgraph);
  CollectInlineInfo(call_node, depth);
  return stop_reason;
}

static bool GuardLoopSequence(Graph *graph, ValueNode *seq_node, Py_ssize_t seq_size) {
  // guard length
  PyObject *seq = seq_node->GetVobj()->GetPyObject().ptr();
  if (seq != nullptr && seq_size == -1) {
    seq_size = PySequence_Size(seq);
  }
  if (seq == nullptr || seq_size == -1) {
    PyErr_Clear();
    return false;
  }
  if (!graph->GuardSequenceNodeLength(seq_node, seq_size)) {
    return false;
  }
  if (!graph->GuardType(seq_node)) {
    return false;
  }
  return true;
}

bool GuardIterInputs(Graph *graph, ValueNode *seq_node, Py_ssize_t seq_size = -1) {
  PyObject *seq = seq_node->GetVobj()->GetPyObject().ptr();
  if (seq != nullptr && seq_size == -1) {
    seq_size = PySequence_Size(seq);
  }
  if (seq == nullptr || seq_size == -1) {
    PyErr_Clear();
    return false;
  }
  if (!graph->GuardSequenceNodeLength(seq_node, seq_size)) {
    return false;
  }
  auto input_nodes = seq_node->getInputs();
  for (size_t i = 1; i < input_nodes.size(); ++i) {
    ValueNode *input_node = input_nodes[i];
    if (input_node == nullptr) {
      return false;
    }
    TracePtr tr = graph->TraceValueNode(input_node);
    if (tr == nullptr) {
      return graph->GuardValueNodeClosure(input_node);
    }
    if (!(graph->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GEqual))) {
      MS_LOG(INFO) << "Iterator guard fail: " << seq_node->ToString();
      return false;
    }
  }
  MS_LOG(INFO) << "Iterator guard success: " << seq_node->ToString();
  return true;
}

bool GraphBuilder::TraceRunForIterSequence(int jump_bci) {
  MS_LOG(DEBUG) << "Start do sequence FOR_ITER";
  auto *iter_node = dynamic_cast<IterNode *>(seek(0));
  if (iter_node == nullptr) {
    MS_LOG(INFO) << "TOS node should be IterNode, but actual is: " << seek(0)->ToString();
    return false;
  }
  // check for iter
  ValueNode *seq_node = iter_node->iterable();
  PyObject *seq = seq_node->GetVobj()->GetPyObject().ptr();
  if (seq == nullptr) {
    MS_LOG(INFO) << "no sequence object for loop";
    return false;  // infer failed
  }
  Py_ssize_t size = PySequence_Size(seq);
  if (size == -1) {
    PyErr_Clear();
    MS_LOG(INFO) << "FOR_ITER without __len__";
    return false;
  }

  int index = SizeToInt(iter_node->index());
  if (index == 0 && seq_node->GetVobj()->GetType() == AObject::kTypeTuple) {
    DoLoadConst({LOAD_CONST, 0, py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(&PyTuple_Type))});
    push(seq_node);
    if (!DoCall({CALL_FUNCTION, 1})) {
      MS_LOG(INFO) << "Failed to convert sequence to tuple";
      return false;
    }
    ValueNode *new_tuple_node = pop();
    iter_node->set_iterable(new_tuple_node);
    seq_node = new_tuple_node;
  }
  if (index == 0 && !GuardLoopSequence(graph_, seq_node)) {
    // loop start.
    MS_LOG(INFO) << "guard loop sequence failed";
    return false;
  }

  if (index >= size) {
    MS_LOG(DEBUG) << "Loop end";
    pop();
    cur_bci_ = jump_bci;
    return true;
  }

  push(seq_node);
  DoLoadConst({LOAD_CONST, -1, py::object(py::int_(index))});
  if (!DoItemAccess({BINARY_SUBSCR, 0})) {
    return false;
  }

  iter_node->set_index(index + 1);
  cur_bci_ = cur_bci_ + 1;
  return true;
}

static bool CheckForIterEnumerate(ValueNode *iter_node) {
  ValueNode *enumerate_node = iter_node->input(0);
  if (enumerate_node->GetOpcode() != CALL_FUNCTION || iter_node->bci() - 1 != enumerate_node->bci()) {
    // enumerate object maybe alive, shouldn't reduce it
    MS_LOG(INFO) << "Unsupported enumerate node: " << enumerate_node->ToString();
    return false;
  }
  PyObject *enumerate = enumerate_node->GetVobj()->GetPyObject().ptr();
  if (enumerate == nullptr) {
    MS_LOG(INFO) << "enumerate() python object is null!";
    return false;
  }

  MS_EXCEPTION_IF_NULL(iter_node->GetGraph());

  ValueNode *iterable_node = enumerate_node->input(1);
  PyObject *iterable = iterable_node->GetVobj()->GetPyObject().ptr();
  if (iterable == nullptr || !PySequence_Check(iterable) || !GuardLoopSequence(iter_node->GetGraph(), iterable_node)) {
    // just support sequence iteration
    MS_LOG(INFO) << "Unsupported iterable node: " << iterable_node->ToString();
    return false;
  }
  return true;
}

bool GraphBuilder::TraceRunForIterEnumerate(int jump_bci) {
  MS_LOG(DEBUG) << "Start do enumerate FOR_ITER";
  ValueNode *iter_node = seek(0);
  if (iter_node->marker_ == 0) {
    if (!CheckForIterEnumerate(iter_node)) {
      return false;
    }
    iter_node->marker_ = 1;
  }
  ValueNode *enumerate_node = iter_node->input(0);
  PyObject *enumerate = enumerate_node->GetVobj()->GetPyObject().ptr();
  ValueNode *iterable_node = enumerate_node->input(1);

  // reduce iterable object
  ValueNode *seq_node = iterable_node;
  PyObject *obj = PyIter_Next(enumerate);
  if (obj == nullptr) {
    if (PyErr_Occurred() && !PyErr_ExceptionMatches(PyExc_StopIteration)) {
      MS_LOG(INFO) << "trace FOR_ITER got an error " << py::error_already_set().what();
      PyErr_Clear();
      return false;
    }
    PyErr_Clear();
    pop();
    cur_bci_ = jump_bci;
    return true;
  }

  auto tuple = py::reinterpret_steal<py::tuple>(obj);
  py::object index = tuple[0];
  ValueNode *result_node;
  DoLoadConst({LOAD_CONST, 0, index});
  ValueNode *index_node = pop();
  push(seq_node);
  push(index_node);
  if (!DoItemAccess({BINARY_SUBSCR, 0})) {
    return false;
  }
  ValueNode *item_node = pop();
  push(index_node);
  push(item_node);
  if (!DoBuildOp({BUILD_TUPLE, 2})) {
    return false;
  }
  result_node = pop();

  push(result_node);
  cur_bci_ = cur_bci_ + 1;
  return true;
}

static bool CheckForIterZip(ValueNode *iter_node) {
  ValueNode *zip_node = iter_node->input(0);
  if (zip_node->GetOpcode() != CALL_FUNCTION || iter_node->bci() - 1 != zip_node->bci()) {
    MS_LOG(INFO) << "Unsupported zip node: " << zip_node->ToString();
    return false;
  }
  PyObject *zip = zip_node->GetVobj()->GetPyObject().ptr();
  if (zip == nullptr) {
    MS_LOG(INFO) << "zip() python object is null!";
    return false;
  }
  MS_EXCEPTION_IF_NULL(iter_node->GetGraph());
  Graph *graph = iter_node->GetGraph();

  std::vector<ValueNode *> iterable_nodes = {zip_node->getInputs().begin() + 1, zip_node->getInputs().end()};
  auto iter = std::find_if(iterable_nodes.begin(), iterable_nodes.end(), [&graph](ValueNode *iterable_node) {
    PyObject *iterable = iterable_node->GetVobj()->GetPyObject().ptr();
    return iterable == nullptr || ((!PySequence_Check(iterable) || !GuardLoopSequence(graph, iterable_node)) &&
                                   (!IsTensorPyObject(iterable) || !graph->GuardValueNode(iterable_node, GDeduce)));
  });
  if (iter != iterable_nodes.end()) {
    MS_LOG(INFO) << "Unsupported iterable node: " << (*iter)->ToString();
    return false;
  }
  return true;
}

bool GraphBuilder::TraceRunForIterZip(int jump_bci) {
  MS_LOG(DEBUG) << "Start do zip FOR_ITER";
  ValueNode *iter_node = seek(0);
  int *index = &iter_node->marker_;
  if ((*index) == 0) {
    if (!CheckForIterZip(iter_node)) {
      return false;
    }
  }

  ValueNode *zip_node = iter_node->input(0);
  PyObject *zip = zip_node->GetVobj()->GetPyObject().ptr();
  std::vector<ValueNode *> iterable_nodes = {zip_node->getInputs().begin() + 1, zip_node->getInputs().end()};

  // reduce iterable object
  PyObject *tuple = PyIter_Next(zip);
  py::object handle = py::reinterpret_steal<py::object>(tuple);
  if (handle.ptr() == nullptr) {
    if (PyErr_Occurred() && !PyErr_ExceptionMatches(PyExc_StopIteration)) {
      MS_LOG(INFO) << "trace FOR_ITER got an error " << py::error_already_set().what();
      PyErr_Clear();
      return false;
    }
    PyErr_Clear();
    pop();
    cur_bci_ = jump_bci;
    return true;
  }

  for (auto seq_node : iterable_nodes) {
    DoLoadConst({LOAD_CONST, 0, py::int_(*index)});
    ValueNode *index_node = pop();
    push(seq_node);
    push(index_node);
    if (!DoItemAccess({BINARY_SUBSCR, 0})) {
      return false;
    }
  }
  if (!DoBuildOp({BUILD_TUPLE, SizeToInt(iterable_nodes.size())})) {
    return false;
  }

  (*index)++;
  cur_bci_ = cur_bci_ + 1;
  return true;
}

bool GraphBuilder::TraceRunForIterDict(int jump_bci) {
  MS_LOG(DEBUG) << "Start do dict FOR_ITER";
  auto *iter_node = dynamic_cast<IterNode *>(seek(0));
  if (iter_node == nullptr) {
    MS_LOG(INFO) << "TOS node should be IterNode, but actual is: " << seek(0)->ToString();
    return false;
  }
  MS_EXCEPTION_IF_NULL(iter_node->iterable());

  size_t index = iter_node->index();
  if (index == 0) {
    ValueNode *dict_node = iter_node->iterable();
    push(dict_node);
    DoAttrAccess({LOAD_METHOD, 0, "keys"});
    DoCall({CALL_METHOD, 0});
    ValueNode *keys_node = pop();
    DoLoadConst({LOAD_CONST, 0, py::cast<py::object>(reinterpret_cast<PyObject *>(&PyTuple_Type))});
    push(keys_node);
    if (!DoCall({CALL_FUNCTION, 1})) {
      MS_LOG(INFO) << "Fail to convert dict keys to tuple";
      return false;
    }
    auto tuple_node = pop();
    MS_EXCEPTION_IF_CHECK_FAIL(tuple_node->GetOpcode() == CALL_FUNCTION, "opcode should be CALL_FUNCTION");
    iter_node->set_iterable(tuple_node);
    if (!tuple_node->has_abstract_wrapper()) {
      MS_LOG(INFO) << "Fail to do dict get keys: " << dict_node->ToString();
      return false;
    }
  }

  ValueNode *keys_node = iter_node->iterable();
  AbstractWrapperPtr keys_wrapper = keys_node->abstract_wrapper();
  MS_EXCEPTION_IF_CHECK_FAIL(keys_wrapper != nullptr && keys_wrapper->abstract() != nullptr, "abstract is NULL!");
  auto keys_abstract = keys_wrapper->abstract()->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(keys_abstract);

  if (index >= keys_abstract->size()) {
    MS_LOG(DEBUG) << "End loop";
    pop();
    cur_bci_ = jump_bci;
    return true;
  }

  push(keys_node);
  DoLoadConst({LOAD_CONST, -1, py::object(py::int_(index))});
  if (!DoItemAccess({BINARY_SUBSCR, 0})) {
    return false;
  }
  iter_node->set_index(index + 1);
  cur_bci_ = cur_bci_ + 1;
  return true;
}

bool GraphBuilder::TraceRunForIterDictItems(int jump_bci) {
  MS_LOG(DEBUG) << "Start do dict iterators FOR_ITER";
  auto *iter_node = dynamic_cast<IterNode *>(seek(0));
  if (iter_node == nullptr) {
    MS_LOG(INFO) << "TOS node should be IterNode, but actual is: " << seek(0)->ToString();
    return false;
  }
  MS_EXCEPTION_IF_NULL(iter_node->iterable());

  int index = SizeToInt(iter_node->index());
  if (index == 0) {
    ValueNode *dict_item_node = iter_node->iterable();
    DoLoadConst({LOAD_CONST, 0, py::cast<py::object>(reinterpret_cast<PyObject *>(&PyTuple_Type))});
    push(iter_node->iterable());
    if (!DoCall({CALL_FUNCTION, 1})) {
      MS_LOG(INFO) << "Fail to convert to tuple";
      return false;
    }
    ValueNode *tuple_node = pop();
    iter_node->set_iterable(tuple_node);
    if (!dict_item_node->IsConstantValue()) {
      MS_LOG(INFO) << "Is not constant value, guard failed: " << dict_item_node->ToString();
    }
  }

  ValueNode *dict_node = iter_node->iterable();
  py::object key;
  py::object dict = dict_node->GetVobj()->GetPyObject();
  if (dict.ptr() == nullptr) {
    MS_LOG(INFO) << "infer failed \"dict(dict_items)\"";
    return false;
  }
  auto it = dict.begin();
  auto end = dict.end();
  for (int i = 0; it != end && i < index; ++it, ++i) {
  }
  if (it != end) {
    key = py::cast<py::object>(*it);
  }
  if (key.ptr() == nullptr) {
    // for end
    pop();  // iter node
    cur_bci_ = jump_bci;
    return true;
  }

  push(dict_node);
  DoLoadConst({LOAD_CONST, -1, py::object(py::int_(index))});
  if (!DoItemAccess({BINARY_SUBSCR, 0})) {
    return false;
  }
  iter_node->set_index(index + 1);
  cur_bci_ = cur_bci_ + 1;
  return true;
}

LocationPtr GraphBuilder::GetLocation(const Instr &instr) const {
  if (graph_ == nullptr || graph_->GetCodeObj() == nullptr) {
    return std::make_shared<Location>("anonymous", 0, 0, 0, 0, "", std::vector<std::string>());
  }
  auto file_name = PyCodeWrapper(graph_->GetCodeObj()).FileName();
  auto line_no = instr.line();
  std::vector<std::string> comments;
  return std::make_shared<Location>(file_name, line_no, 0, line_no, 0, "", std::move(comments));
}

bool GraphBuilder::TraceRunForIter(const Instr &instr) {
  MS_EXCEPTION_IF_NULL(instr.extra_jump());
  // check for iter
  ValueNode *iter_node = seek(0);
  AObject *iterable = iter_node->getInputs().empty() ? nullptr : iter_node->input(0)->GetVobj();
  bool succ;
  if (iter_node->GetOpcode() != GET_ITER) {
    MS_LOG(INFO) << "FOR_ITER without GET_ITER";
    succ = false;
  } else if (iterable == nullptr) {
    MS_LOG(INFO) << "iterable is null!";
    succ = false;
  } else if (iterable->GetTypeObject() == &PyEnum_Type) {
    succ = TraceRunForIterEnumerate(instr.extra_jump()->bci());
  } else if (iterable->GetTypeObject() == &PyZip_Type) {
    succ = TraceRunForIterZip(instr.extra_jump()->bci());
  } else if (iterable->GetPyObject().ptr() != nullptr && py::isinstance<py::dict>(iterable->GetPyObject())) {
    succ = TraceRunForIterDict(instr.extra_jump()->bci());
  } else if (iterable->GetTypeObject() == &PyDictKeys_Type || iterable->GetTypeObject() == &PyDictValues_Type ||
             iterable->GetTypeObject() == &PyDictItems_Type) {
    succ = TraceRunForIterDictItems(instr.extra_jump()->bci());
  } else if (iterable->GetPyObject().ptr() != nullptr && PySequence_Check(iterable->GetPyObject().ptr())) {
    succ = TraceRunForIterSequence(instr.extra_jump()->bci());
  } else {
    MS_LOG(INFO) << "Unsupported iterable type: "
                 << (iterable->GetTypeObject() != nullptr ? iterable->GetTypeObject()->tp_name : "NULL");
    succ = false;
  }
  if (!succ) {
    if (graph_->Config().GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
      GRAPH_JIT_LOG_F("loop unsupported by trace, iter node is [%s]", iter_node->ToString().c_str());
    }
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceLoop_Unsupported);
  }
  return succ;
}

static bool IsConstantBoolValue(ValueNode *node) {
  const auto &cnst_info = node->GetConstantInfo();
  if (cnst_info == nullptr) {
    return false;
  }
  if (cnst_info->value().ptr() != nullptr) {
    return true;
  }
  PyTypeObject *tp = cnst_info->type();
  if (tp == nullptr) {
    return false;
  }
  static const std::set<PyTypeObject *> len_to_bool = {&PyTuple_Type, &PyList_Type, &PyDict_Type};
  if (len_to_bool.find(tp) != len_to_bool.end() && cnst_info->len() != -1) {
    return true;
  }
  return false;
}

bool IsShapeOrDtypeRelatedNode(const ValueNode *node) {
  if (node->GetOpcode() == CALL_FUNCTION && node->input(0)->GetVobj()->GetType() == AObject ::kTypeCFunction &&
      node->input(0)->GetName() == "len") {
    node = node->input(1);
  }
  if (node->GetOpcode() == BINARY_SUBSCR) {
    node = node->input(0);
  }
  if (node->GetOpcode() == CALL_FUNCTION) {
    auto func_node = node->input(0);
    // prim
    if (py::isinstance<mindspore::PrimitivePyAdapter>(func_node->GetVobj()->GetPyObject())) {
      auto prime_name = py::cast<mindspore::PrimitivePyAdapterPtr>(func_node->GetVobj()->GetPyObject())->name();
      if (prime_name == "Shape" || prime_name == "DType" || prime_name == "Rank") {
        return true;
      }
    }
  } else if (node->GetOpcode() == LOAD_ATTR) {
    auto attr_name = node->GetName();
    if (attr_name == "dtype" || attr_name == "shape" || attr_name == "ndim" || attr_name == "size") {
      return true;
    }
  }
  return false;
}

bool TryGuardEscape(ValueNode *cond_node) {
  if (cond_node->GetOpcode() == COMPARE_OP &&
      std::any_of(cond_node->getInputs().begin(), cond_node->getInputs().end(),
                  [](const ValueNode *node) { return IsShapeOrDtypeRelatedNode(node); })) {
    return true;
  }
  if (cond_node->GetOpcode() == COMPARE_OP && cond_node->getInputs().size() == 2 &&
      cond_node->input(0)->GetOpcode() == BINARY_SUBSCR && cond_node->input(1)->GetOpcode() == BINARY_SUBSCR) {
    return true;
  }
  if (cond_node->GetOpcode() == CONTAINS_OP &&
      (IsShapeOrDtypeRelatedNode(cond_node->input(0)) || IsShapeOrDtypeRelatedNode(cond_node->input(1)))) {
    return true;
  }
  if (cond_node->GetOpcode() == CALL_FUNCTION &&
      std::all_of(cond_node->getInputs().begin() + 1, cond_node->getInputs().end(),
                  [](const ValueNode *node) { return IsShapeOrDtypeRelatedNode(node); })) {
    return true;
  }
  return false;
}

bool IsSatisfyPruneLimit(int cond, Graph *graph_, ValueNode *cond_node) {
  if (cond == -1) {
    return false;
  }
  int limit_prune = graph_->Config().getIntConfig(GraphJitConfig::kMaxPruneCase);
  if (limit_prune >= 0 && limit_prune < graph_->GetPruneBranchCount()) {
    return false;
  }
  if (IsConstantBoolValue(cond_node)) {
    return true;
  }
  auto tr = graph_->TraceValueNode(cond_node);
  if (tr == nullptr) {
    if (graph_->Config().getIntConfig(GraphJitConfig::kGuardRelaxCount) > 0) {
      PyObject *bool_value = cond_node->GetVobj()->GetPyObject().ptr();
      if ((bool_value == Py_True || bool_value == Py_False) && TryGuardEscape(cond_node)) {
        return true;
      }
    }
    return graph_->GuardValueNodeClosure(cond_node);
  }
  PyObject *bool_value = cond_node->GetVobj()->GetPyObject().ptr();
  if (bool_value != Py_True && bool_value != Py_False) {
    bool strict = graph_->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);
    auto bool_type = CreateOpTrace(reinterpret_cast<PyObject *>(&PyBool_Type), LOAD_CONST, -1, {}, "", "", strict);
    tr = CreateOpTrace(cond ? Py_True : Py_False, CALL_FUNCTION, 1, {bool_type, tr}, "", "", strict);
  } else {
    cond_node->SetConstantValue(true);
  }
  graph_->GetGuard()->GetGuard()->GuardOn(tr, GuardLevel::GId);
  return true;
}

static void LogPrunBranch(ValueNode *cond, const Instr &instr, const GraphJitConfig &conf) {
  MS_LOG(INFO) << "Fail to prune branch, instr: " << instr.ToString() << ", condition: " << cond->ToString();
  if (conf.GetBoolConfig(GraphJitConfig::kPrintGuard)) {
    GRAPH_JIT_LOG_F("Fail to prune bytecode [%s]!\n", instr.ToString().c_str());
  }

  if (conf.GetBoolConfig(GraphJitConfig::kLogGraphBreak)) {
    if (CondIsTrue(cond) == -1) {
      GRAPH_JIT_LOG_F("infer failed\n");
    } else {
      auto tr = GetTrace(cond, false, true, 0, conf.getIntConfig(GraphJitConfig::kMaxTraceDepth));
      std::map<Trace *, size_t> cache;
      GRAPH_JIT_LOG_F("trace:\n%s\n", tr ? tr->FormatString(&cache).c_str() : "trace failed");
    }
    if (cond->GetGraph() == nullptr || cond->GetGraph()->GetCodeObj() == nullptr) {
      return;
    }
    GRAPH_JIT_LOG_F("if branch prune failed, condition [%s] at [%U : %d]", cond->ToString().c_str(),
                    cond->GetGraph()->GetCodeObj()->co_filename, cond->GetLineNo());
  }
}

bool GraphBuilder::TraceRunControl(const Instr &instr) {
  MS_EXCEPTION_IF_NULL(instr.extra_jump());
  Opcode opcode(instr.op());
  ValueNode *cond_node = nullptr;
  int cond = -1;
  int jump_to = -1;
  if (opcode == JUMP_FORWARD || opcode == JUMP_ABSOLUTE) {
    cur_bci_ = instr.extra_jump()->bci();
    return true;
  } else if (opcode == FOR_ITER) {
    return TraceRunForIter(instr);
  } else if (opcode == POP_JUMP_IF_FALSE || opcode == POP_JUMP_IF_TRUE) {
    cond_node = pop();
    cond = CondIsTrue(cond_node);
    jump_to = ((cond == 0) ^ (opcode == POP_JUMP_IF_TRUE)) ? instr.extra_jump()->bci() : cur_bci_ + 1;
  } else if (opcode == JUMP_IF_FALSE_OR_POP || opcode == JUMP_IF_TRUE_OR_POP) {
    cond_node = seek(0);
    cond = CondIsTrue(cond_node);
    bool jump = (cond == 0) ^ (opcode == JUMP_IF_TRUE_OR_POP);
    cond_node = jump ? seek(0) : pop();
    jump_to = jump ? instr.extra_jump()->bci() : cur_bci_ + 1;
  } else {
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceByteCode_Unsupported);
    return false;
  }

  // if branch
  if (!IsSatisfyPruneLimit(cond, graph_, cond_node)) {
    LogPrunBranch(cond_node, instr, graph_->Config());
    graph_->StopTraceAt(cur_bci_, StopTraceReason::kStopTraceIf_Unsupported);
    return false;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(jump_to != -1, "error jump bci");
  cur_bci_ = jump_to;
  return true;
}

StopTraceReason GraphBuilder::TraceRun() {
  current_block_ = graph_->GetCFG()->GetFirstBB();
  cur_bci_ = 0;
  const auto &instrs = graph_->GetCFG()->instr_pool();
  while (true) {
    this->graph_->SetFrame(cur_bci_, frame_);
    MS_EXCEPTION_IF_CHECK_FAIL(static_cast<size_t>(cur_bci_) < instrs.size(), "error control flow");
    MS_EXCEPTION_IF_CHECK_FAIL(instrs[cur_bci_]->bci() == cur_bci_, "check instruction bci");
    if (!DoByteCode(*instrs[cur_bci_])) {
      break;
    }
  }
  return graph_->GetStopTraceReason();
}

extern void AddConfigToGuard(const GraphJitConfig &c, OptGuardPtr guard);
extern void AddGuardForParam(const PyFrameWrapper &f, OptGuardPtr guard, bool detach);

/**
 * Generate a graph from callable, this function will actually create python frame
 */
static std::unique_ptr<GraphBuilder> GenerateRootGraph(const py::object &callable, const py::object &args,
                                                       const py::object &kwargs, const GraphJitConfig &conf) {
  PyFrameObject *frame = Utils::PrepareFrame(callable.ptr(), args.ptr(), kwargs.ptr());
  if (frame == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  PyFrameWrapper ef(FrameConvert(frame));
  PyCodeObject *co = ef.GetCode().ptr();
  auto jcr = JitCompileResults::Create(co);
  jcr->set_conf(std::make_shared<GraphJitConfig>(conf));
  jcr->set_code(jcr->codehub()->AddOptTarget(OptOption::CreateOptionByPoint(jcr)));

  auto res = std::make_unique<GraphBuilder>(ef);

  auto code = res->GetGraph()->GetGuard();
  AddConfigToGuard(conf, code->GetGuard());
  AddGuardForParam(ef, code->GetGuard(), conf.GetBoolConfig(GraphJitConfig::kGuardDetachObject));

  Py_DECREF(frame);
  return res;
}

/**
 * build graph and infer func result
 * it used to infer mindspore function, maybe replace with mindspore func_graph to infer.
 */
AObject *InferFuncResult(const py::object &callable, const py::object &args, const py::object &kwargs,
                         const GraphJitConfig &conf, bool clear_guard) {
  auto g = GenerateRootGraph(callable, args, kwargs, conf);
  if (g == nullptr) {
    return nullptr;
  }
  g->TraceRun();
  if (conf.GetBoolConfig(GraphJitConfig::kPrintAfterAll)) {
    g->DumpDFG();
  }
  if (clear_guard) {
    Graph *graph = g->GetGraph();
    auto jcr = GetJitCompileResults(graph->GetCodeObj());
    jcr->codehub()->DelOptTarget(OptOption::CreateOptionByPoint(jcr), graph->GetGuard());
  }

  ValueNode *res = g->GetGraph()->GetRetVal();
  if (res == nullptr) {
    return nullptr;
  }
  return res->GetVobj();
}

AObject *InferFuncResult(const py::object &func, const std::vector<AObject *> &stack_args, int opcode,
                         const GraphJitConfig &conf, bool clear_guard) {
  std::vector<py::object> args;
  std::transform(stack_args.begin(), stack_args.end(), std::back_inserter(args),
                 [](AObject *i) { return i ? i->GetPyObject() : py::object(); });
  auto pair = Utils::PackCallStackArgs(args, opcode);
  if (pair.first.ptr() == nullptr) {
    return nullptr;
  }
  return InferFuncResult(func, pair.first, pair.second, conf, clear_guard);
}

AObject *InferFuncResult(const py::object &callable, const py::object &args, const py::object &kwargs,
                         const GraphJitConfig &conf) {
  return InferFuncResult(callable, args, kwargs, conf, true);
}

static bool GetGradSens(ValueNode *grad_node) {
  AObject *grad_object = grad_node->GetVobj();
  if (grad_object->GetPyObject().ptr() != nullptr) {
    return grad_object->GetAttr("sens_param")->GetPyObject().ptr() == Py_True;
  }
  bool sens_param = false;
  AObject *cls = grad_node->getInputs().size() > 0 ? grad_node->input(0)->GetVobj() : nullptr;
  if (!(Opcode(grad_node->GetOpcode()).IsCall() && cls != nullptr && cls->GetType() == AObject::kTypeType)) {
    return sens_param;
  }
  if (grad_node->GetOpcode() == CALL_FUNCTION && grad_node->getInputs().size() > 3) {
    AObject *tmp = grad_node->input(3)->GetVobj();
    sens_param = tmp ? tmp->GetPyObject().ptr() == Py_True : false;
  } else if (grad_node->GetOpcode() == CALL_FUNCTION_KW) {
    py::object kwnames = grad_node->getInputs().back()->GetVobj()->GetPyObject();
    PyObject **arr = &PyTuple_GET_ITEM(kwnames.ptr(), 0);
    Py_ssize_t size = PyTuple_GET_SIZE(kwnames.ptr());
    PyObject **iter = std::find_if(arr, arr + size, [](PyObject *k) {
      // find sens_param key
      return !PyUnicode_CompareWithASCIIString(k, "sens_param");
    });
    AObject *tmp = iter - arr != size ? grad_node->input(iter - arr)->GetVobj() : nullptr;
    sens_param = tmp ? tmp->GetPyObject().ptr() == Py_True : false;
  }
  return sens_param;
}

static void SetGradFuncInfo(CallNode *call_node) {
  const int flag = AObject::kMsFlagGradFunc | AObject::kMsFlagShardFunc | AObject::kMsFlagVmapFunc;
  ValueNode *grad_func_node = call_node->input(0);
  if (grad_func_node->getInputs().size() < 2) {
    grad_func_node->GetVobj()->ClearMsFlag(flag);
    return;
  }
  ValueNode *grad_node = grad_func_node->input(0);
  ValueNode *deco_func_node = grad_func_node->input(1);
  AObject *grad_object = grad_node->GetVobj();
  AObject *deco_func = deco_func_node->GetVobj();
  bool sens_param = false;
  if (grad_func_node->GetVobj()->TestMsFlag(AObject::kMsFlagGradFunc) &&
      grad_object->GetType() == AObject::kTypeMetaFuncGraph) {
    sens_param = GetGradSens(grad_node);
  }

  HandleGradFuncCall(call_node, deco_func, sens_param);

  // guard forward net for grad
  if (grad_func_node->GetVobj()->TestMsFlag(flag) && !call_node->GetGraph()->GuardValueNode(deco_func_node)) {
    grad_func_node->GetVobj()->ClearMsFlag(flag);
  }
}

void GraphBuilder::DumpDFG() { GRAPH_JIT_LOG_F("%s", graph_->ToString().c_str()); }

void MindGraphBuilder::AddInput(ValueNode *node) {
  auto obj = node->GetVobj()->GetPyObject();
  // tuple list is expand, this branch always false
  if (FGBuilder()->IsParameterSequence(obj)) {
    MS_LOG(WARNING) << "Get Parameter as function inputs, recompile if it's id changed";
    graph_->GuardValueNode(node, GuardLevel::GDeduce);
    return;
  }
  auto ret = FGBuilder()->AddTopGraphArgInput(obj);
  if (ret == nullptr) {
    if (!IsParameterObject(obj)) {
      MS_LOG(INFO) << "The object can't be a runtime argument of FuncGraph, build value node: " << node->ToString();
      node->set_abstract_wrapper(FGBuilder()->AddLocalVariable(node->GetVobj()->GetPyObject()));
    }
    graph_->GuardValueNode(node, GuardLevel::GDeduce);
    return;
  }
  node->set_abstract_wrapper(ret);
  root()->GetGraph()->PrepareParameter(node);
}

namespace {
bool IsSelfRef(const py::handle &obj) {
  ReprRecursionScope scope(obj.ptr());
  if (scope.ReEnterOrError()) {
    return true;
  }
  if (py::isinstance<py::dict>(obj)) {
    auto dict = py::cast<py::dict>(obj);
    return std::any_of(dict.begin(), dict.end(),
                       [](const auto &item) { return IsSelfRef(item.first) || IsSelfRef(item.second); });
  }
  if (py::isinstance<py::tuple>(obj) || py::isinstance<py::list>(obj)) {
    return std::any_of(obj.begin(), obj.end(), [](const auto &item) { return IsSelfRef(item); });
  }
  return false;
}

bool IsGradOperation(const ValueNode *node) {
  if (node == nullptr) {
    return false;
  }
  auto vobj = node->GetVobj();
  if (vobj == nullptr) {
    return false;
  }
  auto type = vobj->GetTypeObject();
  return type != nullptr && IsGradOperationType<true>(type);
}
}  // namespace

void MindGraphBuilder::ExpandContainerParameters(ValueNode *node) {
  auto expand_list_tuple = [this](ValueNode *node, const py::object &obj) {
    int index = 0;
    std::for_each(obj.begin(), obj.end(), [this, &index, node](const auto &item) {
      auto index_node = NewValueNode(AObject::Convert(py::int_(index)), {LOAD_CONST, -1}, {});
      auto item_node = NewValueNode(AObject::Convert(item.ptr()), {BINARY_SUBSCR, 0}, {node, index_node});
      ExpandContainerParameters(item_node);
      index++;
    });
  };
  auto expand_dict = [this](ValueNode *node, const py::object &obj) {
    auto dict = py::cast<py::dict>(obj);
    std::for_each(dict.begin(), dict.end(), [this, node](const auto &item) {
      // handle key, key will be placed on the top of stack
      (void)DoLoadConst({LOAD_CONST, -1, py::cast<py::object>(item.first)});
      auto value = NewValueNode(AObject::Convert(item.second.ptr()), {BINARY_SUBSCR, 0}, {node, seek(0)});
      ExpandContainerParameters(value);
    });
  };
  MS_EXCEPTION_IF_NULL(node);
  auto vobj = node->GetVobj();
  MS_EXCEPTION_IF_NULL(vobj);
  auto obj = vobj->GetPyObject();
  auto is_dict = py::isinstance<py::dict>(obj);
  auto is_tuple = py::isinstance<py::tuple>(obj);
  auto is_list = py::isinstance<py::list>(obj);
  if ((!is_list && !is_tuple && !is_dict) || IsSelfRef(obj)) {
    AddInput(node);
    push(node);
  } else if (is_dict) {
    expand_dict(node, obj);
    DoBuildOp({BUILD_MAP, SizeToInt(py::len(obj))});
  } else {
    expand_list_tuple(node, obj);
    DoBuildOp({(is_tuple ? BUILD_TUPLE : BUILD_LIST), SizeToInt(py::len(obj))});
  }
}

void MindGraphBuilder::FGAddTopInputsWithExpander() {
  bool has_vargs = false;
  bool has_kwargs = false;
  int args_count = PyCodeWrapper(GetGraph()->GetCodeObj()).ArgCount(&has_vargs, &has_kwargs);
  const auto &locals = frame_.GetLocals();
  MS_EXCEPTION_IF_CHECK_FAIL(args_count <= SizeToInt(locals.size()), "Locals size check failed");
  const auto &closure = frame_.GetClosures();
  for (size_t index = 0; index < closure.size(); ++index) {
    auto value = closure[index]->GetValue();
    if (value == nullptr || value == &ValueNode::kUnboundLocal) {
      continue;
    }
    ExpandContainerParameters(value); /* LOAD_DEREF */
    closure[index]->SetValue(pop());
  }
  auto is_grad_op = args_count > 0 ? IsGradOperation(locals[0]) : false;
  for (int index = 0; index < args_count; ++index) {
    if (locals[index] == nullptr || locals[index] == &ValueNode::kUnboundLocal) {
      continue;
    }
    if (is_grad_op) {
      AddInput(locals[index]);
    } else {
      ExpandContainerParameters(locals[index]);
      locals[index]->set_abstract_wrapper(pop()->abstract_wrapper());
    }
  }
}

void MindGraphBuilder::FGAddTopInputs() {
  if (graph_->Config().GetBoolConfig(GraphJitConfig::kExpandGraphInput)) {
    FGAddTopInputsWithExpander();
  } else {
    auto add_var_input = [this](ValueNode *cur, bool is_var_keywords) {
      if (cur == &ValueNode::kUnboundLocal) {
        return; /* LOAD_DEREF */
      }
      auto cur_object = cur->GetVobj()->GetPyObject();
      auto ret = is_var_keywords ? FGBuilder()->AddTopGraphKwargsInputs(cur_object)
                                 : FGBuilder()->AddTopGraphVargsInputs(cur_object);
      if (ret == nullptr) {
        return;
      }
      cur->set_abstract_wrapper(ret);
      root()->GetGraph()->PrepareParameter(cur);
    };
    bool has_vargs;
    bool has_kwargs;
    int args_count = PyCodeWrapper(GetGraph()->GetCodeObj()).ArgCount(&has_vargs, &has_kwargs);
    const auto &locals = frame_.GetLocals();
    MS_EXCEPTION_IF_CHECK_FAIL(args_count <= SizeToInt(locals.size()), "Locals size check failed");
    args_count = args_count - has_vargs - has_kwargs;

    for (const auto &node : frame_.GetClosures()) {
      auto cur = node->GetValue();
      if (cur != nullptr) {
        AddInput(cur); /* LOAD_DEREF */
      }
    }
    int cur_index = 0;
    for (cur_index = 0; cur_index < args_count; ++cur_index) {
      auto cur = locals[cur_index];
      AddInput(cur);
    }
    if (has_vargs) {
      auto cur = locals[cur_index];
      add_var_input(cur, false);
      cur_index++;
    }
    if (has_kwargs) {
      auto cur = locals[cur_index];
      add_var_input(cur, true);
    }
  }
}

bool MindGraphBuilder::FGAddInputs(const std::vector<ValueNode *> &args) {
  // Add function graph inputs.
  const auto &args_wrapper = HandleInputArgs(args);
  MS_EXCEPTION_IF_CHECK_FAIL(args_wrapper.size() == args.size(), "args size check failed.");
  for (size_t i = 0; i < args_wrapper.size(); ++i) {
    auto ret_abstract_wrapper = FGBuilder()->AddSubGraphInput(args_wrapper[i]);
    if (ret_abstract_wrapper == nullptr) {
      MS_LOG(INFO) << "Add input fail for input: " << args[i]->ToString();
      return false;
    }
    args[i]->set_abstract_wrapper(ret_abstract_wrapper);
    MS_LOG(INFO) << "Add input success for input: " << args[i]->ToString();
  }
  return true;
}

bool MindGraphBuilder::FGAddOutput() {
  if (GetGraph()->GetRetVal() == nullptr) {
    MS_LOG(INFO) << "Add output failed, graph ret value is null";
    return false;
  }
  ValueNode *ret = GetGraph()->GetRetVal();
  if (FGBuilder()->AddOutput(ret->abstract_wrapper(), false)) {
    MS_LOG(DEBUG) << "Add output success for value node: " << ret->ToString();
  } else {
    MS_LOG(INFO) << "Add output fail for value node: " << ret->ToString();
    return false;
  }
  bool succ = FGAddSideEffectOutput();
  if (succ) {
    MS_LOG(DEBUG) << "Add graph output success, total outputs num: " << FGBuilder()->GetOutputSize()
                  << ", side effect num: " << side_effect_outputs_.size();
  } else {
    MS_LOG(INFO) << "Add graph output failed, total outputs num: " << FGBuilder()->GetOutputSize()
                 << ", side effect num: " << side_effect_outputs_.size();
  }
  return succ;
}

bool MindGraphBuilder::FGAddSideEffectOutput() {
  for (ValueNode *node : side_effect_outputs_) {
    auto stop_gradient_node = FGBuilder()->AddNode(prim::kPrimStopGradient, {node->abstract_wrapper()});
    MS_EXCEPTION_IF_NULL(stop_gradient_node);
    node->set_abstract_wrapper(stop_gradient_node);
    bool succ = FGBuilder()->AddOutput(node->abstract_wrapper(), false);
    if (succ) {
      MS_LOG(DEBUG) << "Add side effect output success: " << node->ToString();
    } else {
      MS_LOG(INFO) << "Add side effect output failed: " << node->ToString();
      return false;
    }
  }
  return true;
}

void MindGraphBuilder::FGAddNode(CallNode *call_node, const py::object &callable_info,
                                 const AbstractWrapperPtrList &args, StopTraceReason *stop_reason) {
  MS_LOG(INFO) << "Try add node: " << py::str(callable_info);
  AbstractWrapperPtr res;
  if (call_node->GetOpcode() == CALL_FUNCTION_KW) {
    res = FGBuilder()->AddNodeCallFunctionKw(callable_info, args);
  } else if (call_node->GetOpcode() == CALL_FUNCTION_EX) {
    res = FGBuilder()->AddNodeCallFunctionEx(callable_info, args);
  } else {
    res = FGBuilder()->AddNode(callable_info, args);
  }
  UpdateNodeInfo(res, call_node, stop_reason);
}

void MindGraphBuilder::FGAddNode(CallNode *call_node, const ValuePtr &callable_value,
                                 const AbstractWrapperPtrList &args, StopTraceReason *stop_reason) {
  MS_LOG(INFO) << "Try add node: " << callable_value->ToString();
  AbstractWrapperPtr res;
  if (call_node->GetOpcode() == CALL_FUNCTION_KW) {
    res = FGBuilder()->AddNodeCallFunctionKw(callable_value, args);
  } else if (call_node->GetOpcode() == CALL_FUNCTION_EX) {
    res = FGBuilder()->AddNodeCallFunctionEx(callable_value, args);
  } else {
    res = FGBuilder()->AddNode(callable_value, args);
  }
  UpdateNodeInfo(res, call_node, stop_reason);
}

std::vector<ValueNode *> MindGraphBuilder::GetNewArgs(CallNode *call_node, AObject *vobj,
                                                      const GraphBuilderPtr &subgraph) {
  std::vector<ValueNode *> new_arg_value_nodes;
  vobj = (vobj && vobj->GetType() != AObject::kTypePrimitive) ? vobj : call_node->input(0)->GetVobj();
  if (vobj->GetType() == AObject::kTypeCFunction) {
    MS_LOG(INFO) << "not support cfunction";
  }
  auto new_callable_info = FindPyFunc(vobj);
  FrameStates f;
  if (subgraph == nullptr) {
    ResolveClosure(new_callable_info, call_node, &f);
    if (!HandleCallParameters(new_callable_info, call_node, &f)) {
      MS_LOG(INFO) << "HandleCallParameters error" << std::endl;
    }
  } else {
    f = subgraph->frame();
  }
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(new_callable_info.ptr()));
  int argc = co->co_argcount + co->co_kwonlyargcount;
  argc += (co->co_flags & CO_VARARGS) ? 1 : 0;
  argc += (co->co_flags & CO_VARKEYWORDS) ? 1 : 0;
  for (auto it = f.GetLocals().begin(); it != f.GetLocals().begin() + argc; it++) {
    std::set<AObject::Type> unsupported_parameter = {
      AObject::kTypeAnyValue,  AObject::kTypeFunction,      AObject::kTypeBoundMethod,
      AObject::kTypePrimitive, AObject::kTypeMetaFuncGraph, AObject::kTypeCell,
    };
    auto it_vobj = (*it)->GetVobj();
    if (it_vobj != nullptr) {
      auto pyobj = it_vobj->GetPyObject();
      if (pyobj.ptr() != nullptr) {
        if (unsupported_parameter.find(AbstractObjectBase::GetPyType(pyobj.ptr())) == unsupported_parameter.end()) {
          new_arg_value_nodes.push_back(*it);
        }
      }
    }
  }
  return new_arg_value_nodes;
}

std::pair<bool, std::vector<py::object>> MindGraphBuilder::GetConstantInputsObject(CallNode *call_node) {
  AObject *callable = call_node->input(0)->GetVobj();
  auto callable_info = callable->GetPyObject();
  if (callable_info.ptr() == nullptr) {
    return std::pair<bool, std::vector<py::object>>(false, {});
  }
  std::vector<ValueNode *> args_value_node;
  if (PyFunction_Check(callable_info.ptr())) {
    args_value_node = GetNewArgs(call_node);
  } else {
    const auto &call_node_inputs = call_node->getInputs();
    (void)std::copy(call_node_inputs.begin() + 1, call_node_inputs.end(), std::back_inserter(args_value_node));
  }

  std::vector<py::object> input_objects;
  for (auto arg_value_node : args_value_node) {
    py::object arg_py_object;
    if (arg_value_node->has_abstract_wrapper()) {
      auto arg_abstract_wrapper = arg_value_node->abstract_wrapper();
      if (!arg_abstract_wrapper->IsConstant()) {
        return std::pair<bool, std::vector<py::object>>(false, {});
      }
      arg_py_object = AbstractWrapper::ConvertToPyObject(arg_abstract_wrapper);
    } else {
      arg_py_object = arg_value_node->GetVobj()->GetPyObject();
    }
    if (arg_py_object.ptr() == nullptr) {
      return std::pair<bool, std::vector<py::object>>(false, {});
    }
    input_objects.push_back(arg_py_object);
  }
  return std::pair<bool, std::vector<py::object>>(true, input_objects);
}

BindArgumentsHelper<ValueNode *> MindGraphBuilder::PackInputsForFunc(const py::object &obj, int op_code,
                                                                     const std::vector<ValueNode *> &inputs,
                                                                     ValueNode *self_node, bool eliminate_sens) {
  auto func_info = obj;
  func_info = FindPyFunc(AObject::Convert(func_info));
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func_info.ptr()));
  BindArgumentsHelper<ValueNode *> bind_helper(co);

  bool pack_success = true;
  auto cast_keys = [](ValueNode *node) {
    MS_EXCEPTION_IF_CHECK_FAIL(node->IsConstantValue(), "'CALL_FUNCTION_KW' has error stack");
    return node->GetVobj()->GetPyObject().cast<std::vector<std::string>>();
  };
  auto cast_seq = [this, &pack_success](ValueNode *node) {
    size_t size = this->frame_.GetStacks().size();
    if (!pack_success || !this->UnpackElements(node)) {
      MS_LOG(ERROR) << "Unpack failed for argument [" << node->ToString() << "]";
      pack_success = false;
      return std::vector<ValueNode *>();
    }
    size = this->frame_.GetStacks().size() - size;
    std::vector<ValueNode *> res = {frame_.GetStacks().end() - size, frame_.GetStacks().end()};
    popn(size);
    pack_success = true;
    return res;
  };
  auto cast_map = [this, &pack_success](ValueNode *node) {
    size_t size = this->frame_.GetStacks().size();
    if (!pack_success || !this->UnpackDict(node)) {
      MS_LOG(ERROR) << "Unpack failed for argument [" << node->ToString() << "]";
      pack_success = false;
      return std::map<std::string, ValueNode *>();
    }
    size = this->frame_.GetStacks().size() - size;
    std::map<std::string, ValueNode *> res;
    for (; size > 0; size -= kTwo) {
      ValueNode *v_node = pop();
      ValueNode *k_node = pop();
      PyObject *k = k_node->GetVobj() ? k_node->GetVobj()->GetPyObject().ptr() : nullptr;
      if (k == nullptr || !PyUnicode_Check(k)) {
        MS_LOG(ERROR) << "keyword must be string";
        pack_success = false;
        return std::map<std::string, ValueNode *>();
      }
      res[PyUnicode_AsUTF8(k)] = v_node;
    }
    return res;
  };
  PackCallStackHelper<ValueNode *> pack_helper(op_code);
  if (!pack_helper.Pack({inputs.begin() + 1, inputs.end()}, cast_keys, cast_seq, cast_map)) {
    MS_LOG(EXCEPTION) << "Pack helper pack failed.";
  }
  if (eliminate_sens) {
    auto &result = pack_helper.result();
    auto &kw = result.kw_;
    auto iter = kw.find("sens");
    if (iter != kw.end()) {
      kw.erase(iter);
    } else {
      auto &args = result.args_;
      args.pop_back();
    }
  }
  if (self_node != nullptr) {
    auto &args = pack_helper.result().args_;
    args.insert(args.begin(), self_node);
  }
#if !IS_PYTHON_3_11_PLUS
  if (!bind_helper.Bind(pack_helper.result().args_, pack_helper.result().kw_)) {
    MS_LOG(EXCEPTION) << "Bind helper bind args failed.";
  }
  PyObject *defaults = PyFunction_GET_DEFAULTS(func_info.ptr());
  PyObject *kw_defaults = PyFunction_GET_KW_DEFAULTS(func_info.ptr());
  auto convert = [this](PyObject *, PyObject *, PyObject *value) {
    DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(value)});
    return pop();
  };
  if (!bind_helper.ApplyDefault(defaults, kw_defaults, convert)) {
    MS_LOG(EXCEPTION) << "Bind helper apply default failed.";
  }
#endif
  return bind_helper;
}

std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>> MindGraphBuilder::BuildForwardGraph(CallNode *call_node) {
  auto grad_net_node = static_cast<CallNode *>(call_node->input(0));
  MS_EXCEPTION_IF_NULL(grad_net_node);
  constexpr size_t grad_operation_index = 0;
  constexpr size_t forward_node_index = 1;
  auto grad_operation_node = grad_net_node->input(grad_operation_index);
  auto grad_object = grad_operation_node->GetVobj()->GetPyObject();
  auto forward_node = grad_net_node->input(forward_node_index);
  bool has_sense = py::hasattr(grad_object, "sens_param") && (grad_object.attr("sens_param").ptr() == Py_True);
  auto func_info = forward_node->GetVobj()->GetPyObject();
  bool is_cell = py::isinstance<Cell>(func_info);
  MS_EXCEPTION_IF_NULL(func_info.ptr());

  auto self_node = is_cell ? forward_node : nullptr;
  BindArgumentsHelper<ValueNode *> bind_helper =
    PackInputsForFunc(func_info, call_node->GetOpcode(), call_node->getInputs(), self_node, has_sense);

  auto bind_arguments_result = bind_helper.results();
  const auto &bind_args = bind_arguments_result.args_;
  const auto &bind_vargs = bind_arguments_result.va_;
  const auto &bind_kwargs = bind_arguments_result.kw_va_;

  func_info = FindPyFunc(AObject::Convert(func_info));
  DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(func_info)});
  int arg_size = 0;
  for (auto arg : bind_args) {
    push(arg);
    arg_size = arg_size + 1;
  }
  for (auto varg : bind_vargs) {
    push(varg);
    arg_size = arg_size + 1;
  }

  if (!bind_kwargs.empty()) {
    // Use CALL_FUNCTION_KW to build forward node.
    MS_LOG(EXCEPTION) << "Do not handle kwargs yet.";
  } else {
    DoCall({CALL_FUNCTION, arg_size});
  }
  pop();
  auto forward_graph_builder = std::dynamic_pointer_cast<MindGraphBuilder>(this->sub_graph);
  if (forward_graph_builder == nullptr) {
    MS_LOG(INFO) << "Failed to get function graph builder for forward graph.";
    return std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>>(nullptr, bind_helper);
  }
  auto fg = forward_graph_builder->FGBuilder()->graph();
  if (fg == nullptr) {
    MS_LOG(INFO) << "Failed to get function graph builder for forward graph.";
    return std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>>(nullptr, bind_helper);
  }
  fg = BasicClone(fg);
  if (bind_vargs.size() != 0) {
    MS_LOG(INFO) << "Build call graph for forward graph.";
    std::vector<size_t> arg_len = {bind_args.size(), bind_vargs.size(), bind_kwargs.size()};
    auto outer_fg = FuncGraphBuilder::BuildCallForwardGraphForGrad(fg, arg_len, is_cell);
    return std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>>(outer_fg, bind_helper);
  }

  return std::pair<FuncGraphPtr, BindArgumentsHelper<ValueNode *>>(fg, bind_helper);
}

TracePtr CreateRegisterHookTrace(const TracePtr &trace) {
  auto obj = py::cast<py::object>(trace->GetObject());
  auto hooks = FuncGraphBuilder::GetRegisterHookList(obj);
  auto hook_str = std::string(py::str(py::object(hooks)));
  auto hook_trace = std::make_shared<CustomizedTrace>(
    Py_True,
    [trace, hooks](PTraceContext context) -> PyObject * {
      auto obj = py::cast<py::object>(trace->Retrieve(context));
      auto hook_list = FuncGraphBuilder::GetRegisterHookList(obj);
      if (hook_list.size() != hooks.size()) {
        return Py_False;
      }
      for (size_t index = 0; index < hooks.size(); index++) {
        if (hook_list[index].ptr() != hooks[index].ptr()) {
          return Py_False;
        }
      }
      return Py_True;
    },
    [trace, hook_str](bool simple) -> std::string {
      auto obj = py::cast<py::object>(trace->GetObject());
      if (!py::isinstance<tensor::BaseTensor>(obj)) {
        return "Hook can't be attached to non-tensor.";
      }
      auto tensor = py::cast<tensor::BaseTensorPtr>(obj);
      if (simple) {
        return "Guard backward hook fn on Tensor[" + std::string(tensor->id()) + "].";
      }
      auto hook_list = FuncGraphBuilder::GetRegisterHookList(obj);
      auto hook_list_str = std::string(py::str(hook_list));
      return "{Backward hook fn of Tensor[" + std::string(tensor->id()) + "] Now : " + hook_str +
             " Before : " + hook_list_str + "}(type:" + std::to_string(TraceType::Customized) + ")";
    });
  return std::move(hook_trace);
}

void GuardRegisterHook(ValueNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->abstract_wrapper());
  auto abs = node->abstract_wrapper()->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  bool has_hook = abs->has_user_data(kRegisterHookKey);
  if (!has_hook && !abs->isa<abstract::AbstractTuple>()) {
    return;
  }
  auto graph = node->GetGraph();
  MS_EXCEPTION_IF_NULL(graph);
  const auto &guard = graph->GetGuard()->GetGuard();
  auto trace = node->GetTrace();
  if (trace == nullptr) {
    trace = graph->TraceValueNode(node);
  }
  if (has_hook) {
    auto hook_trace = CreateRegisterHookTrace(trace);
    guard->GuardOn(hook_trace, mindspore::pijit::GuardLevel::GEqual, true);
  } else {
    auto tuple = py::cast<py::tuple>(trace->GetObject());
    auto strict = graph->Config().GetBoolConfig(GraphJitConfig::kStrictTrace);
    for (size_t index = 0; index < tuple.size(); index++) {
      if (!FuncGraphBuilder::HasRegisterHook(py::cast<py::object>(tuple[index]))) {
        continue;
      }
      auto index_trace = CreateOpTrace(py::int_(index).ptr(), LOAD_CONST, -1, {}, "", "", strict);
      auto param_trace = CreateOpTrace(tuple[index].ptr(), BINARY_SUBSCR, 2, {trace, index_trace}, "", "", strict);
      auto hook_trace = CreateRegisterHookTrace(trace);
      guard->GuardOn(hook_trace, mindspore::pijit::GuardLevel::GEqual, true);
    }
  }
}

AbstractWrapperPtrList MindGraphBuilder::HandleInputsForGrad(CallNode *call_node,
                                                             BindArgumentsHelper<ValueNode *> forward_inputs) {
  auto grad_net_node = static_cast<CallNode *>(call_node->input(0));
  MS_EXCEPTION_IF_NULL(grad_net_node);
  constexpr size_t grad_operation_index = 0;
  constexpr size_t forward_node_index = 1;
  constexpr size_t param_tuple_index = 2;
  auto grad_operation_node = grad_net_node->input(grad_operation_index);
  auto grad_object = grad_operation_node->GetVobj()->GetPyObject();
  auto forward_node = grad_net_node->input(forward_node_index);
  bool has_sense = py::hasattr(grad_object, "sens_param") && (grad_object.attr("sens_param").ptr() == Py_True);
  auto func_info = forward_node->GetVobj()->GetPyObject();
  MS_EXCEPTION_IF_NULL(func_info.ptr());

  const auto &bind_arguments_result = forward_inputs.results();
  const auto &bind_args = bind_arguments_result.args_;
  const auto &bind_vargs = bind_arguments_result.va_;
  const auto &bind_kwargs = bind_arguments_result.kw_va_;

  auto wrapper_args = HandleInputArgs(bind_args);
  const auto &wrapper_vargs = HandleInputArgs(bind_vargs);

  bool get_all = py::hasattr(grad_object, "get_all") && (grad_object.attr("get_all").ptr() == Py_True);
  bool get_by_list = py::hasattr(grad_object, "get_by_list") && (grad_object.attr("get_by_list").ptr() == Py_True);
  auto offset = py::isinstance<Cell>(func_info) ? 1 : 0;
  std::vector<ValueNode *> forward_input(bind_args.begin() + offset, bind_args.end());
  size_t input_grad_cnt = get_all ? forward_input.size() : (get_by_list ? 0 : (forward_input.empty() ? 0 : 1));

  for (size_t index = 0; index < input_grad_cnt; index++) {
    if (!forward_input[index]->has_abstract_wrapper() || forward_input[index]->GetVobj() == nullptr) {
      continue;
    }
    auto obj = forward_input[index]->GetVobj()->GetPyObject();
    if (py::isinstance<py::none>(obj)) {
      continue;
    }
    auto wrapper = forward_input[index]->abstract_wrapper();
    auto node = FGBuilder()->ReadLocalVariable(wrapper);
    FuncGraphBuilder::SaveTensorRegisterHook(obj, node);
    GuardRegisterHook(forward_input[index]);
  }
  if (get_by_list) {
    GuardRegisterHook(grad_net_node->input(param_tuple_index));
  }

  if (has_sense) {
    if (!bind_vargs.empty() || !bind_kwargs.empty()) {
      MS_LOG(EXCEPTION) << "Do not support sense param with vargs and kwargs yet.";
    }
    MS_EXCEPTION_IF_CHECK_FAIL(call_node->getInputs().size() > bind_args.size(), "Arg size check failed.");
    auto sens_value_node = call_node->getInputs().back();
    auto new_wrapper = sens_value_node->has_abstract_wrapper()
                         ? sens_value_node->abstract_wrapper()
                         : FGBuilder()->AddLocalVariable(sens_value_node->GetVobj()->GetPyObject());
    sens_value_node->set_abstract_wrapper(new_wrapper);
    wrapper_args.push_back(new_wrapper);
  }

  AbstractWrapperPtrList final_wrapper;
  // Eliminate self input for cell when building grad graph.
  bool input_offset = py::isinstance<Cell>(func_info) ? 1 : 0;
  (void)std::copy(wrapper_args.begin() + input_offset, wrapper_args.end(), std::back_inserter(final_wrapper));
  (void)std::copy(wrapper_vargs.begin(), wrapper_vargs.end(), std::back_inserter(final_wrapper));
  return final_wrapper;
}

AnfNodePtr CreateInsertGradientOfNode(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  auto hooks = node->abstract()->user_data<py::tuple>(kRegisterHookKey);
  MS_LOG(INFO) << "Apply " << py::str(py::object(*hooks)) << " to " << node->DebugString();
  auto ops_mod = python_adapter::GetPyModule("mindspore.ops.operations.debug_ops");
  auto op_class = python_adapter::GetPyObjAttr(ops_mod, "InsertGradientOf");
  auto insert_grad_of = node;
  for (const auto &hook : *hooks) {
    // Create class instance.
    auto params = py::make_tuple(hook);
    auto obj = parse::data_converter::CreatePythonObject(op_class, params);
    if (py::isinstance<py::none>(obj)) {
      MS_LOG(ERROR) << "Create python object `" << py::str(op_class)
                    << "` failed, only support to create 'Cell', 'Primitive' or "
                    << "user-defined Class decorated with 'jit_class'.";
      return nullptr;
    }
    ValuePtr converted_res = nullptr;
    bool converted = parse::ConvertData(obj, &converted_res, false);
    if (!converted) {
      MS_LOG(ERROR) << "Convert the python object failed";
      return nullptr;
    }
    MS_EXCEPTION_IF_NULL(converted_res);
    insert_grad_of = func_graph->NewCNode({NewValueNode(converted_res), insert_grad_of});
    insert_grad_of->set_abstract(node->abstract());
  }
  return insert_grad_of;
}

bool ApplyRegisterHook(const AnfNodePtr &node) {
  if (node->abstract() == nullptr || !node->abstract()->has_user_data(kRegisterHookKey)) {
    return true;
  }
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users = manager->node_users()[node];
  for (const auto &node_and_index : node_users) {
    if (IsPrimitiveCNode(node_and_index.first, prim::kPrimInsertGradientOf)) {
      continue;
    }
    auto insert_grad_of = CreateInsertGradientOfNode(node, node_and_index.first->func_graph());
    if (insert_grad_of == nullptr) {
      return false;
    }
    manager->SetEdge(node_and_index.first, node_and_index.second, insert_grad_of);
  }
  return true;
}

void HandleRegisterHook(const FuncGraphPtr &func_graph, StopTraceReason *stop_reason) {
  if (func_graph->manager() == nullptr) {
    auto mng = Manage(func_graph, true);
    func_graph->set_manager(mng);
  }
  AnfNodePtrList nodes(func_graph->parameters());
  auto vars = func_graph->free_variables();
  std::transform(vars.begin(), vars.end(), std::back_inserter(nodes), [](const auto &var) { return var.first; });
  if (!std::all_of(nodes.begin(), nodes.end(), [](const auto &node) { return ApplyRegisterHook(node); })) {
    *stop_reason = StopTraceReason::kStopTraceTensorHook;
  }
}

py::object MindGraphBuilder::ResolveGradCall(CallNode *call_node, StopTraceReason *stop_reason) {
  auto grad_net_node = static_cast<CallNode *>(call_node->input(0));
  if (grad_net_node == nullptr) {
    return py::object();
  }
  auto grad_net_wrapper = grad_net_node->abstract_wrapper();
  if (grad_net_wrapper == nullptr) {
    MS_LOG(ERROR) << "Fail to get abstract wrapper for grad net node: " << grad_net_node->ToString();
    return py::object();
  }
  constexpr size_t grad_operation_index = 0;
  constexpr size_t forward_net_index = 1;
  bool guard_grad_operation = graph_->GuardValueNode(grad_net_node->input(grad_operation_index), GId);
  if (!guard_grad_operation) {
    MS_LOG(WARNING) << "Guard GradOperation value node failed, value node: "
                    << grad_net_node->input(grad_operation_index)->ToString();
  }

  auto forward_net_object = grad_net_node->input(forward_net_index)->GetVobj()->GetPyObject();
  (void)pi_jit_should_compile(forward_net_object, py::dict(), py::none());

  bool guard_forward_net = graph_->GuardValueNode(grad_net_node->input(forward_net_index), GId);
  if (!guard_forward_net) {
    MS_LOG(WARNING) << "Guard forward net value node for GradOperation failed, value node: "
                    << grad_net_node->input(forward_net_index)->ToString();
  }

  auto forward_result = BuildForwardGraph(call_node);
  auto forward_fg = forward_result.first;
  if (forward_fg == nullptr) {
    MS_LOG(INFO) << "Build forward fg failed.";
    return py::object();
  }
  auto forward_graph_builder = std::dynamic_pointer_cast<MindGraphBuilder>(this->sub_graph);

  if (py::isinstance<Cell>(forward_net_object)) {
    HandleCustomBProp(forward_fg, forward_net_object);
  }

  const auto &inputs_wrapper = HandleInputsForGrad(call_node, forward_result.second);
  for (auto input_wrapper : inputs_wrapper) {
    if (input_wrapper == nullptr) {
      MS_LOG(EXCEPTION) << "Input wrapper is NULL, failed to build graph.";
    }
    MS_LOG(INFO) << "input wrapper is: " << input_wrapper->ToString();
  }
  auto ret = FGBuilder()->BuildGradNode(grad_net_wrapper, forward_fg, inputs_wrapper);
  MS_EXCEPTION_IF_NULL(ret);
  if (ret != nullptr) {
    call_node->SetVobj(AObject::Convert(ret));
    call_node->set_abstract_wrapper(ret);
    *stop_reason = StopTraceReason::kNonStopTrace;
  }
  HandleRegisterHook(forward_fg, stop_reason);
  HandleGradForwardSideEffect(forward_fg, ret, forward_graph_builder, call_node);
  return py::object();
}

void MindGraphBuilder::HandleGradForwardSideEffect(const FuncGraphPtr &forward_fg, const AbstractWrapperPtr &grad,
                                                   const MindGraphBuilderPtr &subgraph_builder, CallNode *call_node) {
  const auto &side_effect_outputs = subgraph_builder->side_effect_outputs();
  if (side_effect_outputs.empty() || !grad->grad_info().get_value_) {
    return;
  }
  // For value_and_grad with forward net side effect,
  // the output is format ((real_forward, side_effect_output), real_grad)
  // Need to adjust abstract wrapper for call node to (real_forward, real_grad) and adjust side_effect_outputs as well.
  MS_EXCEPTION_IF_NULL(forward_fg->output());
  auto grad_abstract = grad->abstract();
  MS_EXCEPTION_IF_NULL(grad_abstract);
  if (!grad_abstract->isa<abstract::AbstractTuple>()) {
    return;
  }

  AbstractWrapperPtr forward_idx = FGBuilder()->AddLocalVariable(py::int_(0));
  auto forward = FGBuilder()->AddNode(prim::kPrimTupleGetItem, {grad, forward_idx});
  MS_EXCEPTION_IF_NULL(forward);
  AbstractWrapperPtr grad_idx = FGBuilder()->AddLocalVariable(py::int_(1));
  auto real_grad = FGBuilder()->AddNode(prim::kPrimTupleGetItem, {grad, grad_idx});
  MS_EXCEPTION_IF_NULL(real_grad);
  AbstractWrapperPtr real_forward_idx = FGBuilder()->AddLocalVariable(py::int_(0));
  auto real_forward = FGBuilder()->AddNode(prim::kPrimTupleGetItem, {forward, real_forward_idx});
  MS_EXCEPTION_IF_NULL(real_forward);
  for (size_t i = 0; i < side_effect_outputs.size(); ++i) {
    AbstractWrapperPtr cur_idx = FGBuilder()->AddLocalVariable(py::int_(i + 1));
    auto cur_side_effect_output = FGBuilder()->AddNode(prim::kPrimTupleGetItem, {forward, cur_idx});
    MS_EXCEPTION_IF_NULL(cur_side_effect_output);
    side_effect_outputs[i]->set_abstract_wrapper(cur_side_effect_output);
  }
  auto real_ret = FGBuilder()->AddNode(prim::kPrimMakeTuple, {real_forward, real_grad});
  MS_EXCEPTION_IF_NULL(real_ret);
  call_node->SetVobj(AObject::Convert(real_ret));
  call_node->set_abstract_wrapper(real_ret);
}

bool MindGraphBuilder::IsGradCallable(ValueNode *node) {
  if (node == nullptr || !node->has_abstract_wrapper()) {
    return false;
  }
  auto abstract = node->abstract_wrapper()->abstract();
  if (abstract == nullptr) {
    return false;
  }
  auto value = abstract->BuildValue();
  if (!value->isa<StringImm>()) {
    return false;
  }
  const auto &call_str = value->cast<StringImmPtr>()->value();
  const std::string fake_grad_prefix = "FakeNodeKey MetaFuncGraph-grad";
  return call_str.substr(0, fake_grad_prefix.size()) == fake_grad_prefix;
}

py::object MindGraphBuilder::GetPyObject(ValueNode *node) {
  if (node == nullptr) {
    return py::object();
  }
  auto v_object = node->GetVobj();
  if (v_object == nullptr) {
    return py::object();
  }
  auto object = v_object->GetPyObject();
  if (object.ptr() != nullptr) {
    return object;
  }
  return AbstractWrapper::ConvertToPyObject(node->abstract_wrapper());
}

static bool MindFGForbiddenConvertFunc(const py::handle &func) {
  auto ptr = func.ptr();
  if (PyMethod_Check(ptr)) {
    ptr = PyMethod_GET_FUNCTION(ptr);
  }
  const auto &qualname_obj = py::getattr(func, "__qualname__", py::object());
  if (qualname_obj.ptr() == nullptr) {
    return false;
  }
  const auto &qualname = qualname_obj.cast<std::string>();
  // forbidden_list includes two kinds of operation:
  //   1. function with side effect such as list.__setitem__.
  //   2. function return iterator, such as enumerate.
  static std::vector<std::string> forbidden_list = {
    "Tensor.__setitem__",
    // tolist require tensor value
    "Tensor.tolist",
    "list.append",
    "list.pop",
    "list.insert",
    "list.clear",
    "list.reverse",
    "list.__setitem__",
    "dict.__setitem__",
    "dict.update",
    "dict.clear",
    "dict.pop",
    "dict.keys",
    "enumerate",
    "zip",
    "map",
    "filter",
    "__setitem__",
    "getattr",
    "range",
    "concat",
  };
  return std::any_of(forbidden_list.begin(), forbidden_list.end(),
                     [&qualname](const std::string &name) { return qualname == name; });
}

std::string GetClassTypeName(const py::object &obj) {
  if (!PyType_Check(obj.ptr())) {
    return "";
  }
  // desc has format "<class xxxx>", strip the '<' and '>' by offset 1.
  std::string tp_name = reinterpret_cast<PyTypeObject *>(obj.ptr())->tp_name;
  return "class '" + tp_name + "'";
}

py::object ConvertPythonBuiltInFunction(const py::object &obj, const std::string &func_name) {
  py::str func_name_obj = py::str(func_name);
  auto dict = python_adapter::GetPyObjAttr(python_adapter::GetPyModule("mindspore._extends.parse.resources"),
                                           "convert_class_to_function_map");
  auto callable_obj_ptr = PyDict_GetItem(dict.ptr(), func_name_obj.ptr());
  return callable_obj_ptr == nullptr ? py::object() : py::cast<py::object>(callable_obj_ptr);
}

bool MindGraphBuilder::ConvertClassType(const py::object &callable_info, CallNode *call_node,
                                        StopTraceReason *stop_reason) {
  // Iterator is not currently supported and will be considered later.
  if (PyIter_Check(callable_info.ptr())) {
    return false;
  }
  const auto &class_type_name = GetClassTypeName(callable_info);
  if (class_type_name == "") {
    return false;
  }
  static std::map<std::string, ValuePtr> list_or_tuple_func_map = {
    {"class 'list'", std::make_shared<prim::ListFunc>("list_func")},
    {"class 'tuple'", std::make_shared<prim::TupleFunc>("tuple_func")},
    {"class 'dict'", std::make_shared<prim::DictFunc>("dict_func")}};
  auto iter = list_or_tuple_func_map.find(class_type_name);
  if (iter != list_or_tuple_func_map.end()) {
    auto callable_value = iter->second;
    MS_LOG(INFO) << "Found built-in class type: " << class_type_name;
    std::vector<ValueNode *> args;
    const auto &call_node_inputs = call_node->getInputs();
    (void)std::copy(call_node_inputs.begin() + 1, call_node_inputs.end(), std::back_inserter(args));
    FGAddNode(call_node, callable_value, HandleInputArgs(args), stop_reason);
    return true;
  }
  auto py_builtin_func = ConvertPythonBuiltInFunction(callable_info, class_type_name);
  if (py_builtin_func.ptr() != nullptr) {
    MS_LOG(INFO) << "Convert python built-in function:" << py::str(callable_info) << " to " << py::str(py_builtin_func);
    std::vector<ValueNode *> args;
    const auto &call_node_inputs = call_node->getInputs();
    (void)std::copy(call_node_inputs.begin() + 1, call_node_inputs.end(), std::back_inserter(args));
    FGAddNode(call_node, py_builtin_func, HandleInputArgs(args), stop_reason);
    return true;
  }
  return false;
}

std::pair<bool, py::object> MindGraphBuilder::ConvertCallableObject(const py::object &callable_info) const {
  bool should_parse_in_ast = false;
  if (MindFGForbiddenConvertFunc(callable_info)) {
    return std::make_pair(should_parse_in_ast, callable_info);
  }
  auto method = FGBuilder()->ConvertMethod(callable_info);
  if (method.ptr() != nullptr) {
    MS_LOG(INFO) << "convert method :" << py::str(callable_info) << " to " << py::str(method);
    should_parse_in_ast = PyMethod_Check(method.ptr()) || PyFunction_Check(method.ptr());
    return std::make_pair(should_parse_in_ast, method);
  }
  auto func = FGBuilder()->ConvertFunction(callable_info);
  if (func.ptr() != nullptr) {
    MS_LOG(INFO) << "convert function:" << py::str(callable_info) << " to " << py::str(func);
    should_parse_in_ast = PyMethod_Check(func.ptr()) || PyFunction_Check(func.ptr());
    return std::make_pair(should_parse_in_ast, func);
  }
  if (PyMethod_Check(callable_info.ptr()) || PyFunction_Check(callable_info.ptr())) {
    const auto &module_name = GetModuleName(callable_info);
    bool match = std::any_of(kAstFunctionList.begin(), kAstFunctionList.end(), [&module_name](const std::string &name) {
      return module_name.substr(0, name.size()) == name;
    });
    if (match) {
      MS_LOG(INFO) << "Found object " << py::str(callable_info) << " with module name " << module_name
                   << "should be parsed in ast.";
    }
    should_parse_in_ast = match;
  }

  if (IsPSJitFunction(callable_info)) {
    MS_LOG(INFO) << "Callable object " << py::str(callable_info) << " is decorated by PSJit, parse with ast.";
    should_parse_in_ast = true;
  }

  return std::make_pair(should_parse_in_ast, callable_info);
}

void MindGraphBuilder::FGAddNodeWithAst(CallNode *call_node, const py::object &callable_info,
                                        const std::vector<ValueNode *> &args, StopTraceReason *stop_reason) {
  const auto &callable_object =
    py::isinstance<mindspore::Cell>(callable_info)
      ? py::reinterpret_steal<py::object>(PyObject_GetAttrString(callable_info.ptr(), "construct"))
      : callable_info;
  FGAddNode(call_node, callable_object, HandleInputArgs(args), stop_reason);
}

// fix cyclomatic complexity
py::object MindGraphBuilder::HandleMSCallable(CallNode *call_node, const py::object &callable_info,
                                              const py::object &original_callable, StopTraceReason *stop_reason) {
  std::vector<ValueNode *> args;
  if (PyFunction_Check(callable_info.ptr())) {
    args = GetNewArgs(call_node);
  } else if (callable_info.ptr() != original_callable.ptr() && py::hasattr(callable_info, PYTHON_PRIMITIVE_FLAG) &&
             (PyMethod_Check(original_callable.ptr()) || PyCFunction_Check(original_callable.ptr()))) {
    // When x.y maps to primitive, x should be added to the first input of the primitive.
    auto self_node = GetBoundSelf(call_node);
    if (self_node != nullptr) {
      args.insert(args.begin(), self_node);
    }
    const auto &call_node_inputs = call_node->getInputs();
    (void)std::copy(call_node_inputs.begin() + 1, call_node_inputs.end(), std::back_inserter(args));
  } else {
    const auto &call_node_inputs = call_node->getInputs();
    (void)std::copy(call_node_inputs.begin() + 1, call_node_inputs.end(), std::back_inserter(args));
  }
  FGAddNode(call_node, callable_info, HandleInputArgs(args), stop_reason);
  return py::object();
}

static void MarkPIJitSpecializedCall(const FuncGraphBuilderPtr &fg_builder, CallNode *call_node) {
  if (!call_node->has_abstract_wrapper() && call_node->GetVobj() && call_node->GetVobj()->GetPyObject().ptr()) {
    MS_LOG(INFO) << "specialized call not set abstract wrapper, it's constant call. [" << call_node->ToString();
    call_node->set_abstract_wrapper(fg_builder->AddLocalVariable(call_node->GetVobj()->GetPyObject()));
    if (call_node->GetSubGraph() && call_node->GetSubGraph()->GetRetVal()) {
      call_node->GetSubGraph()->GetRetVal()->set_abstract_wrapper(call_node->abstract_wrapper());
    }
  }
}

py::object MindGraphBuilder::ResolveCallable(CallNode *call_node, StopTraceReason *stop_reason) {
  py::object callable_info = GetPyObject(call_node->input(0));
  *stop_reason = StopTraceReason::kStopTraceInfer_Fail;
  if (callable_info.ptr() == nullptr) {
    return callable_info;
  }
  py::object original_callable = callable_info;
  if (IsGradCallable(call_node->input(0))) {
    return ResolveGradCall(call_node, stop_reason);
  }
  if (!FGBuilder()->ValidateCallableObject(callable_info)) {
    return py::object();
  }
  if (FGBuilder()->CanConstantFoldFunc(callable_info)) {
    const auto &res = GetConstantInputsObject(call_node);
    if (res.first) {
      return HandleConstantFoldFunc(res.second, call_node, stop_reason);
    }
  }
  if (ConvertClassType(callable_info, call_node, stop_reason)) {
    return py::object();
  }

  auto convert_result = ConvertCallableObject(callable_info);
  callable_info = convert_result.second;
  bool should_parse_in_ast = convert_result.first;

  if (should_parse_in_ast) {
    MS_LOG(INFO) << "Should be parsed by ast for object: " << py::str(callable_info);
    std::vector<ValueNode *> args;
    auto self_node = GetBoundSelf(call_node);
    if (callable_info.ptr() != original_callable.ptr() && self_node != nullptr) {
      args.push_back(self_node);
    }
    const auto &call_node_inputs = call_node->getInputs();
    (void)std::copy(call_node_inputs.begin() + 1, call_node_inputs.end(), std::back_inserter(args));
    FGAddNodeWithAst(call_node, callable_info, args, stop_reason);
    if (!call_node->has_abstract_wrapper()) {
      MS_LOG(INFO) << "Build graph by ast failed for node: " << call_node->ToString();
    }
    return py::object();
  }
  if (FGBuilder()->CheckCallable(callable_info)) {
    return HandleMSCallable(call_node, callable_info, original_callable, stop_reason);
  }

  py::object result = this->GraphBuilder::ResolveCallable(call_node, stop_reason);
  AObject *callable = call_node->input(0)->GetVobj();
  bool pijit_specialized = original_callable.ptr() == callable_info.ptr()  // not converted
                           || call_node->GetSubGraph() != nullptr          // pijit sub graph
                           || callable->GetType() == AObject::kTypeType;   // pijit class instantiation
  if (pijit_specialized) {
    MarkPIJitSpecializedCall(FGBuilder(), call_node);
    return result;
  }
  MS_LOG(DEBUG) << "convert " << std::string(py::str(original_callable)) << " -> "
                << std::string(py::str(callable_info));
  return FindPyFunc(AObject::Convert(callable_info));
}

py::object MindGraphBuilder::HandleConstantFoldFunc(const std::vector<py::object> &args, CallNode *call_node,
                                                    StopTraceReason *stop_reason) {
  py::object callable_info = GetPyObject(call_node->input(0));
  MS_LOG(INFO) << "CanConstantFoldFunc for: " << call_node->ToString() << ", " << py::str(callable_info);

  JustCallAndSetResWithArgs(call_node, args);

  py::object result = call_node->GetVobj()->GetPyObject();
  if (result.ptr() != nullptr) {
    const AbstractWrapperPtr &abs_wrapper = FGBuilder()->AddLocalVariable(result);
    call_node->set_abstract_wrapper(abs_wrapper);
    *stop_reason = StopTraceReason::kNonStopTrace;
  } else {
    MS_LOG(INFO) << "Constant fold failed, result is null";
    *stop_reason = StopTraceReason::kStopTraceInfer_Fail;
  }
  return py::object();
}

namespace {
// Check if it is a scenario of creating namedtuple.
bool IsMakeNamedtuple(const CallNode *call_node);
// Check if it is a scenario of getting element from namedtuple.
bool IsNamedtupleGetElem(const ValueNode *node, const std::string &name);
// Create AObject for namedtuple.
AbstractNamedTuple *MakeNamedtupleAObj(const CallNode *call_node);
// Convert abstract::AbstractTuple to abstract::AbstractNamedTuple.
abstract::AbstractNamedTuplePtr ConvertToAbstractNamedTuple(const AbstractBasePtr &abstract,
                                                            const AbstractNamedTuple *namedtuple_aobj);
}  // namespace

constexpr const char *TensorCastPrim(AObject::Type type) {
  constexpr std::pair<AObject::Type, const char *> prim_map[] = {
    {AObject::kTypeList, "list_to_tensor_"},   {AObject::kTypeTuple, "tuple_to_tensor_"},
    {AObject::kTypeInt, "scalar_to_tensor_"},  {AObject::kTypeFloat, "scalar_to_tensor_"},
    {AObject::kTypeBool, "scalar_to_tensor_"},
  };
  constexpr size_t size = sizeof(prim_map) / sizeof(prim_map[0]);
  for (size_t i = size; i > 0; --i) {
    if (type == prim_map[i - 1].first) {
      return prim_map[i - 1].second;
    }
  }
  return nullptr;
}

py::object TensorDTypeObjectByData(ValueNode *data_node) {
  py::object dtype_object = py::none();
  if (data_node->abstract_wrapper() != nullptr) {
    // fast path. not implement;
    MS_LOG(INFO) << "input data abstract is " << data_node->abstract_wrapper()->ToString();
  }
  // how to fast get data type ?
  auto data = data_node->GetVobj()->GetPyObject();
  if (data.ptr() == nullptr) {
    MS_LOG(INFO) << "data is nullptr";
    return dtype_object;
  }
  dtype_object = py::module::import("mindspore").attr("Tensor").attr("_set_default_dtype")(data, dtype_object);
  if (dtype_object.ptr() == Py_None) {
    // NOTE: list[bool] type is return None, check it
    MS_LOG(INFO) << "dtype_object is None";
  }
  return dtype_object;
}

bool GraphBuilder::HandleCallTensorClass(CallNode *call_node) {
  if (call_node->GetOpcode() != CALL_FUNCTION || call_node->GetOparg() == 0) {
    return false;  // maybe constant
  }
  ValueNode *data_node = call_node->input(1);
  AObject::Type input_type = data_node->GetVobj()->GetType();
  if (input_type == AObject::kTypeNone) {
    return false;  // tensor initialization without input_data, maybe constant
  }
  ValueNode *dtype_node = call_node->getInputs().size() > 2 ? call_node->input(2) : nullptr;
  py::object dtype_object = dtype_node == nullptr ? py::none() : dtype_node->GetVobj()->GetPyObject();
  bool is_default_dtype = dtype_object.ptr() == Py_None;
  MS_LOG(INFO) << "handle call tensor, input type is " << AObject::GetTypeDesc(input_type);
  if (input_type == AObject::kTypeTensor) {
    if (is_default_dtype) {
      dtype_object = data_node->GetVobj()->GetPyObject().attr("dtype");
    }
    MS_EXCEPTION_IF_CHECK_FAIL(dtype_object.ptr() != Py_None, "unknown dtype");
    ValueNode *res = MakePrimCastNode(data_node, dtype_object.ptr());
    call_node->SetVobj(res->GetVobj());
    call_node->set_abstract_wrapper(res->abstract_wrapper());
    return true;
  }
  const char *prim = TensorCastPrim(input_type);
  if (prim == nullptr) {
    return false;
  }
  if (is_default_dtype) {
    dtype_object = TensorDTypeObjectByData(data_node);
    if (dtype_object.ptr() == Py_None) {
      return false;
    }
    DoLoadConst({LOAD_CONST, 0, dtype_object});
    dtype_node = pop();
  }
  py::object prim_object = py::module::import("mindspore.ops.function.nn_func").attr(prim);
  DoLoadConst({LOAD_CONST, 0, prim_object});
  push(data_node);
  push(dtype_node);
  DoCall({CALL_FUNCTION, 2});
  ValueNode *res = pop();
  call_node->SetVobj(res->GetVobj());
  call_node->set_abstract_wrapper(res->abstract_wrapper());
  return true;
}

ValueNode *MindGraphBuilder::HandleCallClass(CallNode *call_node) {
  if (IsMakeNamedtuple(call_node)) {
    return HandleMakeNamedtuple(call_node);
  }
  ValueNode *node = GraphBuilder::HandleCallClass(call_node);
  if (node == nullptr) {
    MS_LOG(INFO) << "Failed to handle call class";
    return nullptr;
  }
  if (node->has_abstract_wrapper()) {
    return node;
  }
  if (node->GetVobj() != nullptr && node->GetVobj()->GetPyObject().ptr() != nullptr) {
    MS_LOG(INFO) << "The class instantiation is constant. [" << call_node->ToString();
    auto abs_wrapper = FGBuilder()->AddLocalVariable(node->GetVobj()->GetPyObject());
    if (abs_wrapper != nullptr) {
      node->set_abstract_wrapper(abs_wrapper);
    }
  }
  if (!node->has_abstract_wrapper()) {
    MS_LOG(INFO) << "Failed to handle call class, failed to create abstract for node: " << node->ToString();
    return nullptr;
  }
  return node;
}

namespace {
bool IsMakeNamedtuple(const CallNode *call_node) {
  AObject *callable = call_node->input(0)->GetVobj();
  if (callable == nullptr || callable->GetType() != AObject::kTypeType) {
    return false;
  }
  auto *abstract_type = static_cast<AbstractType *>(callable);
  return abstract_type->GetTypeType() == AObject::kTypeNamedTuple;
}
}  // namespace

// Do create namedtuple logic. Return node if success, else nullptr.
ValueNode *MindGraphBuilder::HandleMakeNamedtuple(CallNode *call_node) {
  MS_LOG(DEBUG) << "Start make namedtuple";
  AbstractNamedTuple *namedtuple_aobj = MakeNamedtupleAObj(call_node);
  if (namedtuple_aobj == nullptr) {
    return nullptr;
  }
  AbstractWrapperPtr abs_wrapper = MakeNamedtupleInGraph(call_node, namedtuple_aobj);
  if (abs_wrapper == nullptr || abs_wrapper->abstract() == nullptr) {
    return nullptr;
  }
  call_node->set_abstract_wrapper(abs_wrapper);
  call_node->SetVobj(namedtuple_aobj);
  return call_node;
}

namespace {
AbstractNamedTuple *MakeNamedtupleAObj(const CallNode *call_node) {
  AObject *callable = call_node->input(0)->GetVobj();
  MS_EXCEPTION_IF_NULL(callable);
  auto *abstract_type = static_cast<AbstractType *>(callable);
  // 1.create namedtuple python object
  const auto &params = call_node->getInputs();
  if (std::any_of(params.begin() + 1, params.end(), [](auto *node) { return !HasPyObj(node); })) {
    MS_LOG(INFO) << "Create namedtuple failed, has null argument";
    return nullptr;
  }
  std::vector<py::object> args;
  std::transform(params.begin() + 1, params.end(), std::back_inserter(args),
                 [](ValueNode *n) { return n->GetVobj()->GetPyObject(); });
  py::object namedtuple = abstract_type->BuildInstance(args, call_node->GetOpcode());
  if (namedtuple.ptr() == nullptr) {
    MS_LOG(INFO) << "Create namedtuple python object failed";
    return nullptr;
  }
  // 2.create AbstractNamedTuple
  AObject *aobj = AObject::Convert(namedtuple);
  if (aobj == nullptr || aobj->GetType() != AObject::kTypeNamedTuple) {
    MS_LOG(INFO) << "Create AbstractNamedTuple AObject failed";
    return nullptr;
  }
  return static_cast<AbstractNamedTuple *>(aobj);
}
}  // namespace

AbstractWrapperPtr MindGraphBuilder::MakeNamedtupleInGraph(const CallNode *call_node,
                                                           const AbstractNamedTuple *namedtuple_aobj) {
  // 1.Collect all the tuple elements.
  std::vector<AbstractWrapperPtr> elems;
  bool succ = CollectNamedtupleElements(call_node, namedtuple_aobj, &elems);
  if (!succ) {
    return nullptr;
  }
  // 2.There is no primitive for making namedtuple, so we use MakeTuple to create tuple (not namedtuple) in graph.
  const AbstractWrapperPtr &abs_wrapper = FGBuilder()->AddNode(prim::kPrimMakeTuple, elems);
  if (abs_wrapper == nullptr || abs_wrapper->abstract() == nullptr) {
    return nullptr;
  }
  // 3.MakeTuple primitive's output is AbstractTuple, we must convert it to AbstractNamedTuple manually.
  abstract::AbstractNamedTuplePtr new_abs = ConvertToAbstractNamedTuple(abs_wrapper->abstract(), namedtuple_aobj);
  if (new_abs == nullptr) {
    return nullptr;
  }
  AnfNodePtr node = FGBuilder()->ReadLocalVariable(abs_wrapper);
  MS_EXCEPTION_IF_NULL(node);
  node->set_abstract(new_abs);
  AbstractWrapperPtr new_abs_wrapper = std::make_shared<AbstractWrapper>(new_abs);
  FGBuilder()->UpdateNodesMap(new_abs_wrapper, node);
  return new_abs_wrapper;
}

bool MindGraphBuilder::CollectNamedtupleElements(const CallNode *call_node, const AbstractNamedTuple *namedtuple_aobj,
                                                 std::vector<AbstractWrapperPtr> *elems) {
  // 1.Get the __new__ method of namedtuple
  PyTypeObject *type_obj = namedtuple_aobj->GetTypeObject();
  auto type_pyobj = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(type_obj));
  py::object new_method = py::getattr(type_pyobj, "__new__");
  if (new_method.ptr() == nullptr || !PyFunction_Check(new_method.ptr())) {
    MS_LOG(INFO) << "Failed to get __new__ method from namedtuple " << namedtuple_aobj->type_name();
    return false;
  }
  // 2.Collect ValueNodes of the namedtuple elements
  std::vector<ValueNode *> element_nodes;
  ValueNode *cls_node = call_node->getInputs()[0];
  try {
    // This method throws an exception when arguments matching fails, so try-catch is required.
    BindArgumentsHelper<ValueNode *> args_helper =
      PackInputsForFunc(new_method, call_node->GetOpcode(), call_node->getInputs(), cls_node);
    // The __new__ method of namedtuple does not have variable-length arguments, so we just ignore va_ and kw_va_.
    const std::vector<ValueNode *> &args = args_helper.results().args_;
    // The first argument is cls_node, it is not the data element of namedtuple, so we remove it.
    (void)std::copy(args.begin() + 1, args.end(), std::back_inserter(element_nodes));
  } catch (std::exception &e) {
    MS_LOG(INFO) << "Failed to process __new__ method's arguments. Type name: " << namedtuple_aobj->type_name();
    return false;
  }
  // 3.Convert ValueNode to AbstractWrapper
  *elems = HandleInputArgs(element_nodes);
  return true;
}

namespace {
abstract::AbstractNamedTuplePtr ConvertToAbstractNamedTuple(const AbstractBasePtr &abstract,
                                                            const AbstractNamedTuple *namedtuple_aobj) {
  MS_EXCEPTION_IF_NULL(abstract);
  AbstractBasePtrList abstract_keys;
  for (const std::string &key : namedtuple_aobj->keys()) {
    ValuePtr converted_key = nullptr;
    bool succ = parse::ConvertData(py::str(key), &converted_key);
    if (!succ) {
      MS_LOG(INFO) << "Failed to convert namedtuple's key to ValuePtr: " << key;
      return nullptr;
    }
    abstract_keys.push_back(converted_key->ToAbstract());
  }

  auto abstract_tuple = abstract->cast<abstract::AbstractTuplePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  auto abstract_namedtuple = std::make_shared<abstract::AbstractNamedTuple>(namedtuple_aobj->type_name(), abstract_keys,
                                                                            abstract_tuple->elements());
  // The type stored in the user_data will be used to create the namedtuple python object.
  PyTypeObject *namedtuple_type = namedtuple_aobj->GetTypeObject();
  auto type = py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(namedtuple_type));
  abstract_namedtuple->set_user_data(kPijitNamedtupleType, std::make_shared<py::object>(type));
  return abstract_namedtuple;
}
}  // namespace

// Fix dynamic shape tensor get shape issue.
// Guard and Renormalize strategy should be refactored later.
AbstractWrapperPtr MindGraphBuilder::HandleGetShapeOfDynamicLengthTensor(const AbstractWrapperPtr &abstract_wrapper) {
  auto anf_node = FGBuilder()->ReadLocalVariable(abstract_wrapper);
  if (anf_node == nullptr || anf_node->abstract() == nullptr) {
    return nullptr;
  }
  auto abs = anf_node->abstract();
  auto shape = abs->BuildShape();
  if (!shape->isa<abstract::TensorShape>()) {
    return nullptr;
  }
  const auto &tensor_shape = shape->cast<abstract::TensorShapePtr>()->GetShapeVector();
  if (std::all_of(tensor_shape.begin(), tensor_shape.end(), [](auto e) { return e > 0; })) {
    return nullptr;
  }
  AbstractWrapperPtrList input_abstract_wrapper = {abstract_wrapper};
  return FGBuilder()->AddNode(prim::kPrimShape, input_abstract_wrapper);
}

static bool IsMutableAttribute(ValueNode *node) {
  const auto &attr_names = node->GetGraph()->Config().attr_as_param_list();
  if (attr_names.find(node->GetName()) == attr_names.end()) {
    return false;
  }
  auto op = node->GetVobj()->GetPyObject().ptr();
  if (op == nullptr) {
    return false;
  }
  // for mutable tuple and list, PyList_Check, PyTuple_Check
  // ... not implement

  // now, only support these scalar types
  if (PyLong_Check(op) || PyFloat_Check(op) || PyBool_Check(op)) {
    return true;
  }
  // now, for Tensor, mindspore.common.Parameter and mindspore._c_expression.Tensor as func input is unsupported
  if (!IsPybindType<tensor::Tensor, false>(Py_TYPE(op)) && py::isinstance<tensor::Tensor>(op)) {
    constexpr char name[] = "Parameter";
    constexpr auto len = sizeof(name) - 1;
    std::string type_name = Py_TYPE(op)->tp_name;
    return type_name.size() < len ? true : (0 != type_name.compare(type_name.size() - len, len, name));
  }
  return false;
}

void MindGraphBuilder::GuardAttribute(ValueNode *attr_node) {
  static constexpr auto const_type = {
    AObject::kTypeInt,   AObject::kTypeFloat, AObject::kTypeBool, AObject::kTypeMSDType,
    AObject::kTypeTuple, AObject::kTypeList,  AObject::kTypeDict,
  };
  const auto IsPrimWithAttr = [](const py::handle &) { /*delay this guard */ return false; };
  if (attr_node->IsConstantValue()) {
    return;
  }

  py::object attr_object = attr_node->GetVobj() ? attr_node->GetVobj()->GetPyObject() : py::object();
  if (attr_object.ptr() == nullptr) {
    return;
  }
  const auto &src_info = attr_node->input(0)->GetVobj();
  py::object src_object = src_info->GetPyObject();
  AObject::Type src_type = src_info->GetType();
  AObject::Type attr_type = attr_node->GetVobj()->GetType();
  const auto &name = attr_node->GetName();
  bool is_const_attr_type = const_type.end() != std::find(const_type.begin(), const_type.end(), attr_type);

  // the source type is constant, or has a guard and do following ...

  if (src_type == AObject::kTypeModule /* assume the attribute of module(or global variable) is not change */) {
    MS_LOG(INFO) << "skip guard the interface '" << name << "' of " << py::str(src_object.ptr());
    // How to fast check Module attribute(or global varialbe) changed ?
  } else if (attr_type == AObject::kTypeFunction || attr_type == AObject::kTypeCFunction ||
             attr_type == AObject::kTypeBoundMethod || attr_type == AObject::kTypeCell ||
             PyInstanceMethod_Check(attr_object.ptr())) {
    MS_LOG(INFO) << "delay guard the function type of '" << AObject::ToString(src_object.ptr(), false) << "'";
  } else if (src_type == AObject::kTypeTensor /* tensor attribute always constant. */
             || CheckConstPyObject(src_object.ptr()) /* constant python object not need check */) {
    MS_LOG(INFO) << "skip guard the attribute '" << name << "' of the builtin " << Py_TYPE(src_object.ptr())->tp_name;
  } else if (attr_type == AObject::kTypeTensor && IsParameterObject(attr_object)) {
    // The mindspore.Parameter must be guard id. FuncGraph reuse the object
    graph_->GuardValueNode(attr_node, GuardLevel::GId);
  } else if (is_const_attr_type || (attr_type == AObject::kTypePrimitive && IsPrimWithAttr(attr_object))) {
    // For primitive, check the attribute that transform to partial arguments
    graph_->GuardValueNode(attr_node, GuardLevel::GEqual);
  } else {
    // Anyway, guard the type of attribute
    graph_->GuardType(attr_node);
  }
  if (is_const_attr_type) {  // shouldn't set false
    attr_node->SetConstantValue(true);
  }
}

ValueNode *MindGraphBuilder::HandleGetattr(ValueNode *target_node, const Instr &instr) {
  if (IsNamedtupleGetElem(target_node, instr.name())) {
    return HandleNamedtupleGetElem(instr, target_node);
  }
  auto attr_node = NewValueNode(target_node->get_attr(instr.name()), instr, {target_node});
  MS_EXCEPTION_IF_NULL(attr_node);
  ValueNode *graph_attr_node = nullptr;
  auto attr_obj = attr_node->GetVobj()->GetPyObject();

  // handle attribute of dynamic shape tensor
  // Not implement for tensor.size, tensor.ndim ...
  if (instr.name() == "shape") {
    auto ret_wrapper = HandleGetShapeOfDynamicLengthTensor(target_node->abstract_wrapper());
    if (ret_wrapper != nullptr) {
      auto ret = NewValueNode(AObject::Convert(ret_wrapper), instr, {target_node});
      ret->set_abstract_wrapper(ret_wrapper);
      return ret;
    }
  }

  if (IsMutableAttribute(attr_node) && root()->GetGraph()->PrepareParameter(attr_node)) {
    const auto &cur_name = instr.name();
    MS_LOG(INFO) << "Try adding attribute " << cur_name << " as graph input";
    auto abstract_wrapper = FGBuilder()->AddAttributeInput(attr_obj);
    if (abstract_wrapper != nullptr) {
      graph_attr_node = attr_node;
      graph_attr_node->set_abstract_wrapper(abstract_wrapper);
      // tensor must be guard dtype
      bool is_tensor = attr_node->GetVobj()->GetType() == AObject::kTypeTensor;
      graph_->GuardValueNode(attr_node, is_tensor ? GuardLevel::GDeduce : GuardLevel::GType);
      return graph_attr_node;
    }
    MS_LOG(INTERNAL_EXCEPTION) << "Failed to add scalar or Tensor " << cur_name << " as input.";
  }
  MS_LOG(INFO) << "constant attribute " << attr_node->ToString();

  // If the attr_obj can convert to anf node directly, return the origin attr node.
  auto abstract_wrapper = FGBuilder()->AddAttrPythonObject(attr_obj);
  graph_attr_node = attr_node;
  if (abstract_wrapper == nullptr) {
    abstract_wrapper = FGBuilder()->AddLocalVariable(attr_obj);
  }
  if (abstract_wrapper != nullptr) {
    graph_attr_node->set_abstract_wrapper(abstract_wrapper);
  }
  // Need to check whether the guard is failed in the future.
  // Add Guard for getattr node. For scalar/list/tuple/primitive, need to guard value. Otherwise, guard type and shape.
  GuardAttribute(graph_attr_node);
  return graph_attr_node;
}

namespace {
// Check if it is a scenario of getting element from namedtuple.
// For example: `Point = namedtuple('Point', ['x', 'y']); p = Point(1, 2)`, only `p.x` or `p.y` is getting element.
// While `p.index()`, `p.count()`, or any other attributes or methods are not (they should fallback to getattr logic).
bool IsNamedtupleGetElem(const ValueNode *node, const std::string &name) {
  if (node != nullptr && node->GetVobj() != nullptr && node->GetVobj()->GetType() == AObject::kTypeNamedTuple) {
    auto *namedtuple_aobj = static_cast<AbstractNamedTuple *>(node->GetVobj());
    return namedtuple_aobj->HasKey(name);
  }
  return false;
}
}  // namespace

ValueNode *MindGraphBuilder::HandleNamedtupleGetElem(const Instr &instr, ValueNode *node) {
  MS_LOG(DEBUG) << "Do namedtuple getattr '" << instr.name() << "'";
  // Convert namedtuple's getattr by name to tuple's getitem by index.
  auto *namedtuple_aobj = static_cast<AbstractNamedTuple *>(node->GetVobj());
  int idx = namedtuple_aobj->GetIndexOfKey(instr.name());
  MS_EXCEPTION_IF_CHECK_FAIL(idx >= 0, "Can not find attribute '" + instr.name());

  const AbstractWrapperPtr &abs = FGTupleGetItem(node->abstract_wrapper(), idx);

  if (abs == nullptr || abs->abstract() == nullptr) {
    MS_LOG(INFO) << "Failed to do namedtuple getitem, idx=" << idx << ", node: " << node->ToString();
    return nullptr;
  }
  ValueNode *ret = NewValueNode(AObject::Convert(abs), instr, {node});
  ret->set_abstract_wrapper(abs);
  return ret;
}

AbstractWrapperPtr MindGraphBuilder::HandleMultiOp(const Instr &instr, const std::vector<ValueNode *> &p,
                                                   bool is_compare) {
  int opcode = instr.op();
  int oparg = instr.arg();
  std::string op_name;
  if (is_compare) {
    op_name = GraphUtils::OpCompareArgToGraphName(oparg);
  } else if (opcode == CONTAINS_OP) {
    op_name = GraphUtils::ContainsOpToGraphName(oparg);
  } else {
    op_name = GraphUtils::OpCodeToGraphName(opcode);
  }
  MS_LOG(DEBUG) << "operation name is " << op_name;
  if (op_name == "") {
    MS_LOG(INFO) << "Can not find operation for " << instr.ToString();
    return nullptr;
  }
  auto wrapper = FGBuilder()->AddMultiNode(op_name, HandleInputArgs(p));
  return wrapper;
}

AbstractWrapperPtr MindGraphBuilder::HandleBuildOp(const Instr &instr, const std::vector<ValueNode *> &p) {
  auto opcode = instr.op();
  AbstractWrapperPtrList inputs_wrapper = HandleInputArgs(p);
  if (inputs_wrapper.end() != std::find(inputs_wrapper.begin(), inputs_wrapper.end(), nullptr)) {
    MS_LOG(INFO) << "infer failed with nullptr input. " << instr.ToString();
    return nullptr;
  }
  auto primitive = pijit::GraphUtils::GetPrimitive(opcode);
  if (primitive == nullptr) {
    MS_LOG(INFO) << "Can not find primitive for " << instr.ToString();
    return nullptr;
  }
  if (primitive == prim::kPrimStringConcat) {
    return HandleBuildStringOp(primitive, inputs_wrapper);
  }
  if (primitive == prim::kPrimMakeDict) {
    if (opcode == BUILD_CONST_KEY_MAP) {
      MS_LOG(DEBUG) << "BUILD_CONST_KEY_MAP case, need to pack values.";
      AbstractWrapperPtrList value_inputs_wrapper;
      (void)std::copy(inputs_wrapper.begin(), inputs_wrapper.end() - 1, std::back_inserter(value_inputs_wrapper));
      auto value_wrapper = FGBuilder()->AddNode(prim::kPrimMakeTuple, value_inputs_wrapper);
      inputs_wrapper = {inputs_wrapper.back(), value_wrapper};
    } else {
      MS_LOG(DEBUG) << "BUILD_KEY_MAP case, need to pack keys and values.";
      size_t input_len = inputs_wrapper.size();
      if (input_len % 2 != 0) {
        MS_LOG(INTERNAL_EXCEPTION) << "BUILD_KEY_MAP should have even input, but got: " << input_len;
      }
      AbstractWrapperPtrList key_inputs_wrapper;
      AbstractWrapperPtrList value_inputs_wrapper;
      for (size_t i = 0; i < input_len / 2; ++i) {
        key_inputs_wrapper.push_back(inputs_wrapper[2 * i]);
        value_inputs_wrapper.push_back(inputs_wrapper[2 * i + 1]);
      }
      auto key_wrapper = FGBuilder()->AddNode(prim::kPrimMakeTuple, key_inputs_wrapper);
      auto value_wrapper = FGBuilder()->AddNode(prim::kPrimMakeTuple, value_inputs_wrapper);
      inputs_wrapper = {key_wrapper, value_wrapper};
    }
  }
  if (primitive == prim::kPrimMakeSlice) {
    constexpr size_t slice_without_step_len = 2;
    if (inputs_wrapper.size() == slice_without_step_len) {
      // Handle slice without step input scene, such as 0:2. MakeSlice can only handle slice with full inputs.
      (void)inputs_wrapper.emplace_back(FGBuilder()->AddLocalVariable(py::int_(1)));
    }
  }
  auto wrapper = FGBuilder()->AddNode(primitive, inputs_wrapper);
  return wrapper;
}

AbstractWrapperPtr MindGraphBuilder::HandleBuildStringOp(const PrimitivePtr &primitive,
                                                         const AbstractWrapperPtrList &inputs_wrapper) {
  // The string_concat primitive only supports concatenating two strings.
  // Thus, if we want to concatenate multiple strings, we need to call this primitive multiple times.
  MS_LOG(DEBUG) << "Handle BUILD_STRING op, concat " << inputs_wrapper.size() << " strings";
  if (std::any_of(inputs_wrapper.begin(), inputs_wrapper.end(), [](auto abs) { return abs == nullptr; })) {
    MS_LOG(INFO) << "Failed to do string concat, found null input arg";
    return nullptr;
  }
  AbstractWrapperPtr result_str = inputs_wrapper[0];
  for (size_t i = 1; i < inputs_wrapper.size(); ++i) {
    const AbstractWrapperPtr &concated_str = FGBuilder()->AddNode(primitive, {result_str, inputs_wrapper[i]});
    if (concated_str == nullptr || concated_str->abstract() == nullptr) {
      MS_LOG(INFO) << "Failed to do string concat. Left string: " << result_str->ToString()
                   << ". Right string is inputs[" << i << "]: " << inputs_wrapper[i]->ToString();
      return nullptr;
    }
    result_str = concated_str;
  }
  return result_str;
}

bool MindGraphBuilder::DoGetItem(const Instr &instr) {
  auto r = pop();
  auto l = pop();
  auto o = HandleMultiOp(instr, {l, r}, false);
  if (o != nullptr) {
    auto v = NewValueNode(AObject::Convert(o), instr, {l, r});
    v->set_abstract_wrapper(o);
    push(v);
    return true;
  }
  /*
   * Currently, there are several known scenarios where MetaFuncGraph-getitem infer may fail:
   * 1.Cell list getitem: The Cell to be retrieved fails to be parsed (e.g. due to side effects in Cell.construct(),
   * such as STORE_ATTR).
   * 2.Index Out-of-Bounds or Key not exist.
   * In the first case, we can try to fallback to two-stages mode and do getitem from python object.
   * But in the second case, fallback to two-stages is useless. However, we cannot distinguish which scenario caused
   * getitem infer failure, so we always fallback to have a try.
   */
  MS_LOG(INFO) << "Getitem infer failed, fallback to do getitem from python obj. source: " << l->ToString()
               << ", key: " << r->ToString();
  push(l);
  push(r);
  if (GraphBuilder::DoGetItem(instr)) {
    auto v = seek(0);
    o = FGBuilder()->AddLocalVariable(v->GetVobj()->GetPyObject());
    if (o != nullptr) {
      v->set_abstract_wrapper(o);
      return true;
    }
  }
  MS_LOG(INFO) << "Failed to getitem from python obj, do break graph";
  return false;
}

bool MindGraphBuilder::DoUnary(const Instr &instr) {
  auto o = pop();
  auto r = HandleMultiOp(instr, {o}, false);
  auto v = NewValueNode(AObject::Convert(r), instr, {o});
  v->set_abstract_wrapper(r);
  push(v);
  return true;
}

bool MindGraphBuilder::DoBinary(const Instr &instr) {
  auto r = pop();
  auto l = pop();
  auto o = HandleMultiOp(instr, {l, r}, false);
  auto v = NewValueNode(AObject::Convert(o), instr, {l, r});
  v->set_abstract_wrapper(o);
  push(v);
  return true;
}

bool MindGraphBuilder::DoBinaryMul(const Instr &instr) {
  auto r = pop();
  auto l = pop();
  auto o = HandleMultiOp(instr, {l, r}, false);
  auto v = NewValueNode(AObject::Convert(o), instr, {l, r});
  v->set_abstract_wrapper(o);
  push(v);
  return true;
}

bool MindGraphBuilder::DoCompare(const Instr &instr) {
  // python3.7 only
  Opcode opcode(instr.op());
  int oparg = instr.arg();
  if (opcode.CheckIsOp(oparg)) {
    return DoIsOp(instr);
  }
  if (opcode.IsExcMatch(oparg)) {
    auto r = pop();
    auto l = pop();
    auto expectedErrs = r->GetVobj()->GetPyObject().ptr();
    auto gotErr = l->GetVobj()->GetPyObject().ptr();
    if (!PyTuple_Check(expectedErrs) && !PyExceptionClass_Check(expectedErrs)) {
      MS_LOG(ERROR) << "unsupported except types: " << Py_TYPE(expectedErrs);
      return false;
    }

    auto res = PyErr_GivenExceptionMatches(gotErr, expectedErrs);
    auto v = NewValueNode(AObject::Convert(res ? Py_True : Py_False), instr, {l, r});
    push(v);
    MS_LOG(INFO) << "unsupported Exception Match, try to constant fold. [" << v->ToString();
    v->set_abstract_wrapper(FGBuilder()->AddLocalVariable(py::bool_(res)));
    return true;
  }

  auto r = pop();
  auto l = pop();
  auto o = HandleMultiOp(instr, {l, r}, true);
  auto v = NewValueNode(AObject::Convert(o), instr, {l, r});
  v->set_abstract_wrapper(o);
  push(v);
  return true;
}

bool MindGraphBuilder::DoBuildOp(const Instr &instr) {
  if (instr.op() == BUILD_SET) {
    return false;
  }
  int opcode = instr.op();
  int oparg = instr.arg();
  int tmp_arg = oparg;
  tmp_arg += opcode == BUILD_CONST_KEY_MAP;
  tmp_arg += opcode == BUILD_MAP ? tmp_arg : 0;
  std::vector<ValueNode *> p(frame_.GetStacks().end() - tmp_arg, frame_.GetStacks().end());
  auto o = HandleBuildOp(instr, p);
  popn(tmp_arg);
  auto v = NewValueNode(AObject::Convert(o), instr, p);
  v->set_abstract_wrapper(o);
  push(v);
  return true;
}

bool MindGraphBuilder::DoIsOp(const Instr &instr) {
  bool invert;
  if (!Opcode(instr.op()).CheckIsOp(instr.arg(), &invert)) {
    return false;
  }
  auto r = pop();
  auto l = pop();
  int res = AObject::BinaryIs(l->GetVobj(), r->GetVobj());
  auto o = res == -1 ? AObject::MakeAObject(AObject::kTypeBool) : AObject::Convert((res ^ invert) ? Py_True : Py_False);
  auto v = NewValueNode(o, instr, {l, r});
  push(v);
  MS_LOG(INFO) << "unsupported IS_OP, try to constant fold. [" << v->ToString();
  v->set_abstract_wrapper(FGBuilder()->AddLocalVariable(o->GetPyObject()));
  return true;
}

bool MindGraphBuilder::DoContainsOp(const Instr &instr) {
  auto r = pop();
  auto l = pop();
  auto o = HandleMultiOp(instr, {l, r}, false);
  if (o == nullptr) {
    MS_LOG(INFO) << "Failed to handle bytecode CONTAINS_OP";
    return false;
  }
  auto v = NewValueNode(AObject::Convert(o), instr, {l, r});
  v->set_abstract_wrapper(o);
  push(v);
  return true;
}

void MindGraphBuilder::DoLoadGlobal(const Instr &instr) {
  this->GraphBuilder::DoLoadGlobal(instr);
  ValueNode *node = seek(0);
  py::object handle = node->GetVobj()->GetPyObject();
  if (handle.ptr() == nullptr) {
    return;  // name not define
  }
  node->set_abstract_wrapper(FGBuilder()->AddLocalVariable(handle));
}

bool MindGraphBuilder::DoLoadConst(const Instr &instr) {
  const py::object &const_obj = instr.cnst();
  const AbstractWrapperPtr &abs = FGBuilder()->AddLocalVariable(const_obj);
  auto node = NewValueNode(AObject::Convert(const_obj), instr, {});
  node->set_abstract_wrapper(abs);
  push(node);
  return true;
}

bool MindGraphBuilder::HandlePositionParams(const py::object &func, std::vector<ValueNode *> *params,
                                            FrameStates *frame) {
  CallNode *call_node = reinterpret_cast<CallNode *>(seek(0));
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  auto vobj = AObject::Convert(func.ptr());
  AObject::Type callable_type = vobj->GetType();

  ValueNode *self = GetBoundSelf(call_node);
  if (self != nullptr) {
    params->insert(params->begin(), self);
  }

  const int argc = co->co_argcount;
  const int has_varg = (co->co_flags & CO_VARARGS) ? 1 : 0;
  const int has_kwvarg = (co->co_flags & CO_VARKEYWORDS) ? 1 : 0;
  const int varg_loc = argc + co->co_kwonlyargcount;
  const int kwvarg_loc = argc + co->co_kwonlyargcount + has_varg;
  int pargc = params->size();
  if (pargc > argc && !has_varg) {
    MS_LOG(DEBUG) << "too many parameters";
    return false;
  }
  bool append_self_to_varg = has_varg && self && callable_type == AObject::kTypeBoundMethod && argc == 0;
  if (append_self_to_varg) {  // self is in variable arguments
    MS_LOG(INFO) << "not implement append self to variable arguments, inline failed";
    return false;
  }

  if (has_kwvarg && frame->Local(kwvarg_loc) == &ValueNode::kUnboundLocal) {
    DoBuildOp({BUILD_MAP, 0});
    auto m = pop();
    call_node->AddParam(m);
    frame->SetLocal(kwvarg_loc, m);
  }

  if (has_varg) {
    int vargc = pargc > argc ? pargc - argc : 0;
    std::vector<ValueNode *> vargs(params->end() - vargc, params->end());
    params->resize(params->size() - vargc);
    std::for_each(vargs.begin(), vargs.end(), [this](ValueNode *i) { this->push(i); });
    DoBuildOp({BUILD_TUPLE, static_cast<int>(vargs.size())});
    ValueNode *build_tuple = pop();
    call_node->AddParam(build_tuple);
    frame->SetLocal(varg_loc, build_tuple);
  }

  pargc = params->size();
  for (int i = pargc - 1; i >= 0; --i) {
    if (frame->Local(i) != &ValueNode::kUnboundLocal) {
      MS_LOG(DEBUG) << "duplicate key-word parameter error";
      return false;
    }
    frame->SetLocal(i, params->back());
    params->pop_back();
  }

  return CheckAndSetDefaultParams(func, frame, pargc);
}

bool MindGraphBuilder::UnpackCallExParams(std::vector<ValueNode *> *params, int extra_local, bool *has_kw,
                                          CallNode *call_node) {
  bool has_dict = params->size() > 1;
  ValueNode *args_node = params->operator[](0);
  if (!has_dict) {
    params->clear();
  } else if (!UnpackCallExDict(params, call_node)) {
    return false;
  }
  *has_kw = params->size();

  if (args_node->GetVobj() == nullptr) {
    return false;
  }
  py::object object = args_node->GetVobj()->GetPyObject();
  if (!py::isinstance<py::tuple>(object) && !py::isinstance<py::list>(object)) {
    return false;
  }
  size_t args_len = 0;
  if (py::isinstance<py::tuple>(object)) {
    args_len = py::len(py::cast<py::tuple>(object));
  } else {
    args_len = py::len(py::cast<py::list>(object));
  }
  if (args_len == 0) {
    return true;
  }

  std::vector<ValueNode *> new_args_inputs;
  for (size_t i = 0; i < args_len; ++i) {
    push(args_node);
    DoLoadConst({LOAD_CONST, -1, py::int_(i)});
    DoItemAccess({BINARY_SUBSCR, 0});
    new_args_inputs.push_back(pop());
  }

  params->insert(params->begin(), new_args_inputs.begin(), new_args_inputs.end());
  return true;
}

bool MindGraphBuilder::HandleKWParams(const py::object &func, std::vector<ValueNode *> *params, FrameStates *frame) {
  PyCodeObject *co = reinterpret_cast<PyCodeObject *>(PyFunction_GET_CODE(func.ptr()));
  std::vector<ValueNode *> kwvargs;
  if (!PackKwParams(func, params, frame, &kwvargs)) {
    // illegal arguments
    return false;
  }

  const int argc = co->co_argcount + co->co_kwonlyargcount;
  if (!(co->co_flags & CO_VARKEYWORDS)) {
    // kw_2_p_cnt == k_cnt, all kw arguments is positions arguments
    return true;
  }

  int kwvarg_loc = argc + ((co->co_flags & CO_VARARGS) ? 1 : 0);
  std::for_each(kwvargs.begin(), kwvargs.end(), [this](ValueNode *i) { this->push(i); });
  DoBuildOp({BUILD_MAP, SizeToInt(kwvargs.size() / 2)});
  ValueNode *new_node = pop();
  frame->SetLocal(kwvarg_loc, new_node);
  graph_->GetTracedNodes().pop_back();

  static_cast<CallNode *>(seek(0))->AddParam(frame->Local(kwvarg_loc));
  return true;
}

bool MindGraphBuilder::UnpackCallExDict(std::vector<ValueNode *> *params, CallNode *call_node) {
  ValueNode *dict_node = params->back();
  params->clear();

  if (dict_node->GetVobj() == nullptr) {
    return false;
  }

  auto object = dict_node->GetVobj()->GetPyObject();
  if (!py::isinstance<py::dict>(object)) {
    return false;
  }
  auto dict_object = py::cast<py::dict>(object);
  Py_ssize_t dict_len = py::len(dict_object);
  if (dict_len == 0) {
    return true;
  }

  py::tuple keys(dict_len);
  size_t i = 0;
  for (const auto &pair : dict_object) {
    auto cur_key = pair.first;
    if (!py::isinstance<py::str>(cur_key)) {
      return false;
    }
    keys[i] = cur_key;
    push(dict_node);
    DoLoadConst({LOAD_CONST, -1, py::reinterpret_borrow<py::object>(cur_key)});
    DoItemAccess({BINARY_SUBSCR, 0});
    params->push_back(pop());
    i++;
  }

  DoLoadConst({LOAD_CONST, -1, keys});
  params->push_back(pop());
  return true;
}

bool MindGraphBuilder::DoItemAccess(const Instr &instr) {
  int opcode = instr.op();
  bool res = false;
  if (opcode == BINARY_SUBSCR) {
    res = DoGetItem(instr);
  } else if (opcode == STORE_SUBSCR) {
    auto key = pop();
    auto map = pop();
    auto value = pop();
    NewValueNode(nullptr, instr, {value, map, key});
    res = DoSetItem(map, key, value);
  } else if (opcode == DELETE_SUBSCR) {
    auto key = pop();
    auto map = pop();
    NewValueNode(nullptr, instr, {map, key});
    res = DoSetItem(map, key, nullptr);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "parser got an error instruction " << instr.ToString();
  }
  return res;
}
}  // namespace pijit
}  // namespace mindspore
