/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "pipeline/jit/pi/graph_guard/trace.h"
#include "pipeline/jit/pi/utils/opcode_declare.h"

namespace mindspore {
namespace pijit {

namespace {

#define FUNC_MAPPING_ENUM    \
  ENUM_ITEM(Unknown)         \
  ENUM_ITEM(PyObjectGetItem) \
  ENUM_ITEM(PyNumberAdd)     \
  ENUM_ITEM(PyObjectLength)  \
  ENUM_ITEM(PyType)

#define ENUM_ITEM(item) k##item,
enum FuncMapEnum { FUNC_MAPPING_ENUM };
#undef ENUM_ITEM

using OptimizeFuncT = TracePtr (*)(const OpTracePtr &);
constexpr OptimizeFuncT OptimizeUnknown = nullptr;
// create fast item access trace
TracePtr OptimizePyObjectGetItem(const OpTracePtr &);
// create fast add trace
TracePtr OptimizePyNumberAdd(const OpTracePtr &);
// create fast length trace
TracePtr OptimizePyObjectLength(const OpTracePtr &);
// create fast type trace, fold trace path `x=type(range(1))` to `x=range`, this is internal case
TracePtr OptimizePyType(const OpTracePtr &);

#define ENUM_ITEM(item) Optimize##item,
constexpr OptimizeFuncT optimize_func_map[] = {FUNC_MAPPING_ENUM};
#undef ENUM_ITEM
#undef FUNC_MAPPING_ENUM

// fold trace path `x=[a[0]][0]` to `x=a`
TracePtr FoldTupleGetItem(const OpTracePtr &trace);
// fold trace path `x={"k":{"k":a]["k"]}}["k"]` to `x=a`, all key must be constant
// fold trace path `x=[]+[]+[a]` to `x=[a]`
TracePtr FoldTupleAdd(const OpTracePtr &trace);
// fold trace path `x=len([1,2,3])` to `x=3`, this is internal case
TracePtr FoldTupleLengthTrace(const OpTracePtr &trace);
// fold trace path `x=[p[0],p[1]]`, to `x=p[0:2]`

std::string GenSignature(PyTypeObject *const *arr, size_t size) {
  std::stringstream s;
  s << "(";
  for (size_t i = 0; i < size; ++i) {
    s << arr[i]->tp_name << ",";
  }
  s.seekp(-1, s.cur);
  s << ")";
  return s.str();
}

std::string GenSignature(std::initializer_list<PyTypeObject *> arr) { return GenSignature(arr.begin(), arr.size()); }

FuncMapEnum MapPythonBuiltinCall(PyMethodDef *md) {
  if (std::string_view("len") == md->ml_name) {
    return FuncMapEnum::kPyObjectLength;
  }
  return FuncMapEnum::kUnknown;
}

FuncMapEnum MapPythonOperatorCall(PyMethodDef *md) {
  if (std::string_view("getitem") == md->ml_name) {
    return FuncMapEnum::kPyObjectGetItem;
  }
  return FuncMapEnum::kUnknown;
}

FuncMapEnum MapPythonCall(PyObject *callable) {
  if (callable == reinterpret_cast<PyObject *>(&PyType_Type)) {
    return FuncMapEnum::kPyType;
  }
  if (!PyCFunction_Check(callable)) {
    return FuncMapEnum::kUnknown;
  }
  PyCFunctionObject *cfunc = reinterpret_cast<PyCFunctionObject *>(callable);
  if (cfunc->m_module == nullptr || !PyUnicode_Check(cfunc->m_module)) {
    return FuncMapEnum::kUnknown;
  }
  if (PyUnicode_CompareWithASCIIString(cfunc->m_module, "builtins") == 0) {
    return MapPythonBuiltinCall(cfunc->m_ml);
  }
  if (PyUnicode_CompareWithASCIIString(cfunc->m_module, "_operator") == 0) {
    return MapPythonOperatorCall(cfunc->m_ml);
  }
  return FuncMapEnum::kUnknown;
}

FuncMapEnum GetFuncMapEnum(const OpTracePtr &tr) {
  Opcode opcode(tr->GetOpCode());
  FuncMapEnum func = FuncMapEnum::kUnknown;
  if (opcode == BINARY_SUBSCR) {
    func = FuncMapEnum::kPyObjectGetItem;
  } else if (opcode.IsBinaryMath() && opcode == BINARY_ADD) {
    func = FuncMapEnum::kPyNumberAdd;
  } else if (opcode.IsCall() && tr->GetParam(0)->GetTraceType() == TraceType::Const) {
    // only if internal generated call trace
    func = MapPythonCall(tr->GetParam(0)->GetObject());
  }
  return func;
}

TracePtr OptimizePyObjectGetItem(const OpTracePtr &tr) {
  static std::map<std::string, OptimizeFuncT> fold_map = {
    {GenSignature({&PyTuple_Type, &PyLong_Type}), FoldTupleGetItem},
    {GenSignature({&PyList_Type, &PyLong_Type}), FoldTupleGetItem},
  };
  auto sig = GenSignature({Py_TYPE(tr->GetParam(0)->GetObject()), Py_TYPE(tr->GetParam(1)->GetObject())});
  auto iter = fold_map.find(sig);
  return iter == fold_map.end() ? tr : iter->second(tr);
}

TracePtr OptimizePyNumberAdd(const OpTracePtr &tr) {
  static std::map<std::string, OptimizeFuncT> fold_map = {
    {GenSignature({&PyTuple_Type, &PyTuple_Type}), FoldTupleAdd},
    {GenSignature({&PyList_Type, &PyList_Type}), FoldTupleAdd},
  };
  auto sig = GenSignature({Py_TYPE(tr->GetParam(0)->GetObject()), Py_TYPE(tr->GetParam(1)->GetObject())});
  auto iter = fold_map.find(sig);
  return iter == fold_map.end() ? tr : iter->second(tr);
}

TracePtr OptimizePyObjectLength(const OpTracePtr &tr) {
  static std::map<std::string, OptimizeFuncT> fold_map = {
    {GenSignature({&PyTuple_Type}), FoldTupleLengthTrace},
    {GenSignature({&PyList_Type}), FoldTupleLengthTrace},
    {GenSignature({&PyDict_Type}), FoldTupleLengthTrace},
  };
  // only internal generated trace is constant
  if (tr->GetParam(0)->GetTraceType() != TraceType::Const) {
    return tr;
  }
  auto sig = GenSignature({Py_TYPE(tr->GetParam(0)->GetObject())});
  auto iter = fold_map.find(sig);
  return iter == fold_map.end() ? tr : iter->second(tr);
}

TracePtr OptimizePyType(const OpTracePtr &tr) {
  // only internal generated trace is constant
  // pattern `type(input)`
  if (tr->GetParam(1)->GetTraceType() != TraceType::Operation) {
    return tr;
  }
  OpTracePtr input = std::static_pointer_cast<OpTrace>(tr->GetParam(1));
  if (!Opcode(input->GetOpCode()).IsCall()) {
    return tr;
  }
  // pattern `type(class(...))`
  if (!Py_IS_TYPE(input->GetParam(1)->GetObject(), &PyType_Type)) {
    return tr;
  }
  return input->GetParam(1);
}

TracePtr FoldTupleGetItem(const OpTracePtr &trace) {
  TracePtr fast_trace = trace;
  // create fast_trace CustomizeTrace { return PyTuple_Check(src) ? PyTuple_GET_ITEM(src, index) : nullptr; }
  TracePtr index = trace->GetParam(1);
  if (index->GetTraceType() != TraceType::Const) {
    return fast_trace;
  }
  TracePtr src = trace->GetParam(0);
  if (src->GetTraceType() == TraceType::Const) {
    MS_LOG(ERROR) << "should be fold while graph builing, but generated a guard trace: " << std::endl
                  << trace->ToString();
    return std::make_shared<ConstTrace>(trace->GetObject(), -1);
  }
  if (src->GetTraceType() != TraceType::Operation) {
    return fast_trace;
  }
  OpTracePtr tuple = std::static_pointer_cast<OpTrace>(src);
  if (Opcode(tuple->GetOpCode()).IsBuildOp()) {
    auto index_value = PyLong_AsLong(index->GetObject());
    return tuple->GetParam(index_value);
  }
  return fast_trace;
}

TracePtr FoldTupleAdd(const OpTracePtr &trace) {
  TracePtr left = trace->GetParam(0);
  TracePtr right = trace->GetParam(1);
  if (left->GetTraceType() == TraceType::Const || right->GetTraceType() == TraceType::Const) {
    MS_LOG(ERROR) << "should be fold while graph builing, but generated a guard trace: " << std::endl
                  << trace->ToString();
    return std::make_shared<ConstTrace>(trace->GetObject(), -1);
  }
  if (left->GetTraceType() != TraceType::Operation || right->GetTraceType() != TraceType::Operation) {
    return trace;
  }
  OpTracePtr left_op = std::static_pointer_cast<OpTrace>(left);
  OpTracePtr right_op = std::static_pointer_cast<OpTrace>(right);
  bool is_build_pattern = Opcode(left_op->GetOpCode()).IsBuildOp();
  if (!is_build_pattern) {
    return trace;
  }
  TraceVector params;
  for (const auto &tr : {left_op, right_op}) {
    for (size_t i = 0, size = tr->GetParamCount(); i < size; ++i) {
      params.push_back(tr->GetParam(i));
    }
  }
  return CreateOpTrace(trace->GetObject(), left_op->GetOpCode(), params.size(), params);
}

// fold trace path `x=len([1,2,3])` to `x=3`, this is internal case
TracePtr FoldTupleLengthTrace(const OpTracePtr &trace) {
  bool is_call = Opcode(trace->GetOpCode()).IsCall();  // maybe opcode GET_LEN
  TracePtr input = trace->GetParam(is_call);
  if (input->GetTraceType() != TraceType::Operation) {
    return trace;
  }
  Opcode code(std::static_pointer_cast<OpTrace>(input)->GetOpCode());
  if (!code.IsBuildOp()) {
    return trace;
  }
  MS_LOG(ERROR) << "should be fold while graph builing, but generated a trace: " << std::endl << trace->ToString();
  return std::make_shared<ConstTrace>(trace->GetObject(), -1);
}

}  // namespace

TracePtr OpTrace::Fold() {
  OpTracePtr current = std::static_pointer_cast<OpTrace>(shared_from_this());
  if (is_fold_) {
    return current;
  }
  is_fold_ = true;
  for (size_t i = 0, size = this->GetParamCount(); i < size; ++i) {
    if (this->GetParam(i)->GetTraceType() == TraceType::Operation) {
      OpTracePtr input = std::static_pointer_cast<OpTrace>(this->GetParam(i));
      if (!input->is_fold_) {
        MS_LOG(DEBUG) << "The trace not fold: " << input->ToString();
        this->params_[i] = input->Fold();
      }
    }
  }
  FuncMapEnum func = GetFuncMapEnum(current);
  return optimize_func_map[func] == nullptr ? current : optimize_func_map[func](current);
}

}  // namespace pijit
}  // namespace mindspore
