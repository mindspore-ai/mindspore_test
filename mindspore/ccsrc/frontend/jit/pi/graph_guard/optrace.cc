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
#include "frontend/jit/pi/graph_guard/trace.h"
#include "frontend/jit/pi/utils/opcode_declare.h"
#include "frontend/jit/pi/utils/utils.h"
#include "frontend/jit/pi/graph_guard/infer.h"

namespace mindspore {
namespace pijit {

namespace {

struct EmptyData {};
class FastTraceBase;
using FastTraceBasePtr = std::shared_ptr<FastTraceBase>;
class FastTraceBase : public OpTrace {
 protected:
  explicit FastTraceBase(const OpTracePtr &ptr)
      : OpTrace(*ptr), src_type_(nullptr), has_data_(false), check_op_(false) {}

 public:
  std::string ToString(bool include_param) override;
  const InfoPack &Info() override;

  static TracePtr CreateFastLen(const OpTracePtr &tr);
  static TracePtr CreateFastItem(const OpTracePtr &tr);
  static TracePtr CreateFastAttr(const OpTracePtr &tr);

  template <Py_ssize_t (&func_ref)(PyObject *o)>
  static py::object Len(FastTraceBase *this_ptr, PTraceContext context, bool perf);

  static py::object Len4TupleList(FastTraceBase *this_ptr, PTraceContext context, bool perf);
  static py::object Len4Dict(FastTraceBase *this_ptr, PTraceContext context, bool perf);
  static py::object Len4Tensor(FastTraceBase *this_ptr, PTraceContext context, bool perf);

  static py::object Item4TupleList_Int(FastTraceBase *this_ptr, PTraceContext context, bool perf);
  static py::object Item4Dict(FastTraceBase *this_ptr, PTraceContext context, bool perf);
  static py::object Attr4MsCell(FastTraceBase *this_ptr, PTraceContext context, bool perf);

  PyTypeObject *src_type_;
  bool has_data_;
  bool check_op_;  // true if opcode is call and function is not const
};

// use template to inline the func_ref
template <typename DataT, typename FuncT, FuncT &func_ref>
class FastTrace : public FastTraceBase {
 public:
  explicit FastTrace(const OpTracePtr &ptr) : FastTraceBase(ptr) {}
  py::object Retrieve(PTraceContext context, bool perf) override {
    if (retrieved_) {
      return retrieve_cache_;
    }
    py::object ret = func_ref(this, context, perf);
    if (PyErr_Occurred()) {
      MS_LOG(DEBUG) << "guard trace failed because of: " << py::error_already_set().what();
      PyErr_Clear();
      ret = {};
    }
    Cache(context, ret);
    return ret;
  }

  DataT cache_;
};

#define FUNC_MAPPING_ENUM    \
  ENUM_ITEM(Unknown)         \
  ENUM_ITEM(PyObjectGetItem) \
  ENUM_ITEM(PyNumberAdd)     \
  ENUM_ITEM(PyObjectLength)  \
  ENUM_ITEM(PyObjectGetAttr) \
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
// create fast attr trace, transform mindspore.nn.Cell.__getattr__ to c++ code, optimize tensor attr access
TracePtr OptimizePyObjectGetAttr(const OpTracePtr &);

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
  std::string_view name = md->ml_name;
  if (std::string_view("len") == name) {
    return FuncMapEnum::kPyObjectLength;
  }
  if (std::string_view("getattr") == name) {
    return FuncMapEnum::kPyObjectGetAttr;
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
  } else if (opcode == LOAD_ATTR) {
    func = FuncMapEnum::kPyObjectGetAttr;
  } else if (opcode.IsBinaryMath() && opcode == BINARY_ADD) {
    func = FuncMapEnum::kPyNumberAdd;
  } else if (opcode.IsCall()) {
    auto param0 = tr->GetParam(0);
    MS_EXCEPTION_IF_NULL(param0);
    func = MapPythonCall(param0->GetObject());
  }
  return func;
}

TracePtr OptimizePyObjectGetItem(const OpTracePtr &tr) {
  // exactly type match
  static std::map<std::string, OptimizeFuncT> fold_map = {
    {GenSignature({&PyTuple_Type, &PyLong_Type}), FoldTupleGetItem},
    {GenSignature({&PyList_Type, &PyLong_Type}), FoldTupleGetItem},
  };
  auto param0 = tr->GetParam(0);
  MS_EXCEPTION_IF_NULL(param0);
  auto param1 = tr->GetParam(1);
  MS_EXCEPTION_IF_NULL(param1);
  auto sig = GenSignature({Py_TYPE(param0->GetObject()), Py_TYPE(param1->GetObject())});
  auto iter = fold_map.find(sig);
  auto new_tr = iter == fold_map.end() ? tr : iter->second(tr);
  if (new_tr != tr) {
    return new_tr;
  }
  return FastTraceBase::CreateFastItem(tr);
}

TracePtr OptimizePyNumberAdd(const OpTracePtr &tr) {
  static std::map<std::string, OptimizeFuncT> fold_map = {
    {GenSignature({&PyTuple_Type, &PyTuple_Type}), FoldTupleAdd},
    {GenSignature({&PyList_Type, &PyList_Type}), FoldTupleAdd},
  };
  auto param0 = tr->GetParam(0);
  MS_EXCEPTION_IF_NULL(param0);
  auto param1 = tr->GetParam(1);
  MS_EXCEPTION_IF_NULL(param1);
  auto sig = GenSignature({Py_TYPE(param0->GetObject()), Py_TYPE(param1->GetObject())});
  auto iter = fold_map.find(sig);
  return iter == fold_map.end() ? tr : iter->second(tr);
}

TracePtr OptimizePyObjectLength(const OpTracePtr &tr) {
  static std::map<std::string, OptimizeFuncT> fold_map = {
    {GenSignature({&PyTuple_Type}), FoldTupleLengthTrace},
    {GenSignature({&PyList_Type}), FoldTupleLengthTrace},
    {GenSignature({&PyDict_Type}), FoldTupleLengthTrace},
  };
  TracePtr new_tr = tr;
  auto param0 = tr->GetParam(0);
  MS_EXCEPTION_IF_NULL(param0);
  bool fold = param0->GetTraceType() == TraceType::Const;
  if (!fold) {
    MS_LOG(INFO) << "got a length trace without constant len func, skip guard global len is builtin function len";
  }
  if (fold) {
    auto param1 = tr->GetParam(1);
    MS_EXCEPTION_IF_NULL(param1);
    auto sig = GenSignature({Py_TYPE(param1->GetObject())});
    auto iter = fold_map.find(sig);
    new_tr = iter == fold_map.end() ? tr : iter->second(tr);
  }
  if (new_tr != tr) {
    return new_tr;
  }
  return FastTraceBase::CreateFastLen(tr);
}

TracePtr OptimizePyType(const OpTracePtr &tr) {
  // only internal generated trace is constant
  // pattern `type(input)`
  auto param1 = tr->GetParam(1);
  MS_EXCEPTION_IF_NULL(param1);
  if (param1->GetTraceType() != TraceType::Operation) {
    return tr;
  }
  OpTracePtr input = std::static_pointer_cast<OpTrace>(param1);
  if (!Opcode(input->GetOpCode()).IsCall()) {
    return tr;
  }
  // pattern `type(class(...))`
  auto input_param1 = input->GetParam(1);
  MS_EXCEPTION_IF_NULL(input_param1);
  if (!Py_IS_TYPE(input_param1->GetObject(), &PyType_Type)) {
    return tr;
  }
  return input_param1;
}

TracePtr OptimizePyObjectGetAttr(const OpTracePtr &tr) { return FastTraceBase::CreateFastAttr(tr); }

TracePtr FoldTupleGetItem(const OpTracePtr &trace) {
  TracePtr fast_trace = trace;
  TracePtr index = trace->GetParam(1);
  MS_EXCEPTION_IF_NULL(index);
  if (index->GetTraceType() != TraceType::Const) {
    return fast_trace;
  }
  TracePtr src = trace->GetParam(0);
  if (src->GetTraceType() == TraceType::Const) {
    MS_LOG(INFO) << "should be fold while graph builing, but generated a guard trace: " << std::endl
                 << trace->ToString();
    return std::make_shared<ConstTrace>(trace->GetObject(), -1);
  }
  if (src->GetTraceType() != TraceType::Operation) {
    return fast_trace;
  }
  OpTracePtr tuple = std::static_pointer_cast<OpTrace>(src);
  if (Opcode(tuple->GetOpCode()).IsBuildOp()) {
    size_t index_value = PyLong_AsSize_t(index->GetObject());
    return tuple->GetParam(index_value);
  }
  return fast_trace;
}

TracePtr FoldTupleAdd(const OpTracePtr &trace) {
  TracePtr left = trace->GetParam(0);
  TracePtr right = trace->GetParam(1);
  if (left->GetTraceType() == TraceType::Const || right->GetTraceType() == TraceType::Const) {
    MS_LOG(INFO) << "should be fold while graph builing, but generated a guard trace: " << std::endl
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
  MS_EXCEPTION_IF_NULL(input);
  if (input->GetTraceType() != TraceType::Operation) {
    return trace;
  }
  Opcode code(std::static_pointer_cast<OpTrace>(input)->GetOpCode());
  if (!code.IsBuildOp()) {
    return trace;
  }
  MS_LOG(INFO) << "should be fold while graph builing, but generated a trace: " << std::endl << trace->ToString();
  return std::make_shared<ConstTrace>(trace->GetObject(), -1);
}

// ======== base implementation =====

std::string FastTraceBase::ToString(bool include_param) {
  std::stringstream s;
  this->SimpleString(&s);
  return s.str();
}

const InfoPack &FastTraceBase::Info() {
  if (info_ == nullptr) {
    this->OpTrace::InitInfo();
    ((*info_) << src_type_).Update();
  }
  return *info_;
}

// ======== implementation  ==========

template <Py_ssize_t (&func_ref)(PyObject *o)>
py::object FastTraceBase::Len(FastTraceBase *this_p, PTraceContext context, bool perf) {
  auto param1 = this_p->GetParam(1);
  MS_EXCEPTION_IF_NULL(param1);
  py::object src_object = param1->Retrieve(context, perf);
  if (src_object.ptr() == nullptr) {
    return {};
  }
  Py_ssize_t size;
  bool type_match = Py_IS_TYPE(src_object.ptr(), this_p->src_type_);
  if (!type_match) {
    size = PyObject_Size(src_object.ptr());
  } else {
    size = func_ref(src_object.ptr());
  }
  return py::reinterpret_steal<py::object>(PyLong_FromSsize_t(size));
}

static Py_ssize_t TupleListSize(PyObject *ptr) { return Py_SIZE(ptr); }
py::object FastTraceBase::Len4TupleList(FastTraceBase *this_p, PTraceContext context, bool perf) {
  return FastTraceBase::Len<TupleListSize>(this_p, context, perf);
}
py::object FastTraceBase::Len4Dict(FastTraceBase *this_p, PTraceContext context, bool perf) {
  return FastTraceBase::Len<PyDict_Size>(this_p, context, perf);
}
py::object FastTraceBase::Len4Tensor(FastTraceBase *this_p, PTraceContext context, bool perf) {
  auto param1 = this_p->GetParam(1);
  MS_EXCEPTION_IF_NULL(param1);
  py::object src_object = param1->Retrieve(context, perf);
  return src_object.ptr() ? py::int_(PyObject_Size(src_object.ptr())) : py::object{};
}

#define ITEM_FUNC_COMMON                                                                              \
  auto param0 = this_p->GetParam(0);                                                                  \
  MS_EXCEPTION_IF_NULL(param0);                                                                       \
  py::object src_object = param0->Retrieve(context, perf);                                            \
  if (src_object.ptr() == nullptr) {                                                                  \
    return {};                                                                                        \
  }                                                                                                   \
  bool type_match = Py_IS_TYPE(src_object.ptr(), this_p->src_type_);                                  \
  bool retrieve_index = !type_match || !this_p->has_data_;                                            \
  auto param1 = this_p->GetParam(1);                                                                  \
  MS_EXCEPTION_IF_NULL(param1);                                                                       \
  py::object index_object = retrieve_index ? param1->Retrieve(context, perf) : py::object();          \
  if (retrieve_index && index_object.ptr() == nullptr) {                                              \
    return py::object();                                                                              \
  }                                                                                                   \
  if (!type_match) {                                                                                  \
    return py::reinterpret_steal<py::object>(PyObject_GetItem(src_object.ptr(), index_object.ptr())); \
  }

py::object FastTraceBase::Item4TupleList_Int(FastTraceBase *this_p, PTraceContext context, bool perf) {
  using FastTraceT = FastTrace<Py_ssize_t, decltype(Item4TupleList_Int), Item4TupleList_Int>;
  Py_ssize_t index;
  ITEM_FUNC_COMMON;
  if (this_p->has_data_) {
    index = static_cast<FastTraceT *>(this_p)->cache_;
  } else {
    index = PyLong_AsSsize_t(index_object.ptr());
  }
  Py_ssize_t size = Py_SIZE(src_object.ptr());
  PyObject **begin =
    PyTuple_Check(src_object.ptr()) ? &PyTuple_GET_ITEM(src_object.ptr(), 0) : &PyList_GET_ITEM(src_object.ptr(), 0);
  index = index < 0 ? index + size : index;
  return index < 0 || index >= size ? py::object() : py::reinterpret_borrow<py::object>(begin[index]);
}

py::object FastTraceBase::Item4Dict(FastTraceBase *this_p, PTraceContext context, bool perf) {
  using FastTraceT = FastTrace<py::object, decltype(Item4Dict), Item4Dict>;
  py::object index;
  ITEM_FUNC_COMMON;
  if (this_p->has_data_) {
    index = static_cast<FastTraceT *>(this_p)->cache_;
  } else {
    index = index_object;
  }
  PyObject *result = PyDict_GetItemWithError(src_object.ptr(), index.ptr());
  return py::reinterpret_borrow<py::object>(result);
}
#undef ITEM_FUNC_COMMON

constexpr const char *k_name_params = "_params";
constexpr const char *k_name_buffers = "_buffers";
constexpr const char *k_name_cells = "_cells";
constexpr const char *k_name_params_list = "_params_list";

struct MsCellNames {
  py::object attr_name_;
  py::object _params;
  py::object _buffers;
  py::object _cells;
  py::object _params_list;
  bool is_cls_attr_;
};

py::object MsCell__getattr__(PyObject *self, const MsCellNames &names) {
  // must be not defined __getattribute__
  PyObject *name = names.attr_name_.ptr();
  PyObject *result;
  if (names.is_cls_attr_) {
    result = PyObject_GenericGetAttr(self, name);
    if (result != nullptr) {
      return py::reinterpret_steal<py::object>(result);
    }
    PyErr_Clear();
    // unlikely
  }
  PyObject *self_dict = PyObject_GenericGetDict(self, nullptr);
  if (self_dict == nullptr) {
    // no self.__dict__
    // unlikely
    PyErr_Clear();
    return py::object();
  }
  py::object ref_scope = py::reinterpret_steal<py::object>(self_dict);
  result = PyDict_GetItem(ref_scope.ptr(), name);
  if (result) {
    return py::reinterpret_borrow<py::object>(result);
  }
  // '_params', '_buffers', '_cells', '_params_list' must be dict and subclass and not defined __getitem__
  PyObject *params = PyDict_GetItem(self_dict, names._params.ptr());
  if (params != nullptr && (result = PyDict_GetItem(params, name))) {
    return py::reinterpret_borrow<py::object>(result);
  }
  PyObject *buffers = PyDict_GetItem(self_dict, names._buffers.ptr());
  if (buffers != nullptr && (result = PyDict_GetItem(buffers, name))) {
    return py::reinterpret_borrow<py::object>(result);
  }
  PyObject *cells = PyDict_GetItem(self_dict, names._cells.ptr());
  if (cells != nullptr && (result = PyDict_GetItem(cells, name))) {
    return py::reinterpret_borrow<py::object>(result);
  }
  PyObject *params_list = PyDict_GetItem(self_dict, names._params_list.ptr());
  if (params_list != nullptr && (result = PyDict_GetItem(params_list, name))) {
    return py::reinterpret_borrow<py::object>(result);
  }
  return py::object();
}

using FastTraceAttr4MsCell = FastTrace<MsCellNames, decltype(FastTraceBase::Attr4MsCell), FastTraceBase::Attr4MsCell>;
py::object FastTraceBase::Attr4MsCell(FastTraceBase *this_p, PTraceContext context, bool perf) {
  auto param0 = this_p->GetParam(0);
  MS_EXCEPTION_IF_NULL(param0);
  py::object src_object = param0->Retrieve(context, perf);
  if (src_object.ptr() == nullptr) {
    return {};
  }
  py::object name_object;
  if (this_p->has_data_) {
    name_object = static_cast<FastTraceAttr4MsCell *>(this_p)->cache_.attr_name_;
  } else {
    auto param1 = this_p->GetParam(1);
    MS_EXCEPTION_IF_NULL(param1);
    name_object = param1->Retrieve(context, perf);
  }
  if (name_object.ptr() == nullptr) {
    return py::object();
  }
  bool type_match = Py_IS_TYPE(src_object.ptr(), this_p->src_type_);
  if (!type_match) {
    MS_LOG(DEBUG) << "self type changed: expected: " << this_p->src_type_->tp_name
                  << ", but got: " << Py_TYPE(src_object.ptr())->tp_name << std::endl
                  << this_p->ToString(false);
    // likely failed
    return py::getattr(src_object, name_object, nullptr);
  }
  return MsCell__getattr__(src_object.ptr(), static_cast<FastTraceAttr4MsCell *>(this_p)->cache_);
}

// ======== creator  ==========

TracePtr FastTraceBase::CreateFastLen(const OpTracePtr &tr) {
  TracePtr obj_trace = tr->GetParam(1);
  MS_EXCEPTION_IF_NULL(obj_trace);
  PyObject *obj_object = obj_trace->GetObject();
  bool check_func;
  // skip check global object `len` is builtin function `len`
  check_func = false;

  // subclass case
  FastTraceBasePtr new_tr;
  if (PyDict_Check(obj_object)) {
    new_tr = std::make_shared<FastTrace<EmptyData, decltype(Len4Dict), Len4Dict>>(tr);
  } else if (PyTuple_Check(obj_object) || PyList_Check(obj_object)) {
    new_tr = std::make_shared<FastTrace<EmptyData, decltype(Len4TupleList), Len4TupleList>>(tr);
  } else if (IsTensorPyObject(obj_object)) {
    new_tr = std::make_shared<FastTrace<EmptyData, decltype(Len4Tensor), Len4Tensor>>(tr);
  } else {
    return tr;
  }
  new_tr->src_type_ = Py_TYPE(obj_object);
  new_tr->has_data_ = false;
  new_tr->check_op_ = check_func;
  return new_tr;
}

TracePtr FastTraceBase::CreateFastItem(const OpTracePtr &tr) {
  bool is_call = Opcode(tr->GetOpCode()).IsCall();
  TracePtr src_trace = tr->GetParam(is_call);
  MS_EXCEPTION_IF_NULL(src_trace);
  TracePtr index_trace = tr->GetParam(1 + is_call);
  MS_EXCEPTION_IF_NULL(index_trace);
  PyObject *src_object = src_trace->GetObject();
  PyObject *index_object = index_trace->GetObject();
  auto param0 = tr->GetParam(0);
  MS_EXCEPTION_IF_NULL(param0);
  bool check_func = is_call ? param0->GetTraceType() != TraceType::Const : false;
  bool const_index = index_trace->GetTraceType() == TraceType::Const;
  if (check_func) {
    MS_LOG(DEBUG) << "got a getitem trace without constant len func";
  }
  // skip check global object `operator.getitem`
  check_func = false;

  // subclass case
  FastTraceBasePtr new_tr;
  if ((PyTuple_Check(src_object) || PyList_Check(src_object)) && PyLong_Check(index_object)) {
    auto tmp = std::make_shared<FastTrace<Py_ssize_t, decltype(Item4TupleList_Int), Item4TupleList_Int>>(tr);
    tmp->cache_ = PyLong_AsSsize_t(index_object);
    new_tr = tmp;
    new_tr->has_data_ = const_index;
  } else if (PyDict_Check(src_object)) {
    auto tmp = std::make_shared<FastTrace<py::object, decltype(Item4Dict), Item4Dict>>(tr);
    tmp->cache_ = py::reinterpret_borrow<py::object>(index_object);
    new_tr = tmp;
    new_tr->has_data_ = const_index;
  } else {
    return tr;
  }
  new_tr->src_type_ = Py_TYPE(src_object);
  new_tr->check_op_ = check_func;
  new_tr->params_ = {src_trace, index_trace};
  return new_tr;
}

static bool ValidateFastAttr4MsCell(PyTypeObject *tp) {
  if (!IsCellType<true>(tp)) {  // cpp type instance check
    return false;
  }
  PyTypeObject *base_cell = reinterpret_cast<PyTypeObject *>(py::module::import("mindspore.nn").attr("Cell").ptr());
  py::str name("__getattr__");
  return _PyType_Lookup(tp, name.ptr()) == _PyType_Lookup(base_cell, name.ptr());
}

TracePtr FastTraceBase::CreateFastAttr(const OpTracePtr &tr) {
  bool is_call = Opcode(tr->GetOpCode()).IsCall();
  TracePtr self_trace = tr->GetParam(is_call);
  PyObject *self_object = self_trace->GetObject();
  py::object name_object;
  bool has_data;
  if (is_call) {
    TracePtr name_trace = tr->GetParam(1 + is_call);
    name_object = py::reinterpret_borrow<py::object>(name_trace->GetObject());
    has_data = name_trace->IsConst();
  } else {
    name_object = py::str(tr->GetName());
    has_data = true;
  }
  bool check_func = is_call ? tr->GetParam(0)->GetTraceType() != TraceType::Const : false;
  if (check_func) {
    MS_LOG(DEBUG) << "got a getattr trace without constant getattr func";
  }
  // skip check global object `getattr`
  check_func = false;

  // subclass case
  FastTraceBasePtr new_tr;
  if (ValidateFastAttr4MsCell(Py_TYPE(self_object))) {
    auto tmp = std::make_shared<FastTraceAttr4MsCell>(tr);
    tmp->cache_ = {
      name_object, py::str(k_name_params), py::str(k_name_buffers), py::str(k_name_cells), py::str(k_name_params_list),
    };
    tmp->has_data_ = has_data;
    new_tr = tmp;
    tmp->cache_.is_cls_attr_ = _PyType_Lookup(Py_TYPE(self_object), name_object.ptr());
    if (!tmp->cache_.is_cls_attr_) {
      PyErr_Clear();
    }
  } else {
    MS_LOG(DEBUG) << Py_TYPE(self_object)->tp_name << " not a mindspore.nn.Cell or redefined __getattr__";
    return tr;
  }
  new_tr->src_type_ = Py_TYPE(self_object);
  new_tr->check_op_ = check_func;
  if (is_call) {
    new_tr->params_ = {self_trace, tr->GetParam(1 + is_call)};
  } else {
    new_tr->params_ = {self_trace};
  }
  return new_tr;
}

}  // namespace

TracePtr OpTrace::Fold() {
  OpTracePtr current = std::static_pointer_cast<OpTrace>(shared_from_this());
  if (is_fold_ || is_const_) {
    return current;
  }
  is_fold_ = true;
  for (size_t i = 0, size = this->GetParamCount(); i < size; ++i) {
    auto param = this->GetParam(i);
    MS_EXCEPTION_IF_NULL(param);
    if (param->GetTraceType() == TraceType::Operation) {
      OpTracePtr input = std::static_pointer_cast<OpTrace>(param);
      if (!input->is_fold_) {
        MS_LOG(DEBUG) << "The trace not fold: " << input->ToString();
        this->params_[i] = input->Fold();
      }
    }
  }
  FuncMapEnum func = GetFuncMapEnum(current);
  return optimize_func_map[func] == nullptr ? current : optimize_func_map[func](current);
}

void SimpleStringUnary(const OpTrace *this_p, std::ostream *s) {
  int c = this_p->GetOpCode();
  const char *op = c == UNARY_INVERT ? "~" : c == UNARY_NEGATIVE ? "-" : c == UNARY_NOT ? "(not " : "+";
  auto param0 = this_p->GetParam(0);
  MS_EXCEPTION_IF_NULL(param0);
  param0->SimpleString(&((*s) << op));
  (*s) << (c == UNARY_NOT ? ")" : "");
}
void SimpleStringBinary(const OpTrace *this_p, std::ostream *s) {
  Opcode c(this_p->GetOpCode());
  (*s) << "(";
  auto param0 = this_p->GetParam(0);
  MS_EXCEPTION_IF_NULL(param0);
  auto param1 = this_p->GetParam(1);
  MS_EXCEPTION_IF_NULL(param1);
  param0->SimpleString(s);
  (*s) << " " << c.BinaryMathString(this_p->GetOpArgs()) << " ";
  param1->SimpleString(s);
  (*s) << ")";
}
void SimpleStringCall(const OpTrace *this_p, std::ostream *s) {
  Opcode c(this_p->GetOpCode());
  auto param0 = this_p->GetParam(0);
  MS_EXCEPTION_IF_NULL(param0);
  param0->SimpleString(s);
  (*s) << "(";
  int input_size = this_p->GetParamCount();
  int i = 1;
  bool is_unpack = c == CALL_FUNCTION_EX;

  auto param_simple_string = [&this_p, &i, &s]() {
    auto param = this_p->GetParam(i++);
    MS_EXCEPTION_IF_NULL(param);
    param->SimpleString(s);
  };

  if (is_unpack) {
    (*s) << "*";
    param_simple_string();
    if (input_size > i) {
      (*s) << ", **";
      param_simple_string();
    }
  } else {
    for (; i < input_size; ++i) {
      param_simple_string();
      (*s) << ",";
    }
    if (input_size) {
      s->seekp(-1, std::ios_base::cur);
    }
  }
  (*s) << ")";
}
void SimpleStringSubscript(const OpTrace *this_p, std::ostream *s) {
  auto param0 = this_p->GetParam(0);
  auto param1 = this_p->GetParam(1);
  MS_EXCEPTION_IF_NULL(param0);
  MS_EXCEPTION_IF_NULL(param1);
  param0->SimpleString(s);
  (*s) << '[';
  param1->SimpleString(s);
  (*s) << ']';
}
void SimpleStringAttr(const OpTrace *this_p, std::ostream *s) {
  auto param0 = this_p->GetParam(0);
  MS_EXCEPTION_IF_NULL(param0);
  param0->SimpleString(s);
  (*s) << '.' << this_p->GetName();
}

void OpTrace::SimpleString(std::ostream *s) const {
  Opcode op(opcode_);
  if (op.IsUnaryMath()) {
    SimpleStringUnary(this, s);
  } else if (op.IsBinaryMath()) {
    SimpleStringBinary(this, s);
  } else if (op.IsCall()) {
    SimpleStringCall(this, s);
  } else if (op == BINARY_SUBSCR) {
    SimpleStringSubscript(this, s);
  } else if (op == LOAD_ATTR) {
    SimpleStringAttr(this, s);
  } else {
    (*s) << op.name() << "(";
    for (const auto &i : this->params_) {
      i->SimpleString(s);
      (*s) << ",";
    }
    s->seekp(-1, std::ios_base::cur);
    (*s) << ")";
  }
}

}  // namespace pijit
}  // namespace mindspore
