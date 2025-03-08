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
#include "pipeline/jit/pi/graph_capture/abstract_object.h"
#include <algorithm>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "utils/log_adapter.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/pi/python_adapter/pydef.h"
#include "pipeline/jit/pi/python_adapter/py_code.h"
#include "pipeline/jit/pi/graph_guard/infer.h"
#include "pipeline/jit/pi/graph_compiler/utils.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "include/common/utils/convert_utils_py.h"
#include "pipeline/jit/pi/utils/opcode_declare.h"
#include "include/common/utils/tensor_py.h"

namespace mindspore {
namespace pijit {
static const size_t DictStep = 2;
constexpr size_t kValueToStringLimit = 120;

#define FIND_MAP_CACHE(map, target) \
  do {                              \
    auto iter = (map).find(target); \
    if (iter != (map).end()) {      \
      return iter->second;          \
    }                               \
  } while (0)

#ifdef DEBUG
#define CHECK_PYTHON_EXCEPTION(check_res)       \
  if (PyErr_Occurred()) {                       \
    MS_LOG(DEBUG) << "has an python exception"; \
    MS_ASSERT((check_res) == nullptr);          \
    PyErr_Print();                              \
    PyErr_Clear();                              \
  }
#else
#define CHECK_PYTHON_EXCEPTION(check_res) PyErr_Clear()
#endif

// mindspore graph can accept these value
static const std::set<AObject::Type> kMsSupportedType = {
  AObject::kTypeInt,  AObject::kTypeBool,   AObject::kTypeFloat,
  AObject::kTypeNone, AObject::kTypeString, AObject::kTypeTensor,
};

std::vector<AbstractObjectBase::Resource *> AbstractObjectBase::Resource::weak_this_;

AbstractObjectBase::Resource::Resource() : pool_(__FILE__, __LINE__, "AObject") {
  MS_EXCEPTION_IF_CHECK_FAIL(weak_this_.empty(), "can't reentrant");
  weak_this_.push_back(this);
}
AbstractObjectBase::Resource::~Resource() {
  MS_EXCEPTION_IF_CHECK_FAIL(weak_this_.size() == 1, "can't reentrant");
  Release();
  weak_this_.pop_back();
}

std::unordered_map<AObject::Type, PyTypeObject *> AbstractObjectBase::aobj_type_map = {
  {AObject::kTypeFunction, &PyFunction_Type}, {AObject::kTypeBoundMethod, &PyMethod_Type},
  {AObject::kTypeCodeObject, &PyCode_Type},   {AObject::kTypeSlice, &PySlice_Type},
  {AObject::kTypeSet, &PySet_Type},           {AObject::kTypeSet, &PyFrozenSet_Type},
  {AObject::kTypeBool, &PyBool_Type},         {AObject::kTypeFloat, &PyFloat_Type},
  {AObject::kTypeInt, &PyLong_Type},          {AObject::kTypeList, &PyList_Type},
  {AObject::kTypeTuple, &PyTuple_Type},       {AObject::kTypeNamedTuple, &PyTuple_Type},
  {AObject::kTypeDict, &PyDict_Type},         {AObject::kTypeDictValues, &PyDictValues_Type},
  {AObject::kTypeDictKeys, &PyDictKeys_Type}, {AObject::kTypeDictItems, &PyDictItems_Type},
  {AObject::kTypeType, &PyType_Type},         {AObject::kTypeString, &PyUnicode_Type},
  {AObject::kTypeModule, &PyModule_Type},     {AObject::kTypeCFunction, &PyCFunction_Type},
  {AObject::kTypeAnyValue, nullptr},
};

// exact equal check
static const std::unordered_map<PyTypeObject *, AObject::Type> exact_type_map = {
  {&PyFunction_Type, AObject::kTypeFunction},
  {&PyMethod_Type, AObject::kTypeBoundMethod},
  {&PyCode_Type, AObject::kTypeCodeObject},
  {&PySlice_Type, AObject::kTypeSlice},
  {&PySet_Type, AObject::kTypeSet},
  {&PyFrozenSet_Type, AObject::kTypeSet},
  {&PyBool_Type, AObject::kTypeBool},
  {&PyFloat_Type, AObject::kTypeFloat},
  {&PyLong_Type, AObject::kTypeInt},
  {&PyList_Type, AObject::kTypeList},
  {&PyTuple_Type, AObject::kTypeTuple},
  {&PyDict_Type, AObject::kTypeDict},
  {&PyDictValues_Type, AObject::kTypeDictValues},
  {&PyDictKeys_Type, AObject::kTypeDictKeys},
  {&PyDictItems_Type, AObject::kTypeDictItems},
  {&PyType_Type, AObject::kTypeType},
  {&PyUnicode_Type, AObject::kTypeString},
  {&PyModule_Type, AObject::kTypeModule},
  {&PyCFunction_Type, AObject::kTypeCFunction},
  {nullptr, AObject::kTypeAnyValue},
};

// shouldn't add nullptr to this map
static const std::unordered_map<PyObject *, AObject::Type> const_object_type_map = {
  {Py_Ellipsis, AObject::kTypeEllipsis},
  {Py_None, AObject::kTypeNone},
  {Py_True, AObject::kTypeBool},
  {Py_False, AObject::kTypeBool},
};

static const std::vector<std::pair<PyTypeObject *, AObject::Type>> sub_type_map = {
  {&PyModule_Type, AObject::kTypeModule}, {&PyCFunction_Type, AObject::kTypeCFunction}};

constexpr size_t fast_type_mask = Py_TPFLAGS_LONG_SUBCLASS | Py_TPFLAGS_LIST_SUBCLASS | Py_TPFLAGS_TUPLE_SUBCLASS |
                                  Py_TPFLAGS_UNICODE_SUBCLASS | Py_TPFLAGS_DICT_SUBCLASS | Py_TPFLAGS_TYPE_SUBCLASS;

const char *AbstractObjectBase::GetTypeDesc(AObject::Type type) {
#define ABSTRACT_TYPE_DEF(unit)       \
  if (type == AObject::kType##unit) { \
    return "kType" #unit;             \
  }
#include "abstract_type_kind.def"
#undef ABSTRACT_TYPE_DEF
  return "unknown type";
}

bool AbstractObjectBase::IsMindSporeSupportedType() {
  return kMsSupportedType.find(GetType()) != kMsSupportedType.end();
}

static void PrintPyObject(std::ostream *out_s, const py::handle &obj, bool print_type) {
  auto &s = *out_s;
  PyObject *op = obj.ptr();
  AObject::Type t = AObject::GetPyType(obj.ptr());
  switch (t) {
    case AObject::kTypeTensor:
    case AObject::kTypeStubTensor:
      s << "Tensor'" << std::string(py::str(obj.attr("shape"))) << ", " << std::string(py::str(obj.attr("dtype")))
        << "'";
      break;
    case AObject::kTypeBoundMethod:
      s << "<bound method " << AbstractObjectBase::ToString(PyMethod_GET_FUNCTION(op)) << " of "
        << AbstractObjectBase::ToString(PyMethod_GET_SELF(op), print_type) << ">";
      break;
    case AObject::kTypeNNCellList:
    case AObject::kTypeList:
    case AObject::kTypeTuple:
      s << (t == AObject::kTypeTuple ? "(" : "[");
      for (auto i : py::iter(obj)) {
        s << AbstractObjectBase::ToString(i.ptr(), print_type) << ",";
      }
      s << (t == AObject::kTypeTuple ? ")" : "]");
      break;
    case AObject::kTypeDict: {
      PyObject *key;
      PyObject *val;
      Py_ssize_t pos = 0;
      s << "{";
      while (PyDict_Next(op, &pos, &key, &val)) {
        s << AbstractObjectBase::ToString(key, print_type) << ":" << AbstractObjectBase::ToString(val, print_type)
          << ",";
      }
      s << "}";
      break;
    }
    case AObject::kTypeCell:
      s << (Py_TYPE(op)->tp_name ? Py_TYPE(op)->tp_name : "<unnamed>") << " object at " << op;
      break;
    default:
      s << std::string(py::str(obj));
      break;
  }
}

std::string AbstractObjectBase::ToString(PyObject *op, bool print_type, size_t limit) {
  if (op == nullptr) {
    return "<NULL>";
  }
  ReprRecursionScope scope(op);
  if (scope.ReEnter()) {
    return "...";
  }

  std::stringstream s;
  if (print_type) {
    s << (Py_TYPE(op)->tp_name ? Py_TYPE(op)->tp_name : "<unnamed>") << "{";
  }
  PrintPyObject(&s, op, print_type);
  s << (print_type ? "}" : "");
  auto ret = s.str();
  return ret.size() < limit ? ret : ret.substr(0, limit) + "...";
}

std::string AbstractObjectBase::ToString() const {
  std::stringstream s;
#define ABSTRACT_MS_FLAG_DEF(unit, bit) s << ((ms_flag_ & kMsFlag##unit) ? #unit "|" : "");
#include "abstract_ms_flag.def"
#undef ABSTRACT_MS_FLAG_DEF
  if (ms_flag_) {
    s.seekp(-1, s.cur);
  }
  if (type_object_ != nullptr) {
    s << (type_object_->tp_name ? type_object_->tp_name : "<unnamed>");
  } else {
    s << GetTypeDesc(type_);
  }
  return s.str();
}

AObject *AbstractObjectBase::GetLatestVersion() {
  if (next_version_ == nullptr) {
    return this;
  }
  auto latest = next_version_;
  while (latest->next_version_ != nullptr) {
    latest = latest->next_version_;
  }
  return latest;
}

void AbstractObjectBase::SetPreVersion(AObject *pre_version) {
  MS_EXCEPTION_IF_CHECK_FAIL(pre_version_ == nullptr, "Try to overwrite a version.");
  pre_version_ = pre_version;
  MS_EXCEPTION_IF_CHECK_FAIL(pre_version->next_version_ == nullptr, "Try to change a next-version.");
  pre_version->next_version_ = this;
  // Notify this's user to update version
  for (const auto &user : users_) {
    user->CreateVersionWithNewValue();
  }
}

void AbstractObjectBase::SetNextVersion(AObject *next_version) {
  MS_EXCEPTION_IF_CHECK_FAIL(next_version_ == nullptr, "Try to overwrite a version.");
  next_version_ = next_version;
  MS_EXCEPTION_IF_CHECK_FAIL(next_version->pre_version_ == nullptr, "Try to change a pre-version.");
  next_version->pre_version_ = this;
  // Notify this's user to update version
  for (const auto &user : users_) {
    user->CreateVersionWithNewValue();
  }
}

const AObject *AbstractObjectBase::GetBaseVersion() const {
  if (pre_version_ == nullptr) {
    return this;
  }
  auto pre_version = pre_version_;
  while (pre_version->pre_version_ != nullptr) {
    pre_version = pre_version->pre_version_;
  }
  return pre_version;
}

std::string AbstractObject::ToString() const {
  std::stringstream s;
  s << AbstractObjectBase::ToString();
  if (value_.ptr() != nullptr) {
    s << "{value=" << AObject::ToString(value_.ptr(), false, kValueToStringLimit) << "}";
  }
  return s.str();
}

AbstractObjectBase::Type AbstractObjectBase::GetPyType(PyTypeObject *tp) {
  if (tp == nullptr) {
    return kTypeAnyValue;
  }
  FIND_MAP_CACHE(exact_type_map, tp);
  // fast sub type check
  // __builtin_clz(tp->tp_flags & fast_type_mask), or std::countl_zero
  /**
   * sub-class int, float, list, tuple, str, is mindspore unsupported
   */
  switch (tp->tp_flags & fast_type_mask) {
    case Py_TPFLAGS_TUPLE_SUBCLASS:
      return AbstractNamedTuple::IsNamedTuple(tp) ? kTypeNamedTuple : kTypeAnyValue;
    case Py_TPFLAGS_LONG_SUBCLASS:
    case Py_TPFLAGS_LIST_SUBCLASS:
    case Py_TPFLAGS_UNICODE_SUBCLASS:
    case Py_TPFLAGS_DICT_SUBCLASS:
      return kTypeAnyValue;
    case Py_TPFLAGS_TYPE_SUBCLASS:
      return kTypeType;
    default:
      break;
  }
  // sub type check
  for (auto &i : sub_type_map) {
    if (PyType_IsSubtype(tp, i.first)) {
      return i.second;
    }
  }
  return GetMsType(tp);
}

AbstractObjectBase::Type AbstractObjectBase::GetPyType(PyObject *o) {
  if (o == nullptr) {
    return kTypeAnyValue;
  }
  FIND_MAP_CACHE(const_object_type_map, o);
  if (PyLong_Check(o)) {
    return (Py_ABS(Py_SIZE(o)) > 2) ? kTypeAnyValue : kTypeInt;
  }
  return GetPyType(Py_TYPE(o));
}

AbstractObjectBase::Type AbstractObjectBase::GetMsType(PyTypeObject *tp) {
  static const std::vector<std::pair<bool (*)(PyTypeObject *), AObject::Type>> match_func = {
    {IsStubTensorType<true>, kTypeStubTensor}, {IsTensorType<true>, kTypeTensor},
    {IsCellListType<false>, kTypeNNCellList},  {IsCellType<true>, kTypeCell},
    {IsPrimitiveType<true>, kTypePrimitive},   {IsMetaFuncGraphType<true>, kTypeMetaFuncGraph},
    {IsMSDTypeType<true>, kTypeMSDType},       {IsPrimitiveFunctionType<true>, kTypePrimitiveFunction},
  };
  if (tp == nullptr) {
    return kTypeAnyValue;
  }
  for (auto i : match_func) {
    if (i.first(tp)) {
      return i.second;
    }
  }
  return kTypeAnyValue;
}

AObject *AbstractObjectBase::TryConvertDynamicLengthSequence(const abstract::AbstractBasePtr &abstract) {
  if (abstract->isa<abstract::AbstractTuple>() && abstract->cast<abstract::AbstractSequencePtr>()->dynamic_len()) {
    return MakeAObject(kTypeTuple);
  }
  if (abstract->isa<abstract::AbstractList>() && abstract->cast<abstract::AbstractSequencePtr>()->dynamic_len()) {
    return MakeAObject(kTypeList);
  }
  return nullptr;
}

AObject *AbstractObjectBase::Convert(const abstract::AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return MakeAObject(kTypeAnyValue);
  }
  if (auto ret = TryConvertDynamicLengthSequence(abstract); ret) {
    return ret;
  }
  py::object res = AbstractWrapper::ConvertToPyObject(abstract);
  if (res.ptr() != nullptr) {
    return Convert(res.ptr());
  }

  if (abstract->isa<abstract::AbstractSequence>()) {
    auto abstract_seq = abstract->cast<abstract::AbstractSequencePtr>();
    const auto &elements = abstract_seq->elements();
    std::vector<AObject *> items;
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(items),
                         [](const auto &e) { return Convert(e); });
    if (abstract->isa<abstract::AbstractTuple>()) {
      return MakeAObject(kTypeTuple, &PyTuple_Type, nullptr, items);
    }
    return MakeAObject(kTypeList, &PyList_Type, nullptr, items);
  }

  if (abstract->isa<abstract::AbstractDictionary>()) {
    auto abstract_dict = abstract->cast<abstract::AbstractDictionaryPtr>();
    const auto &elements = abstract_dict->elements();
    std::vector<AObject *> key_values;
    std::for_each(elements.begin(), elements.end(), [&key_values](const auto &element) {
      key_values.push_back(Convert(element.first));
      key_values.push_back(Convert(element.second));
    });
    return MakeAObject(kTypeDict, &PyDict_Type, nullptr, key_values);
  }

  if (!abstract->isa<abstract::AbstractScalar>()) {
    return MakeAObject(kTypeAnyValue, nullptr, nullptr);
  }
  auto type_id = abstract->BuildType()->type_id();
  MS_LOG(INFO) << "Current type_id is " << TypeIdToString(type_id);
  switch (type_id) {
    case kNumberTypeInt:
    case kNumberTypeInt64:
    case kNumberTypeInt32:
      return MakeAObject(kTypeInt, &PyLong_Type, nullptr);
    case kNumberTypeFloat:
    case kNumberTypeFloat16:
    case kNumberTypeFloat32:
    case kNumberTypeFloat64:
      return MakeAObject(kTypeFloat, &PyFloat_Type, nullptr);
    case kNumberTypeBool:
      return MakeAObject(kTypeBool, &PyBool_Type, nullptr);
    default:
      return MakeAObject(kTypeAnyValue, nullptr, nullptr);
  }
}

AObject *AbstractObjectBase::Convert(const AbstractWrapperPtr &wrapper) {
  if (wrapper == nullptr) {
    return Resource::Current()->pool()->New<AbstractObjectBase>(kTypeAnyValue);
  }
  return Convert(wrapper->abstract());
}

AObject *AbstractObjectBase::MakeAObject(AObject::Type type, PyTypeObject *tp, PyObject *o,
                                         const std::vector<AObject *> &elements) {
  MS_EXCEPTION_IF_CHECK_FAIL(Resource::Current() != nullptr, "can't take resource");
  MS_EXCEPTION_IF_CHECK_FAIL(tp == nullptr || o == nullptr || Py_TYPE(o) == tp, "check type match value");
  py::object h = py::cast<py::object>(o);
  const auto &obj_map = Resource::Current()->GetObjMap();
  if (o != nullptr && obj_map.find(o) != obj_map.end()) {
    return obj_map.at(o);
  }
  MS_LOG(INFO) << "Create AbstractObject " << GetTypeDesc(type) << " Start...";
  AObject *res;
  switch (type) {
    case kTypeStubTensor:
    case kTypeTensor:
      res = Resource::Current()->pool()->New<AbstractTensor>(h, type == kTypeStubTensor);
      break;
    case kTypeType:
      res = Resource::Current()->pool()->New<AbstractType>(h);
      break;
    case kTypeString:
      res = Resource::Current()->pool()->New<AbstractString>(h);
      break;
    case kTypeNNCellList:
      res = Resource::Current()->pool()->New<AbstractSequence>(kTypeNNCellList, h);
      break;
    case kTypeList:
      if (o != nullptr) {
        res = Resource::Current()->pool()->New<AbstractList>(h);
      } else {
        res = Resource::Current()->pool()->New<AbstractList>(elements);
      }
      break;
    case kTypeTuple:
      if (o != nullptr) {
        res = Resource::Current()->pool()->New<AbstractTuple>(h);
      } else {
        res = Resource::Current()->pool()->New<AbstractTuple>(elements);
      }
      break;
    case kTypeNamedTuple:
      res = Resource::Current()->pool()->New<AbstractNamedTuple>(h, tp);
      break;
    case kTypeDict:
      if (o != nullptr) {
        res = Resource::Current()->pool()->New<AbstractDict>(h);
      } else {
        res = Resource::Current()->pool()->New<AbstractDict>(elements);
      }
      break;
    case kTypeAnyValue:
      if (o == nullptr) {
        res = Resource::Current()->pool()->New<AbstractObjectBase>(kTypeAnyValue, tp);
        break;
      }
      // fall-through
    default:
      // known type
      res = Resource::Current()->pool()->New<AbstractObject>(type, h);
      break;
  }
  Resource::Current()->AddVobj(h, res);
  MS_LOG(INFO) << "Create AbstractObject " << res << " End. The AObj is " << res->ToString();
  return res;
}

AObject *AbstractObjectBase::MakeFunction(const std::vector<AObject *> &args, const py::object &globals, int oparg) {
  std::vector<py::object> pyarg;
  std::transform(args.begin(), args.end(), std::back_inserter(pyarg), [](AObject *i) { return i->GetPyObject(); });
  auto iter = pyarg.end() - 1;
  PyObject *qualname = (*iter--).ptr();
  PyObject *code = (*iter--).ptr();
  py::object f_handle = py::reinterpret_steal<py::object>(PyFunction_NewWithQualName(code, globals.ptr(), qualname));
  PyFunctionObject *func = reinterpret_cast<PyFunctionObject *>(f_handle.ptr());
  MS_EXCEPTION_IF_CHECK_FAIL(func, "MAKE_FUNCTION failed");
  if (IntToSize(oparg) & 0x08) {
    func->func_closure = (*iter--).inc_ref().ptr();
    Py_ssize_t nfrees = PyCodeWrapper(code).FreeVarsSize();
    bool is_valid = func->func_closure && nfrees == PyTuple_GET_SIZE(func->func_closure);
    MS_EXCEPTION_IF_CHECK_FAIL(is_valid, "must be has python objects, and it is tuple of cell objects");
  }
  if (IntToSize(oparg) & 0x04) {
    func->func_annotations = (*iter--).inc_ref().ptr();
    MS_EXCEPTION_IF_CHECK_FAIL(func->func_annotations, "must be has python objects, and it is const key map");
  }
  if (IntToSize(oparg) & 0x02) {
    func->func_kwdefaults = (*iter--).inc_ref().ptr();
    MS_EXCEPTION_IF_CHECK_FAIL(func->func_kwdefaults, "must be has python objects, and it is const key map");
  }
  if (IntToSize(oparg) & 0x01) {
    func->func_defaults = (*iter--).inc_ref().ptr();
    MS_EXCEPTION_IF_CHECK_FAIL(func->func_defaults, "must be has python objects, and it is const tuple");
  }
  AObject *res = AObject::Convert(f_handle);
  return res;
}

py::object AbstractObjectBase::BuildOperations(const std::vector<py::object> &args, int opcode) {
  PyObject *res = nullptr;
  PyObject **tmp;
  std::vector<PyObject *> arr;
  if (opcode == BUILD_SLICE) {
    res = PySlice_New(args[0].ptr(), args[1].ptr(), args.size() > 2 ? args[2].ptr() : nullptr);
  } else if (opcode == BUILD_STRING) {
    std::transform(args.begin(), args.end(), std::back_inserter(arr), [](const py::object &o) { return o.ptr(); });
    res = _PyUnicode_JoinArray(py::str().ptr(), arr.data(), arr.size());
  } else if (opcode == BUILD_SET) {
    res = PySet_New(nullptr);
    (void)std::find_if(args.begin(), args.end(), [&res](const py::object &i) { return PySet_Add(res, i.ptr()); });
  } else if (opcode == BUILD_LIST) {
    res = PyList_New(args.size());
    tmp = &PyList_GET_ITEM(res, 0);
    std::for_each(args.begin(), args.end(), [&tmp](const py::object &i) { return *(tmp++) = i.inc_ref().ptr(); });
  } else if (opcode == BUILD_TUPLE) {
    res = PyTuple_New(args.size());
    tmp = &PyTuple_GET_ITEM(res, 0);
    std::for_each(args.begin(), args.end(), [&tmp](const py::object &i) { return *(tmp++) = i.inc_ref().ptr(); });
  } else if (opcode == BUILD_CONST_KEY_MAP) {
    res = PyDict_New();
    // must be tuple, here has a cast check
    tmp = &PyTuple_GET_ITEM(args.back().ptr(), 0);
    (void)std::find_if(args.begin(), args.end() - 1, [&res, &tmp](const py::object &i) {
      return PyDict_SetItem(res, *(tmp++), i.ptr());  // break if err_ocurred
    });
  } else if (opcode == BUILD_MAP) {
    res = PyDict_New();
    for (size_t i = 0; !PyErr_Occurred() && i < args.size(); i += 2) {
      PyDict_SetItem(res, args[i].ptr(), args[i + 1].ptr());
    }
  }
  if (PyErr_Occurred()) {
    Py_XDECREF(res);
    MS_LOG(DEBUG) << "build operation failed: " << Opcode(opcode).name();
    PyErr_Clear();
    res = nullptr;
  }
  return py::reinterpret_steal<py::object>(res);
}

AObject *AbstractObjectBase::BuildOperations(const std::vector<AObject *> &inputs, int opcode) {
  AObject *res = nullptr;
  if (opcode == BUILD_LIST || opcode == BUILD_TUPLE) {
    auto type = opcode == BUILD_LIST ? kTypeList : kTypeTuple;
    auto tp = opcode == BUILD_LIST ? &PyList_Type : &PyTuple_Type;
    res = MakeAObject(type, tp, nullptr, inputs);
  } else if (opcode == BUILD_CONST_KEY_MAP) {
    auto keys = inputs.back()->GetPyObject().ptr();
    std::vector<AObject *> key_values;
    for (size_t index = 0; index < inputs.size() - 1; index++) {
      key_values.push_back(Convert(PyTuple_GET_ITEM(keys, index)));
      key_values.push_back(inputs[index]);
    }
    res = MakeAObject(kTypeDict, &PyDict_Type, nullptr, key_values);
  } else if (opcode == BUILD_MAP) {
    res = MakeAObject(kTypeDict, &PyDict_Type, nullptr, inputs);
  } else if (opcode == BUILD_STRING) {
    res = MakeAObject(kTypeString);
  } else if (opcode == BUILD_SLICE) {
    res = MakeAObject(kTypeSlice);
  } else if (opcode == BUILD_SET) {
    res = MakeAObject(kTypeSet);
  } else {
    return MakeAObject(kTypeAnyValue);
  }
  return res;
}

AObject *AbstractObjectBase::MergeOperations(AObject *container, std::vector<AObject *> args, int opcode) {
  Type type = container ? container->GetType() : kTypeAnyValue;
  bool success = false;
  if (opcode == LIST_EXTEND) {
    success = type == kTypeList && (static_cast<AbstractList *>(container))->ListExtend(args[0]);
  } else if (opcode == LIST_APPEND) {
    success = type == kTypeList && (static_cast<AbstractList *>(container))->ListAppend(args[0]);
  } else if (opcode == DICT_MERGE) {
    success = type == kTypeDict && (static_cast<AbstractDict *>(container))->DictMerge(args[0]);
  } else if (opcode == DICT_UPDATE) {
    success = type == kTypeDict && (static_cast<AbstractDict *>(container))->DictUpdate(args[0]);
  } else if (opcode == MAP_ADD) {
    success = type == kTypeDict && (static_cast<AbstractDict *>(container))->MapAdd(args[0], args[1]);
  } else if (opcode == SET_UPDATE || opcode == SET_ADD) {
    success = true;
    container = MakeAObject(kTypeSet);
  }
  if (!success) {
    return MakeAObject(kTypeAnyValue);
  }
  return container;
}

AObject *AbstractObject::GetIter() const {
  if (this->GetType() == kTypeAnyValue || value_.ptr() == nullptr) {
    return MakeAObject(kTypeAnyValue);
  }
  PyObject *iter = PyObject_GetIter(value_.ptr());
  CHECK_PYTHON_EXCEPTION(iter);
  AObject *res = Convert(iter);
  Py_XDECREF(iter);
  return res;
}

AObject *AbstractObjectBase::GetAttr(const std::string &name) {
  PyTypeObject *tp = type_object_;
  if (tp == nullptr) {
    return MakeAObject(kTypeAnyValue);
  }
  py::str name_obj(name);
  PyObject *attr_obj = PyObject_GetAttr(reinterpret_cast<PyObject *>(tp), name_obj.ptr());
  if (attr_obj == nullptr) {
    PyErr_Clear();
    return MakeAObject(kTypeAnyValue);
  }
  AObject *attr = AObject::Convert(attr_obj);
  Py_DECREF(attr_obj);

  // look up mro, borrowed
  PyObject *descr = _PyType_Lookup(tp, name_obj.ptr());
  if (descr) {
    // check @staticmethod and @classmethod
    if (Py_IS_TYPE(descr, &PyStaticMethod_Type) || Py_IS_TYPE(descr, &PyClassMethod_Type)) {
      // attr not modify
    } else if (PyFunction_Check(descr)) {
      MS_EXCEPTION_IF_CHECK_FAIL(attr_obj == descr, "unknown user defined descriptor");
      PyObject *meth = PyMethod_New(descr, Py_None);
      AObject *m = AObject::Convert(meth);
      Py_DECREF(meth);
      m->SetAttr("__self__", this);
      m->SetAttr("__func__", attr);
      attr = m;
    } else {
      // other type
      attr = MakeAObject(kTypeAnyValue);
    }
  }
  return attr;
}

AObject *AbstractObject::GetAttr(const std::string &name) {
  FIND_MAP_CACHE(attrs_, name);
  AObject *res = nullptr;
  if (value_.ptr() != nullptr) {
    PyObject *attr = PyObject_GetAttrString(value_.ptr(), name.c_str());
    CHECK_PYTHON_EXCEPTION(attr);
    res = Convert(attr);
    Py_XDECREF(attr);
  } else {
    res = this->AbstractObjectBase::GetAttr(name);
  }
  attrs_[name] = res;
  return res;
}

bool AbstractObject::SetAttr(const std::string &n, AObject *v) {
  attrs_[n] = v ? v : MakeAObject(kTypeAnyValue);
  return true;
}

AObject *AbstractObject::GetItem(AObject *k) {
  PyObject *s = this->GetPyObject().ptr();
  PyObject *i = k ? k->GetPyObject().ptr() : nullptr;
  PyObject *t = nullptr;
  if (s != nullptr && i != nullptr && k->GetType() != kTypeAnyValue) {
    t = PyObject_GetItem(s, i);
    CHECK_PYTHON_EXCEPTION(t);
  }
  AObject *res = Convert(t);
  res->AddUser(this);
  Py_XDECREF(t);
  return res;
}

AObject *AbstractObject::UnaryValue(int op) const {
  PyObject *res = nullptr;
  if (op == UNARY_POSITIVE) {
    res = PyNumber_Positive(value_.ptr());
  } else if (op == UNARY_NEGATIVE) {
    res = PyNumber_Negative(value_.ptr());
  } else if (op == UNARY_INVERT) {
    res = PyNumber_Invert(value_.ptr());
  } else if (op == UNARY_NOT) {
    int err = PyObject_IsTrue(value_.ptr());
    res = err > 0 ? Py_False : (err == 0 ? Py_True : nullptr);
  }
  CHECK_PYTHON_EXCEPTION(res);
  AObject *ret = Convert(res);
  Py_XDECREF(res);
  return ret;
}

AObject *AbstractObject::Unary(int op) const {
  if (this->GetType() == kTypeAnyValue) {
    return MakeAObject(kTypeAnyValue);
  }
  if (value_.ptr() != nullptr) {
    return UnaryValue(op);
  }
  Type res_type = kTypeAnyValue;
  Type type = this->GetType();
  if (op == UNARY_POSITIVE || op == UNARY_NEGATIVE || op == UNARY_INVERT) {
    if (type == kTypeBool || type == kTypeInt) {
      res_type = kTypeInt;
    } else if (type == kTypeFloat) {
      res_type = kTypeFloat;
    }
  } else if (op == UNARY_NOT) {
    bool is_num = type == kTypeBool || type == kTypeInt || type == kTypeFloat;
    if (is_num || type == kTypeList || type == kTypeTuple || type == kTypeDict) {
      res_type = kTypeBool;
    }
  }
  return MakeAObject(res_type);
}

static PyObject *BinaryPow(PyObject *base, PyObject *exp) { return PyNumber_Power(base, exp, Py_None); }
static PyObject *InplacePow(PyObject *base, PyObject *exp) { return PyNumber_InPlacePower(base, exp, Py_None); }

AObject *AbstractString::GetItem(AObject *index) {
  MS_EXCEPTION_IF_NULL(index);
  auto subscript = Utils::FormatSubscript(index->GetPyObject(), str_.size());
  if (subscript.empty()) {
    return AObject::MakeAObject(kTypeAnyValue);
  }
  if ((subscript[0] + subscript[2]) >= SizeToInt(str_.size())) {
    MS_LOG(ERROR) << "The range should be in [0, " << str_.size() << "), but got [" << subscript[0] << ", "
                  << (subscript[0] + subscript[2]) << ").";
    return AObject::MakeAObject(kTypeAnyValue);
  }
  return Convert(py::str(str_.substr(subscript[0], subscript[2])).ptr());
}

static AObject::Type BinaryIntOp(AObject::Type l, AObject::Type r) {
  AObject::Type type = AObject::kTypeAnyValue;
  switch (l) {
    case AObject::kTypeInt:
    case AObject::kTypeBool:
      if (r == AObject::kTypeInt || r == AObject::kTypeBool) {
        type = AObject::kTypeInt;
      }
      break;
    default:
      break;
  }
  return type;
}

// operator '&', '^', '|'
static AObject::Type NumberLogic(AObject::Type l, AObject::Type r) {
  AObject::Type type = AObject::kTypeAnyValue;
  if (l == AObject::kTypeBool) {
    if (r == AObject::kTypeInt || r == AObject::kTypeBool) {
      type = r;
    }
  } else {
    type = BinaryIntOp(l, r);
  }
  return type;
}

// operator '+', '-', '*', '/', '%', '**', '//'
static AObject::Type NumberArithmetic(AObject::Type l, AObject::Type r) {
  AObject::Type type = AObject::kTypeAnyValue;
  if (l == AObject::kTypeFloat || r == AObject::kTypeFloat) {
    if (l == AObject::kTypeInt || l == AObject::kTypeBool || r == AObject::kTypeInt || r == AObject::kTypeBool) {
      type = AObject::kTypeFloat;
    }
  } else {
    type = BinaryIntOp(l, r);
  }
  return type;
}

static AObject::Type BinaryAdd(AObject::Type l, AObject::Type r) {
  AObject::Type type = AObject::kTypeAnyValue;
  switch (l) {
    case AObject::kTypeTuple:
    case AObject::kTypeList:
    case AObject::kTypeString:
      if (r == l) {
        type = l;
      }
      break;
    default:
      type = NumberArithmetic(l, r);
      break;
  }
  return type;
}

static AObject::Type BinaryInferDefault(AObject::Type, AObject::Type) { return AObject::kTypeAnyValue; }

static int CheckConstantIs(PyObject *a, PyObject *b, bool const_a, bool const_b) {
  // all is const object
  if (const_a && const_b) {
    return a == b;
  }
  // unknown type
  if (a == nullptr || b == nullptr) {
    return -1;
  }
  // type is type
  if (PyType_Check(a) && PyType_Check(b)) {
    return a == b;
  }
  // type not match
  if (Py_TYPE(a) != Py_TYPE(b)) {
    return false;
  }
  if (const_a || const_b) {
    return false;
  }
  return -1;
}

int AObject::BinaryIs(AObject *l, AObject *r) {
  PyObject *a = l ? l->GetPyObject().ptr() : nullptr;
  PyObject *b = r ? r->GetPyObject().ptr() : nullptr;
  const auto &map = const_object_type_map;
  bool const_a = map.find(a) != map.end();
  bool const_b = map.find(b) != map.end();
  int constant = CheckConstantIs(a, b, const_a, const_b);
  if (constant != -1) {
    return constant;
  }
  // a const object and a unknown object, but known it's type
  if (const_a && r != nullptr && r->GetType() != AObject::kTypeAnyValue && r->GetType() != AObject::kTypeBool) {
    MS_EXCEPTION_IF_CHECK_FAIL(!const_b, "shouldn't reach here");
    return false;
  }
  if (const_b && l != nullptr && l->GetType() != AObject::kTypeAnyValue && l->GetType() != AObject::kTypeBool) {
    MS_EXCEPTION_IF_CHECK_FAIL(!const_a, "shouldn't reach here");
    return false;
  }
  return -1;
}

int AObject::BinaryContains(AObject *l, AObject *r) {
  PyObject *o = l->GetPyObject().ptr();
  PyObject *c = r->GetPyObject().ptr();
  if (c == nullptr || o == nullptr || r->GetType() == AObject::kTypeAnyValue) {
    return -1;
  }
  int res = PySequence_Contains(c, o);
  CHECK_PYTHON_EXCEPTION(res < 0 ? nullptr : Py_True);
  return res;
}

AObject *BinaryIs(AObject *l, AObject *r) {
  int res = AObject::BinaryIs(l, r);
  return res == -1 ? AObject::MakeAObject(AObject::kTypeBool) : AObject::Convert(res ? Py_True : Py_False);
}

AObject *BinaryContains(AObject *l, AObject *r) {
  int res = AObject::BinaryContains(l, r);
  return res == -1 ? AObject::MakeAObject(AObject::kTypeBool) : AObject::Convert(res ? Py_True : Py_False);
}

using InferBinaryFunc = AObject *(*)(AObject *, AObject *);
using InferBinaryTypeFunc = AObject::Type (*)(AObject::Type, AObject::Type);

template <binaryfunc pyfunc, InferBinaryTypeFunc type_infer>
AObject *InferBinary(AObject *a, AObject *b) {
  PyObject *l = a->GetPyObject().ptr();
  PyObject *r = b->GetPyObject().ptr();
  if (l == nullptr || r == nullptr) {
    return AObject::MakeAObject(type_infer(a->GetType(), b->GetType()));
  }
  if (a->GetType() == AObject::kTypeAnyValue || b->GetType() == AObject::kTypeAnyValue) {
    return AObject::MakeAObject(AObject::kTypeAnyValue);
  }
  PyObject *o = pyfunc(l, r);
  CHECK_PYTHON_EXCEPTION(o);
  AObject *res = AObject::Convert(o);
  Py_XDECREF(o);
  return res;
}

// the inplace binary operations of known type don't modify original python object
// list, tuple, dict already override binary
static std::unordered_map<int, InferBinaryFunc> infer_binary_func = {
  {BINARY_MATRIX_MULTIPLY, InferBinary<PyNumber_MatrixMultiply, BinaryInferDefault>},          // '@'
  {INPLACE_MATRIX_MULTIPLY, InferBinary<PyNumber_InPlaceMatrixMultiply, BinaryInferDefault>},  // '@='
  {BINARY_POWER, InferBinary<BinaryPow, NumberArithmetic>},                                    // '**'
  {INPLACE_POWER, InferBinary<InplacePow, NumberArithmetic>},                                  // '**='
  {BINARY_MULTIPLY, InferBinary<PyNumber_Multiply, NumberArithmetic>},                         // '*'
  {INPLACE_MULTIPLY, InferBinary<PyNumber_InPlaceMultiply, NumberArithmetic>},                 // '*='
  {BINARY_MODULO, InferBinary<PyNumber_Remainder, NumberArithmetic>},                          // '%'
  {INPLACE_MODULO, InferBinary<PyNumber_InPlaceRemainder, NumberArithmetic>},                  // '%='
  {BINARY_ADD, InferBinary<PyNumber_Add, BinaryAdd>},
  {INPLACE_ADD, InferBinary<PyNumber_InPlaceAdd, BinaryAdd>},
  {BINARY_SUBTRACT, InferBinary<PyNumber_Subtract, NumberArithmetic>},
  {INPLACE_SUBTRACT, InferBinary<PyNumber_InPlaceSubtract, NumberArithmetic>},
  {BINARY_FLOOR_DIVIDE, InferBinary<PyNumber_FloorDivide, NumberArithmetic>},          // '//'
  {INPLACE_FLOOR_DIVIDE, InferBinary<PyNumber_InPlaceFloorDivide, NumberArithmetic>},  // '//='
  {BINARY_TRUE_DIVIDE, InferBinary<PyNumber_TrueDivide, NumberArithmetic>},
  {INPLACE_TRUE_DIVIDE, InferBinary<PyNumber_InPlaceTrueDivide, NumberArithmetic>},
  {BINARY_LSHIFT, InferBinary<PyNumber_Lshift, BinaryIntOp>},
  {INPLACE_LSHIFT, InferBinary<PyNumber_InPlaceLshift, BinaryIntOp>},
  {BINARY_RSHIFT, InferBinary<PyNumber_Rshift, BinaryIntOp>},
  {INPLACE_RSHIFT, InferBinary<PyNumber_InPlaceRshift, BinaryIntOp>},
  {BINARY_AND, InferBinary<PyNumber_And, NumberLogic>},
  {INPLACE_AND, InferBinary<PyNumber_InPlaceAnd, NumberLogic>},
  {BINARY_XOR, InferBinary<PyNumber_Xor, NumberLogic>},
  {INPLACE_XOR, InferBinary<PyNumber_InPlaceXor, NumberLogic>},
  {BINARY_OR, InferBinary<PyNumber_Or, NumberLogic>},
  {INPLACE_OR, InferBinary<PyNumber_InPlaceOr, NumberLogic>},
  {CONTAINS_OP, BinaryContains},
  {IS_OP, BinaryIs}};

AObject *AbstractObject::Binary(AObject *other, int op) {
  if (other == nullptr) {
    return MakeAObject(kTypeAnyValue);
  }
  auto iter = infer_binary_func.find(op);
  return iter == infer_binary_func.end() ? MakeAObject(kTypeAnyValue) : iter->second(this, other);
}

AObject *AbstractType::BuildAbstractInstance(const std::vector<AObject *> &args, int opcode) {
  PyTypeObject *tp = reinterpret_cast<PyTypeObject *>(value_.ptr());
  auto type = kTypeAnyValue;
  switch (type_type_) {
    case kTypeList:
    case kTypeTuple: {
      MS_EXCEPTION_IF_CHECK_FAIL((tp == &PyList_Type || tp == &PyTuple_Type), "Use non-tuple to create tuple.");
      if (args.empty()) {
        return MakeAObject(type_type_, tp, nullptr);
      }
      if (args[0] && (args[0]->GetType() == kTypeTuple || args[0]->GetType() == kTypeList)) {
        return MakeAObject(type_type_, tp, nullptr, static_cast<AbstractSequence *>(args[0])->GetElements());
      }
      return MakeAObject(type_type_, tp, nullptr);
    }
    case kTypeBool: {
      if (args.size() == 0) {
        return Convert(Py_False);
      }
      type = args[0] ? args[0]->GetType() : kTypeAnyValue;
      if (type == kTypeList || type == kTypeTuple) {
        AbstractTuple *tmp = static_cast<AbstractTuple *>(args[0]);
        return Convert(tmp->size() ? Py_True : Py_False);
      }
      if (type == kTypeDict) {
        AbstractDict *tmp = static_cast<AbstractDict *>(args[0]);
        return Convert(tmp->size() ? Py_True : Py_False);
      }
    }
    default:
      break;
  }
  return MakeAObject(type_type_, tp, nullptr);
}

// this function call object without error
py::object AbstractType::BuildInstance(const std::vector<py::object> &args, int opcode) {
  if (value_.ptr() == nullptr) {
    MS_LOG(INFO) << "Create instance failed, unknown class";
    return py::object();
  }
  auto pair = Utils::PackCallStackArgs(args, opcode, true);
  if (pair.first.ptr() == nullptr) {
    MS_LOG(INFO) << "Create instance failed, unknown opcode or arguments";
    return py::object();
  }
  PyObject *const *vector_args = &PyTuple_GET_ITEM(pair.first.ptr(), 0);
  Py_ssize_t kw_cnt = pair.second.ptr() == nullptr ? 0 : PyTuple_GET_SIZE(pair.second.ptr());
  Py_ssize_t nargs = PyTuple_GET_SIZE(pair.first.ptr());
  PyObject *inst = PyObject_Vectorcall(value_.ptr(), vector_args, nargs - kw_cnt, pair.second.ptr());
  CHECK_PYTHON_EXCEPTION(inst);
  return py::reinterpret_steal<py::object>(inst);
}

AbstractSequence::AbstractSequence(Type type, const std::vector<AObject *> &elements)
    : AbstractObject(type, py::object()), elements_(elements) {
  std::for_each(elements_.begin(), elements_.end(), [this](auto element) {
    if (element_type_ == kTypeUnknown) {
      element_type_ = element->GetType();
    } else {
      if (element_type_ != kTypeAnyValue) {
        if (element->GetType() == kTypeAnyValue) {
          element_type_ = kTypeAnyValue;
        } else {
          if (element_type_ != element->GetType()) {
            element_type_ = kTypeMultiType;
          }
        }
      }
    }
    element->AddUser(this);
  });
  py::list res;
  bool is_valid = true;
  for (const auto &element : elements_) {
    auto obj = element->GetPyObject();
    if (obj.ptr() != nullptr) {
      res.append(obj);
    } else {
      is_valid = false;
      break;
    }
  }
  if (is_valid) {
    if (type == kTypeList) {
      value_ = res;
    } else {
      value_ = py::tuple(res);
    }
  }
}

AObject *AbstractSequence::GetItem(AObject *k) {
  MS_EXCEPTION_IF_NULL(k);
  auto subscript = Utils::FormatSubscript(k->GetPyObject(), size());
  // invalid subscript object
  if (subscript.empty()) {
    return AObject::MakeAObject(kTypeAnyValue);
  }
  // valid subscript object slice, but no element
  constexpr int len_index = 2;
  if (subscript[len_index] == 0) {
    auto res = AObject::MakeAObject(type_, type_object_, nullptr, {});
    res->AddUser(this);
    return res;
  }
  constexpr int start_index = 0;
  constexpr int step_index = 1;
  InitElementsListIfNeed();
  std::vector<AObject *> elements;
  for (Py_ssize_t index = 0; index < subscript[len_index]; index++) {
    elements.push_back(elements_[subscript[start_index] + index * subscript[step_index]]);
  }
  if (subscript.back() == 0) {
    return elements[0];
  }
  auto res = AObject::MakeAObject(type_, type_object_, nullptr, elements);
  res->AddUser(this);
  return res;
}

bool AbstractSequence::SetItem(AObject *k, AObject *v) {
  MS_EXCEPTION_IF_NULL(k);
  auto subscript = Utils::FormatSubscript(k->GetPyObject(), size());
  // invalid subscript object
  if (subscript.empty()) {
    return false;
  }
  InitElementsListIfNeed();
  std::vector<AObject *> elements(elements_);
  constexpr int start_index = 0;
  if (subscript.back() == 0) {
    elements[subscript[start_index]] = v;
  } else {
    constexpr int step_index = 1;
    constexpr int len_index = 2;
    for (Py_ssize_t index = 0; index < subscript[len_index]; index++) {
      elements[subscript[start_index] + index * subscript[step_index]] = v->GetItem(Convert(py::int_(index)));
    }
  }
  auto seq = static_cast<AbstractSequence *>(MakeAObject(type_, type_object_, nullptr, elements));
  seq->element_type_ =
    v->GetType() == element_type_ ? element_type_ : v->GetType() == kTypeAnyValue ? kTypeAnyValue : kTypeMultiType;
  SetNextVersion(seq);
  return true;
}

void AbstractSequence::CreateVersionWithNewValue() {
  if (!IsLatestVersion()) {
    return;
  }
  if (std::all_of(elements_.begin(), elements_.end(), [](const auto &element) { return element->IsLatestVersion(); })) {
    return;
  }
  std::vector<AObject *> elements;
  std::transform(elements_.begin(), elements_.end(), std::back_inserter(elements),
                 [](const auto &element) { return element->GetLatestVersion(); });
  auto seq = static_cast<AbstractSequence *>(AObject::MakeAObject(type_, type_object_, nullptr, elements));
  SetNextVersion(seq);
  for (const auto &user : users_) {
    user->CreateVersionWithNewValue();
  }
}

void AbstractSequence::InitElementsListIfNeed() {
  if (!IsBaseVersion() || !elements_.empty()) {
    return;
  }
  auto is_seq = py::isinstance<py::list>(value_) || py::isinstance<py::tuple>(value_) ||
                IsCellListType<false>(Py_TYPE(value_.ptr()));
  MS_EXCEPTION_IF_CHECK_FAIL(is_seq, "Invalid value_ for abstract sequence.");
  std::transform(value_.begin(), value_.end(), std::back_inserter(elements_), [this](const auto &element) {
    auto vobj = Convert(element.ptr());
    vobj->AddUser(this);
    return vobj;
  });
}

AObject *AbstractSequence::Binary(AObject *o, int op) {
  // generic binary
  PyObject *r_obj = o ? o->GetPyObject().ptr() : nullptr;
  if (op == IS_OP) {
    bool cnst = const_object_type_map.find(r_obj) != const_object_type_map.end();
    return cnst ? Convert(Py_False) : MakeAObject(kTypeBool);
  }
  if (op == CONTAINS_OP) {
    return infer_binary_func[CONTAINS_OP](this, o);
  }
  // tuple binary
  if (o == nullptr || this->GetType() != o->GetType()) {
    if (this->GetType() == kTypeList && op == BINARY_MULTIPLY && (o != nullptr) && o->GetType() == kTypeInt) {
      std::vector<AObject *> elements;
      InitElementsListIfNeed();
      int res = PyLong_AsLong(o->GetPyObject().ptr());
      for (int i = 0; i < res; i++) {
        std::copy(elements_.begin(), elements_.end(), std::back_inserter(elements));
      }
      return MakeAObject(type_, type_object_, nullptr, elements);
    }
    return MakeAObject(kTypeAnyValue);
  }
  auto r_list = static_cast<AbstractSequence *>(o);
  if (op == BINARY_ADD || (this->GetType() == kTypeTuple && op == INPLACE_ADD)) {
    std::vector<AObject *> elements;
    InitElementsListIfNeed();
    std::copy(elements_.begin(), elements_.end(), std::back_inserter(elements));
    std::copy(r_list->elements_.begin(), r_list->elements_.end(), std::back_inserter(elements));
    return MakeAObject(type_, type_object_, nullptr, elements);
  }
  if (op == INPLACE_ADD) {
    auto seq = static_cast<AbstractSequence *>(MakeAObject(type_, type_object_, nullptr, elements_));
    std::copy(r_list->elements_.begin(), r_list->elements_.end(), std::back_inserter(seq->elements_));
    SetNextVersion(seq);
    return seq;
  }
  // binary mul, inplace mul
  return MakeAObject(kTypeAnyValue);
}

AObject *AbstractSequence::Unary(int op) const {
  if (op != UNARY_NOT) {
    return MakeAObject(kTypeAnyValue);
  }
  return Convert(this->size() > 0 ? Py_True : Py_False);
}

AObject *AbstractSequence::GetAttr(const std::string &name) {
  py::object list = (type_ == kTypeList) ? (py::object)py::list() : py::tuple();
  PyObject *attr = PyObject_GetAttrString(list.ptr(), name.c_str());
  CHECK_PYTHON_EXCEPTION(attr);
  if (attr == nullptr) {
    FIND_MAP_CACHE(attrs_, name);
  }
  AObject *res = Convert(attr);
  Py_XDECREF(attr);
  return res;
}

bool AbstractSequence::IsMindSporeSupportedType() {
  ReprRecursionScope scope(GetPyObject().ptr());
  if (scope.ReEnterOrError()) {
    return true;
  }
  InitElementsListIfNeed();
  return std::all_of(elements_.begin(), elements_.end(),
                     [](AObject *element) { return element->IsMindSporeSupportedType(); });
}

std::string AbstractSequence::ToString() const {
  std::stringstream s;
  s << (type_ == kTypeTuple ? "Tuple<" : "List<");
  s << GetTypeDesc(element_type_) << " * " << size() << "> ";
  s << this << "{";
  auto v = this->GetBaseVersion();
  while (v != nullptr) {
    s << v << ", ";
    v = v->GetNextVersion();
  }
  s << "}";
  s << " user={";
  for (auto user : users_) {
    s << user << ", ";
  }
  s << "}";
  s << " value=" << (type_ == kTypeTuple ? "(" : "[");
  if (value_.ptr() != nullptr) {
    s << AObject::ToString(value_.ptr(), false, kValueToStringLimit);
  } else {
    for (const auto &element : elements_) {
      s << element->ToString() << ", ";
    }
  }
  s << (type_ == kTypeTuple ? ")" : "]");
  return s.str();
}

AbstractNamedTuple::AbstractNamedTuple(const py::object &o, PyTypeObject *tp)
    : AbstractObject(kTypeNamedTuple, o), type_name_(tp->tp_name), keys_() {
  py::object fields = py::getattr(reinterpret_cast<PyObject *>(tp), "_fields", nullptr);
  if (fields.ptr() == nullptr || !PyTuple_Check(fields.ptr())) {
    MS_LOG(INFO) << type_name_ << "._fields is not a tuple";
    return;
  }
  for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(fields.ptr()); ++i) {
    const auto &name = py::cast<std::string>(PyTuple_GET_ITEM(fields.ptr(), i));
    keys_.push_back(name);
  }
}

bool AbstractNamedTuple::IsNamedTuple(PyTypeObject *tp) {
  // Currently, a subclass that extends namedtuple is not supported, so we add the restrict:
  // PyTuple_GET_SIZE(tp->tp_bases) == 1
  if (PyType_IsSubtype(tp, &PyTuple_Type) && PyTuple_GET_SIZE(tp->tp_bases) == 1) {
    auto *obj = reinterpret_cast<PyObject *>(tp);
    return py::hasattr(obj, "_fields") && py::hasattr(obj, "_make");
  }
  return false;
}

int AbstractNamedTuple::GetIndexOfKey(const std::string &name) const {
  for (size_t i = 0; i < keys_.size(); ++i) {
    if (keys_[i] == name) {
      return SizeToInt(i);
    }
  }
  return -1;
}

AbstractList *AbstractList::ListAppend(AObject *item) {
  InitElementsListIfNeed();
  std::vector<AObject *> elements;
  std::for_each(elements_.begin(), elements_.end(), [&elements](auto element) { elements.push_back(element); });
  elements.push_back(item);
  auto list = static_cast<AbstractList *>(MakeAObject(type_, &PyList_Type, nullptr, elements));
  auto type = item->GetType();
  list->element_type_ = (elements_.empty() || (element_type_ == type)) ? type : kTypeAnyValue;
  SetNextVersion(list);
  // Notify this's user to update version
  for (const auto &user : users_) {
    user->CreateVersionWithNewValue();
  }
  return list;
}

AbstractList *AbstractList::ListExtend(AObject *l) {
  auto seq = dynamic_cast<AbstractSequence *>(l);
  MS_EXCEPTION_IF_NULL(seq);
  InitElementsListIfNeed();
  std::vector<AObject *> elements;
  std::for_each(elements_.begin(), elements_.end(), [&elements](auto element) { elements.push_back(element); });
  std::for_each(seq->GetElements().begin(), seq->GetElements().end(),
                [&elements](auto element) { elements.push_back(element); });
  auto list = static_cast<AbstractList *>(MakeAObject(type_, &PyList_Type, nullptr, elements));
  list->element_type_ =
    (elements_.empty() || (element_type_ == seq->GetElementType())) ? seq->GetElementType() : kTypeAnyValue;
  SetNextVersion(list);
  // Notify this's user to update version
  for (const auto &user : users_) {
    user->CreateVersionWithNewValue();
  }
  return list;
}

AbstractTuple *AbstractList::ListToTuple() {
  InitElementsListIfNeed();
  std::vector<AObject *> elements;
  std::for_each(elements_.begin(), elements_.end(), [&elements](auto element) { elements.push_back(element); });
  auto tuple = static_cast<AbstractTuple *>(MakeAObject(kTypeTuple, &PyTuple_Type, nullptr, elements));
  tuple->SetElementType(element_type_);
  return tuple;
}

AObjectPairList CreateAbstractPairList(const std::vector<AObject *> &elements) {
  std::map<AObject *, int> keys_2_index;
  std::vector<AObjectPair> key_values;
  for (size_t index = 0; index < elements.size(); index += 2) {
    if (keys_2_index.find(elements[index]) != keys_2_index.end()) {
      key_values[keys_2_index.at(elements[index])].second = elements[index + 1];
    } else {
      keys_2_index[elements[index]] = key_values.size();
      key_values.push_back(std::make_pair(elements[index], elements[index + 1]));
    }
  }
  return key_values;
}

AbstractDict::AbstractDict(const std::vector<AObject *> &key_values)
    : AbstractObject(kTypeDict, py::object()),
      k_type_(kTypeUnknown),
      v_type_(kTypeUnknown),
      key_values_(CreateAbstractPairList(key_values)) {
  std::for_each(key_values_.begin(), key_values_.end(), [this](auto element) {
    if (k_type_ == kTypeUnknown) {
      k_type_ = element.first->GetType();
    } else {
      if (k_type_ != kTypeAnyValue) {
        if (element.first->GetType() == kTypeAnyValue) {
          k_type_ = kTypeAnyValue;
        } else {
          if (k_type_ != element.first->GetType()) {
            k_type_ = kTypeMultiType;
          }
        }
      }
    }
    element.first->AddUser(this);
    if (v_type_ == kTypeUnknown) {
      v_type_ = element.second->GetType();
    } else {
      if (v_type_ != kTypeAnyValue) {
        if (element.second->GetType() == kTypeAnyValue) {
          v_type_ = kTypeAnyValue;
        } else {
          if (v_type_ != element.second->GetType()) {
            v_type_ = kTypeMultiType;
          }
        }
      }
    }
    element.second->AddUser(this);
  });
  auto res = py::dict();
  bool is_valid = true;
  for (const auto &[key, value] : key_values_) {
    auto k = key->GetPyObject();
    auto v = value->GetPyObject();
    if (k.ptr() != nullptr && v.ptr() != nullptr) {
      res[k] = v;
    } else {
      is_valid = false;
      break;
    }
  }
  if (is_valid) {
    value_ = res;
  }
}

AObject *AbstractDict::Unary(int op) const {
  if (op != UNARY_NOT) {
    return MakeAObject(kTypeAnyValue);
  }
  return Convert(this->size() ? Py_True : Py_False);
}

bool AbstractDict::IsMindSporeSupportedType() { return false; }

std::string AbstractDict::ToString() const {
  std::stringstream s;
  s << "Dict{<<" << GetTypeDesc(k_type_) << ", " << GetTypeDesc(v_type_) << "> * " << size() << "> ";
  s << this << "{";
  auto v = this->GetBaseVersion();
  while (v != nullptr) {
    s << v << ", ";
    v = v->GetNextVersion();
  }
  s << "}";
  s << " user={";
  for (auto user : users_) {
    s << user << ", ";
  }
  s << "}";
  if (value_.ptr() != nullptr) {
    s << "{value=" << AObject::ToString(value_.ptr(), false, kValueToStringLimit) << "}";
  } else {
    for (const auto &[key, value] : key_values_) {
      s << "{" << key->ToString() << ", " << value->ToString() << "}, ";
    }
  }
  return s.str();
}

AObject *AbstractDict::Binary(AObject *other, int op) {
  if (op == IS_OP) {
    PyObject *b = other ? other->GetPyObject().ptr() : nullptr;
    bool cnst = const_object_type_map.find(b) != const_object_type_map.end();
    return cnst ? Convert(Py_False) : MakeAObject(kTypeBool);
  }
  if (op == CONTAINS_OP && other != nullptr) {
    return infer_binary_func[CONTAINS_OP](this, other);
  }
  return MakeAObject(kTypeAnyValue);
}

AObject *AbstractDict::GetAttr(const std::string &name) {
  if (value_.ptr() == nullptr) {
    return AObject::MakeAObject(kTypeAnyValue);
  }
  PyObject *attr = PyObject_GetAttrString(value_.ptr(), name.c_str());
  CHECK_PYTHON_EXCEPTION(attr);
  AObject *res = Convert(attr);
  Py_XDECREF(attr);
  return res;
}

void AbstractDict::InitKeyValuesListIfNeed() {
  if (!IsBaseVersion() || !key_values_.empty()) {
    return;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(py::isinstance<py::dict>(value_), "Invalid value_ for abstract dict.");
  auto dict = py::cast<py::dict>(value_);
  std::transform(dict.begin(), dict.end(), std::back_inserter(key_values_), [](const auto &item) {
    return std::make_pair(Convert(item.first.ptr()), Convert(item.second.ptr()));
  });
}

AObject *AbstractDict::GetItem(AObject *k) {
  MS_EXCEPTION_IF_NULL(k);
  if (k->GetType() == kTypeAnyValue) {
    return MakeAObject(kTypeAnyValue);
  }
  if (key_values_.empty()) {
    auto key = k->GetPyObject();
    if (key.ptr() == nullptr) {
      return MakeAObject(kTypeAnyValue);
    }
    auto res = Convert(PyDict_GetItem(value_.ptr(), key.ptr()));
    res->AddUser(this);
    return res;
  }
  for (const auto &[key, value] : key_values_) {
    if (key == k) {
      return value;
    }
  }
  return MakeAObject(kTypeAnyValue);
}

bool AbstractDict::DictMerge(const AObject *dict) {
  MS_EXCEPTION_IF_NULL(dict);
  MS_EXCEPTION_IF_CHECK_FAIL(dict->GetType() == kTypeDict, "Only dict can call DictMerge.");
  std::vector<AObject *> key_values;
  InitKeyValuesListIfNeed();
  auto d = static_cast<const AbstractDict *>(dict);
  std::for_each(d->GetElements().begin(), d->GetElements().end(), [&key_values](const auto &item) {
    key_values.push_back(item.first);
    key_values.push_back(item.second);
  });
  auto new_dict = static_cast<AbstractDict *>(MakeAObject(type_, &PyDict_Type, nullptr, key_values));
  SetNextVersion(new_dict);
  for (const auto &user : users_) {
    user->CreateVersionWithNewValue();
  }
  return true;
}

bool AbstractDict::DictUpdate(const AObject *dict) { return DictMerge(dict); }

bool AbstractDict::MapAdd(AObject *k, AObject *v) {
  MS_EXCEPTION_IF_NULL(k);
  MS_EXCEPTION_IF_NULL(v);
  if (k->GetType() == kTypeAnyValue || v->GetType() == kTypeAnyValue) {
    return false;
  }
  std::vector<AObject *> key_values;
  InitKeyValuesListIfNeed();
  std::for_each(key_values_.begin(), key_values_.end(), [&key_values](const auto &item) {
    key_values.push_back(item.first);
    key_values.push_back(item.second);
  });
  auto new_dict = static_cast<AbstractDict *>(MakeAObject(type_, &PyDict_Type, nullptr, key_values));
  SetNextVersion(new_dict);
  for (const auto &user : users_) {
    user->CreateVersionWithNewValue();
  }
  return true;
}

void AbstractDict::CreateVersionWithNewValue() {
  if (!IsLatestVersion()) {
    return;
  }
  if (std::all_of(key_values_.begin(), key_values_.end(),
                  [](const auto &item) { return item.first->IsLatestVersion() && item.second->IsLatestVersion(); })) {
    return;
  }
  std::vector<AObject *> key_values;
  std::for_each(key_values_.begin(), key_values_.end(), [&key_values](const auto &item) {
    key_values.push_back(item.first);
    key_values.push_back(item.second);
  });
  auto dict = static_cast<AbstractDict *>(MakeAObject(type_, &PyDict_Type, nullptr, key_values));
  SetNextVersion(dict);
  for (const auto &user : users_) {
    user->CreateVersionWithNewValue();
  }
}

py::object AbstractTensor::GetTensor(bool sync) {
  if (!is_stub_ || !sync) {
    return value_;
  }
  std::string attr_key = "tensor";
  auto iter = attrs_.find(attr_key);
  if (iter != attrs_.end()) {
    return iter->second->GetPyObject();
  }
  PyObject *res = PyObject_GetAttrString(value_.ptr(), attr_key.c_str());
  if (res != nullptr && res != Py_None) {
    attrs_[attr_key] = AObject::Convert(res);
    return py::reinterpret_steal<py::object>(res);
  }
  if (res == nullptr) {
    PyErr_Clear();
  } else {
    Py_DECREF(res);
  }
  PyObject *meth = PyObject_GetAttrString(value_.ptr(), "stub_sync");
  MS_EXCEPTION_IF_CHECK_FAIL(meth && PyMethod_Check(meth), "check value");
  res = PyObject_Call(meth, py::tuple().ptr(), nullptr);
  Py_DECREF(meth);
  CHECK_PYTHON_EXCEPTION(res);
  attrs_[attr_key] = AObject::Convert(res);
  return py::reinterpret_steal<py::object>(res);
}

AbstractBasePtr PyObjectToAbstract(const py::object &arg) {
  ValuePtr converted = nullptr;
  bool success;
  if (IsStubTensor(arg)) {
    success = mindspore::parse::ConvertStubData(arg, &converted);
  } else {
    success = mindspore::parse::ConvertData(arg, &converted);
  }
  if (!success) {
    MS_LOG(EXCEPTION) << "Fail to convert the object: " << py::str(arg);
  }
  auto res = GraphUtils::ArgsToAbstract(arg, converted, false);
  if (res->isa<mindspore::abstract::AbstractTensor>()) {
    bool check = CheckAdapterTensor(arg);
    dyn_cast_ptr<mindspore::abstract::AbstractTensor>(res)->set_is_adapter(check);
  }
  return res;
}

bool TensorInferBinarySupport(int opcode) {
  static const std::set<int> support_op = {
    BINARY_POWER,         BINARY_MULTIPLY,     BINARY_MODULO,       BINARY_ADD,
    BINARY_SUBTRACT,      BINARY_SUBSCR,       BINARY_FLOOR_DIVIDE, BINARY_TRUE_DIVIDE,
    INPLACE_FLOOR_DIVIDE, INPLACE_TRUE_DIVIDE, INPLACE_ADD,         INPLACE_SUBTRACT,
    INPLACE_MULTIPLY,     INPLACE_MODULO,      BINARY_LSHIFT,       BINARY_RSHIFT,
    BINARY_AND,           BINARY_XOR,          BINARY_OR,           INPLACE_POWER,
    INPLACE_LSHIFT,       INPLACE_RSHIFT,      INPLACE_AND,         INPLACE_XOR,
    INPLACE_OR,
  };

  return support_op.find(opcode) != support_op.end();
}

mindspore::abstract::AbstractTensorPtr InferWithMetaFunc(const AbstractBasePtr &left, const AbstractBasePtr &right,
                                                         int opcode) {
  auto func = GraphUtils::GetPrimOrMetaFuncGraph(opcode);
  auto res = mindspore::pipeline::AbstractAnalyze(GetValueNode(func), {left, right});
  return dyn_cast<mindspore::abstract::AbstractTensor>(res.eval_result->abstract());
}

mindspore::abstract::AbstractTensorPtr InferWithPrim(const AbstractBasePtr &left, const AbstractBasePtr &right,
                                                     int opcode) {
  static std::unordered_map<int, PrimitivePtr> prim_func = {{BINARY_ADD, prim::kPrimAdd},
                                                            {BINARY_SUBTRACT, prim::kPrimSub},
                                                            {BINARY_MULTIPLY, prim::kPrimMul},
                                                            {BINARY_TRUE_DIVIDE, prim::kPrimDiv},
                                                            {BINARY_FLOOR_DIVIDE, prim::kPrimFloorDiv}};

  auto left_dtype_ptr = dyn_cast_ptr<mindspore::abstract::AbstractTensor>(left)->element()->BuildType();
  MS_EXCEPTION_IF_NULL(left_dtype_ptr);
  auto right_dtype_ptr = dyn_cast_ptr<mindspore::abstract::AbstractTensor>(right)->element()->BuildType();
  MS_EXCEPTION_IF_NULL(right_dtype_ptr);
  if (left_dtype_ptr->type_id() != right_dtype_ptr->type_id() || prim_func.find(opcode) == prim_func.end()) {
    return InferWithMetaFunc(left, right, opcode);
  }

  auto func = prim_func.find(opcode)->second;
  auto infer_res = mindspore::abstract::TryInferAbstract(func, {left, right});
  if (infer_res.has_value()) {
    MS_EXCEPTION_IF_NULL(infer_res.value());
    return dyn_cast<mindspore::abstract::AbstractTensor>(infer_res.value());
  } else {
    return nullptr;
  }
}

py::object TensorInferBinary(const AbstractBasePtr &left, const AbstractBasePtr &right, int opcode) {
  mindspore::abstract::AbstractTensorPtr abs;
  if (right->isa<mindspore::abstract::AbstractTensor>()) {
    abs = InferWithPrim(left, right, opcode);
  } else if (right->isa<mindspore::abstract::AbstractScalar>()) {
    auto new_right = std::make_shared<mindspore::abstract::AbstractTensor>(right);
    abs = InferWithPrim(left, new_right, opcode);
  } else {
    abs = InferWithMetaFunc(left, right, opcode);
  }
  MS_EXCEPTION_IF_NULL(abs);
  auto dtype_ptr = abs->element()->BuildType();
  MS_EXCEPTION_IF_NULL(dtype_ptr);
  auto shape_ptr = abs->BuildShape();
  MS_EXCEPTION_IF_NULL(shape_ptr);
  auto shape = shape_ptr->cast<mindspore::abstract::ShapePtr>()->shape();
  auto dtype = dtype_ptr->type_id();
  py::object tensorpyObject = PackTensorToPyObject(std::make_shared<mindspore::tensor::Tensor>(dtype, shape));
  return tensorpyObject;
}

py::object AbstractTensor::Binary(int op, const py::object &l_tensor, const py::object &r_tensor) {
  auto left = PyObjectToAbstract(l_tensor);
  auto right = PyObjectToAbstract(r_tensor);
  auto res = TensorInferBinary(left, right, op);
  if (CheckAdapterTensor(l_tensor)) {
    res = ConvertToAdapterTensor(res);
  } else {
    res = ConvertToMsTensor(res);
  }
  return res;
}

AObject *AbstractTensor::Binary(AObject *other, int op) {
  if (op == IS_OP) {
    PyTypeObject *b = other ? other->GetTypeObject() : nullptr;
    PyTypeObject *a = GetTypeObject();
    return a != b && b != nullptr ? Convert(Py_False) : MakeAObject(kTypeBool);
  }

  if (other == nullptr || GetPyObject().ptr() == nullptr || !TensorInferBinarySupport(op)) {
    return MakeAObject(kTypeTensor);
  }

  AbstractBasePtr left = PyObjectToAbstract(this->GetPyObject());
  AbstractBasePtr right;
  if (other->GetPyObject().ptr() == nullptr) {
    // if other is scalar with empty value, then transfer to AbstractScalar
    // else return any value
    switch (other->GetType()) {
      case kTypeBool:
        right = std::make_shared<mindspore::abstract::AbstractScalar>(kValueAny, kBool);
        break;
      case kTypeInt:
        right = std::make_shared<mindspore::abstract::AbstractScalar>(kValueAny, kInt32);
        break;
      case kTypeFloat:
        right = std::make_shared<mindspore::abstract::AbstractScalar>(kValueAny, kFloat32);
        break;
      default:
        return MakeAObject(kTypeAnyValue);
    }
  } else {
    right = PyObjectToAbstract(other->GetPyObject());
  }
  auto res = TensorInferBinary(left, right, op);
  if (CheckAdapterTensor(value_)) {
    res = ConvertToAdapterTensor(res);
  } else {
    res = ConvertToMsTensor(res);
  }
  return Convert(res);
}

AObject *AbstractTensor::GetItem(AObject *key) {
  PyObject *s = value_.ptr();
  PyObject *i = key ? key->GetPyObject().ptr() : nullptr;
  PyObject *t = nullptr;
  if (s != nullptr && i != nullptr) {
    // avoid Tensor as index and Tensor data sync
    t = PyObject_GetItem(s, i);
    CHECK_PYTHON_EXCEPTION(t);
  } else {
    return MakeAObject(kTypeAnyValue);
  }
  py::object res = py::reinterpret_steal<py::object>(t);
  if (CheckAdapterTensor(value_)) {
    res = ConvertToAdapterTensor(res);
  } else {
    res = ConvertToMsTensor(res);
  }
  auto vobj = Convert(res);
  vobj->AddUser(this);
  return vobj;
}

AObject *AbstractTensor::Unary(int op) const {
  if (this->value_.ptr() != nullptr) {
    return this->AbstractObject::UnaryValue(op);
  }
  if (op == UNARY_POSITIVE) {
    return const_cast<AbstractTensor *>(this);
  } else if (op == UNARY_NEGATIVE || op == UNARY_INVERT) {
    AbstractTensor *res = static_cast<AbstractTensor *>(MakeAObject(kTypeTensor));
    auto it = attrs_.find("shape");
    if (it != attrs_.end()) {
      res->attrs_["shape"] = it->second;
    }
    it = attrs_.find("dtype");
    if (it != attrs_.end()) {
      res->attrs_["dtype"] = it->second;
    }
    return res;
  } else if (op == UNARY_NOT) {
    auto it = attrs_.find("shape");
    if (it == attrs_.end() || it->second == nullptr) {
      return MakeAObject(kTypeTensor);
    }
    AObject *shape_info = it->second;
    PyObject *shape = shape_info->GetPyObject().ptr();
    Py_ssize_t ndim = PyTuple_GET_SIZE(shape);
    if (ndim == 0 || (ndim == 1 && PyLong_AS_LONG(PyTuple_GET_ITEM(shape, 0))) == 1) {
      return MakeAObject(kTypeBool);
    }
    return MakeAObject(kTypeAnyValue);
  }
  return MakeAObject(kTypeAnyValue);
}

static const std::unordered_map<std::string, AObject::Type> tensor_attr_type = {
  // py Tensor property
  {"shape", AObject::kTypeTuple},
  {"dtype", AObject::kTypeMSDType},
  {"size", AObject::kTypeInt},
  {"itemsize", AObject::kTypeInt},
  {"nbytes", AObject::kTypeInt},
  {"strides", AObject::kTypeTuple},
  {"ndim", AObject::kTypeInt},
  {"has_init", AObject::kTypeBool},
  {"H", AObject::kTypeTensor},
  {"mH", AObject::kTypeTensor},
  {"T", AObject::kTypeTensor},
  {"mT", AObject::kTypeTensor},
  // cpp Tensor property
  {"_shape", AObject::kTypeTuple},
  {"_dtype", AObject::kTypeMSDType},
  {"_size", AObject::kTypeInt},
  {"_itemsize", AObject::kTypeInt},
  {"_nbytes", AObject::kTypeInt},
  {"_strides", AObject::kTypeTuple},
  {"init_flag", AObject::kTypeBool},
  {"adapter_flag", AObject::kTypeBool},
  {"param_info", AObject::kTypeAnyValue},
};

// return an uninitialized python tensor
static PyObject *GetUninitializedTensor() {
  static PyObject *tensor = nullptr;
  if (tensor != nullptr) {
    return tensor;
  }
  py::object py_cls = Utils::GetModuleAttr("mindspore", "Tensor", false, true);
  py::object cpp_cls = Utils::GetModuleAttr("mindspore._c_expression", "Tensor", false, true);
  py::object dtype = Utils::GetModuleAttr("mindspore", "int32", false, true);
  py::tuple shape;
  tensor = py_cls(cpp_cls(dtype, shape)).inc_ref().ptr();
  return tensor;
}

AbstractTensor::AbstractTensor(const py::object &o, bool is_stub) : AbstractObject(kTypeTensor, o), is_stub_(is_stub) {}

AObject *AbstractTensor::GetAttr(const std::string &name) {
  if (value_.ptr()) {
    if (is_stub_ && GetTensor(true).ptr()) {
      return attrs_["tensor"]->GetAttr(name);
    }
    return this->AbstractObject::GetAttr(name);
  }

  PyObject *tmp = GetUninitializedTensor();
  if (type_object_ != Py_TYPE(tmp)) {
    // tensor subclass or StubTensor and it's subclass
    // generic attribute
    AObject *attr = this->AbstractObjectBase::GetAttr(name);
    attrs_[name] = attr;
    return attr;
  }
  // get attribute for exact mindspore.Tensor,
  // not MetaTensor, not mindspore._c_expression.Tensor, not StubTensor

  // known @property attribute
  auto iter = tensor_attr_type.find(name);
  if (iter != tensor_attr_type.end()) {
    AObject *attr = MakeAObject(iter->second);
    if (iter->second == kTypeTuple) {
      static_cast<AbstractTuple *>(attr)->SetElementType(kTypeInt);
    }
    attrs_[name] = attr;
    return attr;
  }

  // know function attribute
  PyObject *op = PyObject_GetAttrString(tmp, name.c_str());
  AObject *attr = Convert(op);
  if (op == nullptr) {
    PyErr_Clear();
  } else {
    Py_DECREF(op);
  }
  if (attr->GetType() == kTypeBoundMethod) {
    attr->SetAttr("__self__", this);
    Py_INCREF(Py_None);
    Py_SETREF(PyMethod_GET_SELF(op), Py_None);
  } else {
    // not initialized attribute is not accept
    attr = MakeAObject(kTypeAnyValue);
  }
  attrs_[name] = attr;
  return attr;
}

std::string AbstractTensor::ToString() const {
  std::stringstream s;
  s << this->AbstractObjectBase::ToString();
  s << this << "{";
  auto v = this->GetBaseVersion();
  while (v != nullptr) {
    s << v << ", ";
    v = v->GetNextVersion();
  }
  s << "}";
  s << " user={";
  for (auto user : users_) {
    s << user << ", ";
  }
  s << "}";
  if (value_.ptr()) {
    s << "{" << AObject::ToString(value_.ptr(), false, kValueToStringLimit) << "}";
  } else {
    s << "{NULL,NULL}";
  }
  if (value_.ptr() != nullptr) {
    s << " data_allocated=" << (CheckTensorDataInitialized(value_) ? "True" : "False");
  }
  return s.str();
}

}  // namespace pijit
}  // namespace mindspore
