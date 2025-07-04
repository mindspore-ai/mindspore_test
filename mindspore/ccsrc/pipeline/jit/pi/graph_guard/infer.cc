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
#include "pipeline/jit/pi/graph_guard/infer.h"
#include <map>
#include <string>
#include <functional>
#include <unordered_set>
#include <utility>
#include <algorithm>
#include <set>
#include "base/base.h"
#include "abstract/ops/primitive_infer_map.h"
#include "frontend/ir/primitive_py.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/stub_tensor.h"
#include "ir/anf.h"
#include "utils/flags.h"
#include "pipeline/jit/pi/utils/utils.h"
#include "pipeline/jit/ps/static_analysis/static_analysis.h"
#include "frontend/operator/composite/composite.h"
#include "ir/cell.h"
#include "pipeline/jit/ps/resource.h"
#include "pipeline/jit/pi/python_adapter/pydef.h"
#include "pipeline/jit/pi/graph_guard/guard_utils.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/pi/graph_build/func_graph_builder.h"
#include "include/common/utils/tensor_py.h"
#include "include/common/pynative/common_utils.h"

namespace mindspore {
namespace parse {
extern bool ConvertData(const py::object &obj, mindspore::ValuePtr *data, bool use_signature,
                        const mindspore::TypePtr &dtype, bool forbid_reuse);
extern bool IsParameterObject(const py::object &);
}  // namespace parse

namespace abstract {
extern mindspore::abstract::AbstractBasePtr ToAbstract(const mindspore::ValuePtr &value,
                                                       const mindspore::abstract::AnalysisContextPtr &context,
                                                       const mindspore::abstract::AnfNodeConfigPtr &conf);
extern std::optional<StandardPrimitiveImplReg> GetPrimitiveInferImpl(const PrimitivePtr &primitive);
}  // namespace abstract

namespace pijit {

static InferEnginePtr g_pInferEngine = nullptr;
constexpr const int ArgsSizeTwo = 2;

template <>
bool IsPrimitiveFunctionType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::PrimitiveFunctionAdapter, true>(tp);
}

InferEnginePtr InferEngine::GetInstance() {
  if (g_pInferEngine == nullptr) {
    g_pInferEngine = std::shared_ptr<InferEngine>(new InferEngine());
  }
  if (g_pInferEngine->Init()) {
    return g_pInferEngine;
  } else {
    return nullptr;
  }
}

InferEngine::InferEngine() {}

bool InferEngine::Init() {
  if (!bInit_) {
    bInit_ = GetMsTensorType() != nullptr;
  }
  return bInit_;
}

bool InferEngine::Deinit() {
  if (bInit_) {
    bInit_ = false;
  }
  return bInit_;
}

static std::map<mindspore::TypeId, std::string> g_type2attr = {
  {mindspore::kNumberTypeBool, "bool_"},          {mindspore::kNumberTypeInt, "int_"},
  {mindspore::kNumberTypeInt4, "int_"},           {mindspore::kNumberTypeInt8, "int8"},
  {mindspore::kNumberTypeInt16, "int16"},         {mindspore::kNumberTypeInt32, "int32"},
  {mindspore::kNumberTypeInt64, "int64"},         {mindspore::kNumberTypeUInt, "uint"},
  {mindspore::kNumberTypeUInt8, "uint8"},         {mindspore::kNumberTypeUInt16, "uint16"},
  {mindspore::kNumberTypeUInt32, "uint32"},       {mindspore::kNumberTypeUInt64, "uint64"},
  {mindspore::kNumberTypeFloat, "float_"},        {mindspore::kNumberTypeFloat16, "float16"},
  {mindspore::kNumberTypeFloat32, "float32"},     {mindspore::kNumberTypeFloat64, "float64"},
  {mindspore::kNumberTypeDouble, "float64"},      {mindspore::kNumberTypeComplex, "complex128"},
  {mindspore::kNumberTypeComplex64, "complex64"}, {mindspore::kNumberTypeComplex128, "complex128"},
};

static py::object MakeObjectFromAbstract(const mindspore::abstract::BaseShapePtr &base_shape,
                                         const mindspore::TypePtr &type, bool *is_abstract);

static py::object CreateMetaTensor(const ShapeVector &shape, const mindspore::TypePtr &type) {
  mindspore::TypePtr dtype;
  if (type->isa<mindspore::TensorType>()) {
    dtype = type->cast<mindspore::TensorTypePtr>()->element();
  } else {
    dtype = type;
  }
  /**
   * NOTE: here create a lazy initialized tensor, avoid allocate data
   */
  py::object tensorpyObject =
    PackTensorToPyObject(std::make_shared<mindspore::tensor::Tensor>(dtype->type_id(), shape));
  return tensorpyObject;
}

static py::object CreateMetaTensor(const mindspore::abstract::ShapePtr &shape, const mindspore::TypePtr &type) {
  MS_EXCEPTION_IF_NULL(shape);
  return CreateMetaTensor(shape->shape(), type);
}

static py::object CreateScalar(const mindspore::TypePtr &type) {
  static std::map<mindspore::TypeId, py::object> ms_type2py_type_map = {
    {mindspore::kNumberTypeBool, py::bool_()},
    {mindspore::kNumberTypeInt, py::int_()},
    {mindspore::kNumberTypeInt4, py::int_()},
    {mindspore::kNumberTypeInt8, py::int_()},
    {mindspore::kNumberTypeInt16, py::int_()},
    {mindspore::kNumberTypeInt32, py::int_()},
    {mindspore::kNumberTypeInt64, py::int_()},
    {mindspore::kNumberTypeUInt, py::int_()},
    {mindspore::kNumberTypeUInt8, py::int_()},
    {mindspore::kNumberTypeUInt16, py::int_()},
    {mindspore::kNumberTypeUInt32, py::int_()},
    {mindspore::kNumberTypeUInt64, py::int_()},
    {mindspore::kNumberTypeFloat, py::float_()},
    {mindspore::kNumberTypeFloat16, py::float_()},
    {mindspore::kNumberTypeFloat32, py::float_()},
    {mindspore::kNumberTypeFloat64, py::float_()},
    {mindspore::kNumberTypeDouble, py::float_()},
    {mindspore::kNumberTypeComplex, py::reinterpret_steal<py::object>(PyComplex_FromDoubles(0.0, 0.0))},
    {mindspore::kNumberTypeComplex64, py::reinterpret_steal<py::object>(PyComplex_FromDoubles(0.0, 0.0))},
    {mindspore::kNumberTypeComplex128, py::reinterpret_steal<py::object>(PyComplex_FromDoubles(0.0, 0.0))},
  };
  auto it = ms_type2py_type_map.find(type->type_id());
  if (it != ms_type2py_type_map.cend()) {
    return it->second;
  } else {
    return py::cast<py::object>(nullptr);
  }
}

static py::object CreateTuple(const mindspore::abstract::BaseShapePtr &base_shape, const mindspore::TypePtr &type,
                              bool *is_abstract) {
  bool dynamic;
  mindspore::abstract::SequenceShapePtr shape_tuple;
  size_t elem_count = 0;
  auto type_tuple = type->cast_ptr<mindspore::Tuple>();
  MS_EXCEPTION_IF_NULL(type_tuple);
  if (base_shape->isa<mindspore::abstract::DynamicSequenceShape>()) {
    dynamic = true;
    elem_count = type_tuple->elements().size();
  } else {
    dynamic = false;
    shape_tuple = base_shape->cast<mindspore::abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_tuple);
    elem_count = shape_tuple->size();
  }
  py::tuple tuple = py::tuple(elem_count);
  for (size_t it = 0; it < elem_count; ++it) {
    bool is_abstract_obj = false;
    auto tensor_it =
      MakeObjectFromAbstract(dynamic ? base_shape : (*shape_tuple)[it], type_tuple->elements()[it], &is_abstract_obj);
    Py_INCREF(tensor_it.ptr());
    PyTuple_SetItem(tuple.ptr(), it, tensor_it.ptr());
    *is_abstract |= is_abstract_obj;
  }
  return tuple;
}

static py::object CreateList(const mindspore::abstract::BaseShapePtr &base_shape, const mindspore::TypePtr &type,
                             bool *is_abstract) {
  bool dynamic;
  mindspore::abstract::SequenceShapePtr shape_list;
  size_t elem_count = 0;
  auto type_list = type->cast_ptr<mindspore::List>();
  MS_EXCEPTION_IF_NULL(type_list);
  if (base_shape->isa<mindspore::abstract::DynamicSequenceShape>()) {
    dynamic = true;
    elem_count = type_list->elements().size();
  } else {
    dynamic = false;
    shape_list = base_shape->cast<mindspore::abstract::ListShapePtr>();
    elem_count = shape_list->size();
  }
  py::list list = py::list(elem_count);
  for (size_t it = 0; it < elem_count; ++it) {
    bool is_abstract_obj = false;
    auto tensor_it =
      MakeObjectFromAbstract(dynamic ? base_shape : (*shape_list)[it], type_list->elements()[it], &is_abstract_obj);
    Py_INCREF(tensor_it.ptr());
    PyList_SetItem(list.ptr(), it, tensor_it.ptr());
    *is_abstract |= is_abstract_obj;
  }
  return list;
}

static py::object MakeObjectFromAbstract(const mindspore::abstract::BaseShapePtr &base_shape,
                                         const mindspore::TypePtr &type, bool *is_abstract) {
  *is_abstract = false;
  if (base_shape->isa<mindspore::abstract::Shape>()) {
    return CreateMetaTensor(base_shape->cast<mindspore::abstract::ShapePtr>(), type);
  } else if (base_shape->isa<mindspore::abstract::NoShape>() && type->isa<mindspore::Number>()) {
    *is_abstract = true;
    return CreateScalar(type);
  } else if (base_shape->isa<mindspore::abstract::TupleShape>() && type->isa<mindspore::Tuple>()) {
    return CreateTuple(base_shape, type, is_abstract);
  } else if (base_shape->isa<mindspore::abstract::ListShape>() && type->isa<mindspore::List>()) {
    return CreateList(base_shape, type, is_abstract);
  } else if (base_shape->isa<mindspore::abstract::NoShape>() && type->isa<mindspore::TypeNone>()) {
    // AbstractNone indicates there is no output for this CNode node.
    return py::cast<py::object>(Py_None);
  } else if (type->isa<mindspore::Monad>()) {
    // Return monad abstract if it is monad type.
    return py::cast<py::object>(nullptr);
  } else if (base_shape->isa<mindspore::abstract::DynamicSequenceShape>()) {
    *is_abstract = true;
    if (type->isa<mindspore::Tuple>()) {
      return CreateTuple(base_shape, type, is_abstract);
    } else if (type->isa<mindspore::List>()) {
      return CreateList(base_shape, type, is_abstract);
    } else if (type->isa<mindspore::TensorType>()) {
      return CreateMetaTensor({-2}, type);
    } else if (type->isa<mindspore::Number>()) {
      return CreateScalar(type);
    } else {
      MS_LOG(EXCEPTION) << "Evaluator return invalid shape " << base_shape->ToString() << " or type. "
                        << type->ToString();
      return py::cast<py::object>(nullptr);
    }
  } else {
    MS_LOG(EXCEPTION) << "Evaluator return invalid shape " << base_shape->ToString() << " or type. "
                      << type->ToString();
    return py::cast<py::object>(nullptr);
  }
}

static py::object MakeObjectFromPyObject(const py::object &shape_obj, const py::object &type_obj, bool *is_abstract) {
  *is_abstract = false;
  if ((py::isinstance<py::list>(shape_obj) || py::isinstance<py::tuple>(shape_obj)) &&
      py::isinstance<mindspore::Type>(type_obj)) {
    auto res_vec = shape_obj.cast<ShapeVector>();
    auto res_dtype = type_obj.cast<mindspore::TypePtr>();
    if (res_vec.empty() && (!res_dtype->isa<TensorType>())) {
      *is_abstract = true;
      return CreateScalar(res_dtype);
    }
    return CreateMetaTensor(res_vec, res_dtype);
  } else if (py::isinstance<py::tuple>(shape_obj) && py::isinstance<py::tuple>(type_obj)) {
    auto typeid_tuple = type_obj.cast<py::tuple>();
    py::tuple ptr_list(typeid_tuple.size());
    for (size_t it = 0; !(*is_abstract) && it < typeid_tuple.size(); ++it) {
      py::object tmp =
        MakeObjectFromPyObject(shape_obj.cast<py::tuple>()[it], type_obj.cast<py::tuple>()[it], is_abstract);
      ptr_list[it] = tmp;
    }
    return ptr_list;
  } else if (py::isinstance<py::list>(shape_obj) && py::isinstance<py::list>(type_obj)) {
    auto typeid_list = type_obj.cast<py::list>();
    py::list ptr_list;
    for (size_t it = 0; !(*is_abstract) && it < typeid_list.size(); ++it) {
      py::object tmp =
        MakeObjectFromPyObject(shape_obj.cast<py::list>()[it], type_obj.cast<py::list>()[it], is_abstract);
      ptr_list.append(tmp);
    }
    return ptr_list;
  } else if (shape_obj.is_none() && type_obj.is_none()) {
    return py::cast<py::object>(Py_None);
  } else if (py::isinstance<mindspore::Type>(type_obj) &&
             type_obj.cast<mindspore::Type *>()->isa<mindspore::MonadType>()) {
    return py::cast<py::object>(nullptr);
  } else {
    MS_LOG(EXCEPTION) << "Python evaluator return invalid shape or type. " << py::str(type_obj);
  }
}

static bool HasTensor(py::object obj) {
  if (obj.ptr() == nullptr) {
    return false;
  }

  ReprRecursionScope scope(obj.ptr());
  if (scope.ReEnterOrError()) {
    return false;
  }
  if (tensor::IsTensorPy(obj)) {
    return true;
  } else if (py::isinstance<py::list>(obj)) {
    auto list_obj = py::cast<py::list>(obj);
    if (std::any_of(list_obj.begin(), list_obj.end(),
                    [](const auto &e) { return HasTensor(py::cast<py::object>(e)); })) {
      return true;
    }
  } else if (py::isinstance<py::tuple>(obj)) {
    auto tuple_obj = py::cast<py::tuple>(obj);
    if (std::any_of(tuple_obj.begin(), tuple_obj.end(),
                    [](const auto &e) { return HasTensor(py::cast<py::object>(e)); })) {
      return true;
    }
  } else if (py::isinstance<py::dict>(obj)) {
    auto dict_obj = py::cast<py::dict>(obj);
    if (std::any_of(dict_obj.begin(), dict_obj.end(), [](const auto &e) {
          return HasTensor(py::cast<py::object>(e.first)) || HasTensor(py::cast<py::object>(e.second));
        })) {
      return true;
    }
  }
  return false;
}

ValuePtr DtypeToEnum(const ValuePtr &value) {
  if (!value->isa<mindspore::Type>()) {
    return value;
  }
  auto type_id = value->cast<TypePtr>()->type_id();
  return MakeValue<int64_t>(type_id);
}

using ArgHandlerFunc = std::function<ValuePtr(const ValuePtr &)>;

ArgHandlerFunc GetOppArgHandlerFunc(const std::string &arg_handler) {
  static const std::unordered_map<std::string, ArgHandlerFunc> opp_arg_handler_funcs = {
    {"dtype_to_type_id", DtypeToEnum},
  };
  if (opp_arg_handler_funcs.find(arg_handler) != opp_arg_handler_funcs.end()) {
    return opp_arg_handler_funcs.at(arg_handler);
  } else {
    return nullptr;
  }
}

mindspore::ValuePtr ConvertArgByArgHandler(mindspore::ValuePtr value, ops::OpDef *op_def, size_t i) {
  if (op_def != nullptr && value != nullptr) {
    auto opp_arg_handler_func = GetOppArgHandlerFunc(op_def->args_[i].arg_handler_);
    if (opp_arg_handler_func != nullptr) {
      return opp_arg_handler_func(value);
    }
  }
  return value;
}

mindspore::ValuePtr ConvertArgByCastDtype(py::object arg, ops::OpInputArg op_arg) {
  mindspore::ValuePtr value = nullptr;
  parse::OpDefConvertFunc convert_func = parse::GetConverterByType(static_cast<int32_t>(op_arg.arg_dtype_));
  MS_EXCEPTION_IF_NULL(convert_func);
  value = convert_func(arg);
  if (value != nullptr) {
    return value;
  }
  if (!op_arg.cast_dtype_.empty()) {
    for (auto cast_dtype : op_arg.cast_dtype_) {
      convert_func = parse::GetConverterByType(parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
      MS_EXCEPTION_IF_NULL(convert_func);
      auto val = convert_func(arg);
      if (val != nullptr) {
        return val;
      }
    }
  }
  return value;
}

mindspore::ValuePtr convertData(py::object param_obj, bool is_stub, ops::OpDef *op_def, size_t i) {
  mindspore::ValuePtr converted = nullptr;
  if (op_def != nullptr) {
    if (op_def->args_.size() <= i) {
      MS_LOG(EXCEPTION) << "Fail to convert the " << i << "th argument by dtype, args[" << i
                        << "]: " << py::str(param_obj);
      return nullptr;
    }
    converted = ConvertArgByCastDtype(param_obj, op_def->args_[i]);
  }
  if (converted) {
    return converted;
  }
  if (is_stub) {
    if (!mindspore::parse::ConvertStubData(param_obj, &converted, false, nullptr, false)) {
      MS_LOG(EXCEPTION) << "Fail to convert the " << i << "th argument, args[" << i << "]: " << py::str(param_obj);
      return nullptr;
    }
  } else {
    if (!mindspore::parse::ConvertData(param_obj, &converted, false, nullptr, false)) {
      MS_LOG(EXCEPTION) << "Fail to convert the " << i << "th argument, args[" << i << "]: " << py::str(param_obj);
      return nullptr;
    }
  }
  return converted;
}

static AbstractBasePtrList ChangeAbstractArgList(PrimitivePtr prim, std::vector<PyObject *> args, bool *has_tensor,
                                                 int *monad_count) {
  std::vector<std::string> prim_cast_ops = {"Div"};
  py::object handle;
  if (std::find(prim_cast_ops.begin(), prim_cast_ops.end(), prim->name()) != prim_cast_ops.end() &&
      args.size() == ArgsSizeTwo) {
    auto tensor_type = py::reinterpret_borrow<py::object>(GetMsTensorType());
    if (tensor::IsTensorPy(args[0]) && CheckScalar(args[1])) {
      py::object dtype = py::reinterpret_borrow<py::object>(args[0]).attr("dtype");
      py::object arg1 = py::reinterpret_borrow<py::object>(args[1]);
      handle = tensor_type(arg1, dtype);
      args[1] = handle.ptr();
    } else if (CheckScalar(args[0]) && tensor::IsTensorPy(args[1])) {
      py::object dtype = py::reinterpret_borrow<py::object>(args[1]).attr("dtype");
      py::object arg0 = py::reinterpret_borrow<py::object>(args[0]);
      handle = tensor_type(arg0, dtype);
      args[0] = handle.ptr();
    }
  }
  auto op_def = mindspore::ops::GetOpDef(prim->name());
  AbstractBasePtrList list;
  for (size_t i = 0; i < args.size(); ++i) {
    mindspore::ValuePtr converted = nullptr;
    py::object param_obj = py::reinterpret_borrow<py::object>(args[i]);
    bool is_stub = false;
    if (py::isinstance<mindspore::Monad>(param_obj)) {
      *monad_count = *monad_count + 1;
    }
    *has_tensor = HasTensor(param_obj);
    converted = convertData(param_obj, is_stub, op_def, i);
    converted = ConvertArgByArgHandler(converted, op_def, i);
    auto arg = mindspore::abstract::ToAbstract(converted, nullptr, nullptr);
    list.push_back(arg);
  }
  return list;
}

void GeneratePrimitiveArgs(PrimitivePtr prim, std::vector<PyObject *> *list, PyObject *py_primitive) {
  auto op_def = mindspore::ops::GetOpDef(prim->name());
  if (op_def == nullptr) {
    return;
  }
  std::vector<ops::OpInputArg> op_call_args;
  std::vector<ops::OpInputArg> op_init_args;
  auto op_args = op_def->args_;
  for (const auto &op_arg : op_args) {
    if (op_arg.as_init_arg_) {
      op_init_args.emplace_back(op_arg);
    } else {
      op_call_args.emplace_back(op_arg);
    }
  }
  size_t args_size = list->size();
  if (args_size < op_call_args.size()) {
    for (size_t i = args_size; i < op_call_args.size(); i++) {
      auto default_value = parse::GetArgDefaultValue(prim->name(), op_call_args[i].arg_name_);
      if (default_value == nullptr) {
        continue;
      }
      auto arg_value = ValueToPyData(default_value);
      list->push_back(arg_value.ptr());
    }
  }
  auto obj = py_primitive;
  for (const auto &op_arg : op_init_args) {
    auto arg_name = common::SafeCStr(op_arg.arg_name_);
    if (py::hasattr(obj, arg_name)) {
      py::object arg_value = py::getattr(obj, arg_name);
      if (arg_value.ptr() == nullptr) {
        continue;
      }
      list->push_back(arg_value.ptr());
    }
  }
}

namespace {
mindspore::PrimitivePtr GetPrim(PyObject *primitive) {
  MS_EXCEPTION_IF_NULL(primitive);

  bool isPrimitiveFunction = py::hasattr(primitive, PYTHON_PRIMITIVE_FUNCTION_FLAG);
  py::object adapter_obj = py::reinterpret_borrow<py::object>(primitive);

  mindspore::PrimitivePtr prim = nullptr;

  if (isPrimitiveFunction) {
    PrimitiveFunctionAdapterPtr prim_func_adapter = adapter_obj.cast<PrimitiveFunctionAdapterPtr>();
    MS_EXCEPTION_IF_NULL(prim_func_adapter);
    PrimitivePtr cpp_primitive_func = prim_func_adapter->attached_primitive_function();
    if (cpp_primitive_func == nullptr) {
      std::string prim_name = py::getattr(primitive, "name").cast<std::string>();
      prim = std::make_shared<Primitive>(prim_name);
    } else {
      prim = cpp_primitive_func;
    }
  } else {
    mindspore::PrimitivePyAdapterPtr prim_adapter = adapter_obj.cast<mindspore::PrimitivePyAdapterPtr>();
    mindspore::PrimitivePyPtr primitive_py = prim_adapter->attached_primitive();
    if (primitive_py == nullptr) {
      primitive_py = std::make_shared<mindspore::PrimitivePy>(adapter_obj);
      prim_adapter->set_attached_primitive(primitive_py);
    }
    prim = primitive_py;
  }

  return prim;
}

PyObject *InferByAbstract(const AbstractBasePtr &abs, bool *is_abstract) {
  MS_EXCEPTION_IF_NULL(abs);
  MS_EXCEPTION_IF_NULL(is_abstract);

  py::object pyObj;
  if (abs != nullptr) {
    pyObj = AbstractWrapper::ConvertToPyObject(abs);
    if (pyObj.ptr() == nullptr) {
      pyObj = MakeObjectFromAbstract(abs->BuildShape(), abs->BuildType(), is_abstract);
    }
    if (pyObj.ptr() != nullptr) {
      pyObj = ConvertCppTensorToMsTensor(pyObj);
    }
  }
  return pyObj.inc_ref().ptr();
}

PyObject *InferByPrimitive(PyObject *primitive, bool has_tensor, int monad_count,
                           const std::vector<PyObject *> &arglist, bool *is_abstract) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(is_abstract);

  py::object adapter_obj = py::reinterpret_borrow<py::object>(primitive);

  if (py::hasattr(primitive, PY_PRIM_METHOD_INFER)) {
    size_t list_count = arglist.size() - size_t(monad_count);
    py::tuple py_vals(list_count);
    for (size_t i = 0; i < list_count; ++i) {
      py_vals[i] = py::reinterpret_borrow<py::object>(arglist[i]);
    }
    auto infer_func = adapter_obj.attr(PY_PRIM_METHOD_INFER);
    py::dict output = infer_func(*py_vals);
    if (output[ATTR_VALUE].is_none()) {
      auto ret = MakeObjectFromPyObject(output[ATTR_SHAPE], output[ATTR_DTYPE], is_abstract);
      Py_INCREF(ret.ptr());
      return ret.ptr();
    } else {
      Py_INCREF(output[ATTR_VALUE].ptr());
      return output[ATTR_VALUE].ptr();
    }
  } else if (!has_tensor && py::hasattr(primitive, PY_PRIM_METHOD_INFER_VALUE)) {
    // Tensor maybe uninitialized, avoid infer value and allocate data.
    // because tensor has no data when doing inference for type, infer_value will crash!
    py::tuple py_vals(arglist.size());
    for (size_t i = 0; i < arglist.size(); ++i) {
      py_vals[i] = py::reinterpret_borrow<py::object>(arglist[i]);
    }
    auto infer_value = adapter_obj.attr(PY_PRIM_METHOD_INFER_VALUE);
    auto output = infer_value(*py_vals);
    Py_INCREF(output.ptr());
    return output.ptr();
  }
  return nullptr;
}
}  // namespace

// return new reference
PyObject *InferEngine::InferPrimitive(PyObject *primitive, const std::vector<PyObject *> &args, bool *is_abstract) {
  if (!SupportInfer(primitive)) {
    return nullptr;
  }
  int monad_count = 0;
  bool has_tensor = false;
  std::vector<PyObject *> arglist = args;
  mindspore::PrimitivePtr prim = GetPrim(primitive);

  PyObject *special_type = InferSpecialPrimitive(primitive, arglist);
  if (special_type != nullptr) {
    return special_type;
  }
  GeneratePrimitiveArgs(prim, &arglist, primitive);
  AbstractBasePtrList list = ChangeAbstractArgList(prim, arglist, &has_tensor, &monad_count);

  if (!PromotePrimitiveInputsType(prim, &list)) {
    return nullptr;
  }

  *is_abstract = false;
  std::optional<AbstractBasePtr> opt_res = mindspore::abstract::TryInferAbstract(prim, list);
  if (opt_res.has_value()) {
    auto abs = opt_res.value();
    return InferByAbstract(abs, is_abstract);
  } else if (primitive) {
    return InferByPrimitive(primitive, has_tensor, monad_count, arglist, is_abstract);
  }

  return nullptr;
}

static PyObject *InferShape(PyObject *, const std::vector<PyObject *> &args) {
  PyObject *arg = args[0];
  ShapeVector shape;

  auto pyObj = py::cast<py::object>(arg);
  auto tensor_ptr = tensor::ConvertToTensor(pyObj);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  shape = tensor_ptr->shape();

  PyObject *tuple = PyTuple_New(shape.size());
  for (size_t it = 0; it < shape.size(); ++it) {
    py::int_ ss(shape[it]);
    Py_INCREF(ss.ptr());
    PyTuple_SetItem(tuple, it, ss.ptr());
  }
  return tuple;
}

static PyObject *InferDType(PyObject *, const std::vector<PyObject *> &args) {
  PyObject *arg = args[0];
  mindspore::TypePtr dtype;
  auto pyObj = py::cast<py::object>(arg);
  auto tensor_ptr = tensor::ConvertToTensor(pyObj);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  dtype = tensor_ptr->Dtype();
  PyObject *type = nullptr;
  if (g_type2attr.find(dtype->type_id()) != g_type2attr.end()) {
    type = PyObject_GetAttrString(GetMsType(), g_type2attr[dtype->type_id()].c_str());
  } else {
    MS_LOG(EXCEPTION) << "Cannot find suitable type for " << dtype->ToString();
    return nullptr;
  }
  return type;
}

static PyObject *InferRank(PyObject *, const std::vector<PyObject *> &args) {
  PyObject *arg = args[0];
  ShapeVector shape;
  auto pyObj = py::cast<py::object>(arg);
  auto tensor_ptr = tensor::ConvertToTensor(pyObj);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  shape = tensor_ptr->shape();
  return PyLong_FromSize_t(shape.size());
}

static PyObject *InferSize(PyObject *, const std::vector<PyObject *> &args) {
  PyObject *arg = args[0];
  ShapeVector shape;
  auto pyObj = py::cast<py::object>(arg);
  auto tensor_ptr = tensor::ConvertToTensor(pyObj);
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  shape = tensor_ptr->shape();
  size_t elements = 1;
  for (size_t i = 0; i < shape.size(); i++) {
    elements *= size_t(shape[i]);
  }
  return PyLong_FromSize_t(elements);
}

const SpecialPrimitiveInferFuncMap &GetSpecialPrimitiveInferFunc() {
  constexpr const auto CallValue = [](PyObject *prim, const std::vector<PyObject *> &args) {
    PyObject *res = PyObject_Vectorcall(prim, args.data(), args.size(), nullptr);
    PyErr_Clear();
    return res;
  };
  static const SpecialPrimitiveInferFuncMap specialize = {
    {"Size", InferSize},
    {"Rank", InferRank},
    {"DType", InferDType},
    {"Shape", InferShape},
    {"TileSize", CallValue},
    {"ListToTensor", CallValue},
    {"TupleToTensor", CallValue},
    {"ScalarToTensor", CallValue},
    {"make_range", CallValue},
    {"IsShapeUnKnown", [](PyObject *, const std::vector<PyObject *> &) { Py_RETURN_FALSE; }},
  };
  return specialize;
}

PyObject *InferEngine::InferSpecialPrimitive(PyObject *primitive, const std::vector<PyObject *> &arglist) {
  std::string name = py::cast<py::object>(primitive).attr("name").cast<std::string>();
  auto iter = GetSpecialPrimitiveInferFunc().find(name);
  if (iter != GetSpecialPrimitiveInferFunc().end()) {
    return iter->second(primitive, arglist);
  }
  return nullptr;
}

bool InferEngine::SupportInfer(PyObject *primitive) {
  if (!Init()) {
    return false;
  }
  bool isPrimitiveFunction = py::hasattr(primitive, PYTHON_PRIMITIVE_FUNCTION_FLAG);
  py::object adapter_obj = py::reinterpret_borrow<py::object>(primitive);

  mindspore::PrimitivePtr prim;
  if (isPrimitiveFunction) {
    PrimitiveFunctionAdapterPtr prim_func_adapter = adapter_obj.cast<PrimitiveFunctionAdapterPtr>();
    MS_EXCEPTION_IF_NULL(prim_func_adapter);
    PrimitivePtr cpp_primitive_func = prim_func_adapter->attached_primitive_function();
    if (cpp_primitive_func == nullptr) {
      std::string prim_name = py::getattr(primitive, "name").cast<std::string>();
      prim = std::make_shared<Primitive>(prim_name);
    } else {
      prim = cpp_primitive_func;
    }
  } else {
    mindspore::PrimitivePyAdapterPtr prim_adapter = adapter_obj.cast<mindspore::PrimitivePyAdapterPtr>();
    mindspore::PrimitivePyPtr primitive_py = prim_adapter->attached_primitive();
    if (primitive_py == nullptr) {
      primitive_py = std::make_shared<mindspore::PrimitivePy>(adapter_obj);
      prim_adapter->set_attached_primitive(primitive_py);
    }
    prim = primitive_py;
  }

  auto eval_impl = mindspore::abstract::GetPrimitiveInferImpl(prim);
  auto op_name = prim->name();
  if (eval_impl != std::nullopt && eval_impl->Get().get() != nullptr) {
    return true;
  }
  auto frontend_func_impl = ops::GetOpFrontendFuncImplPtr(op_name);
  auto op_def = ops::GetOpDef(op_name);
  if (frontend_func_impl != nullptr || op_def != nullptr) {
    return true;
  }
  if (GetSpecialPrimitiveInferFunc().find(prim->name()) != GetSpecialPrimitiveInferFunc().end()) {
    return true;
  }
  return false;
}

static bool CheckType(const char *mod_name, const char *type_name, bool check_sub_type, PyTypeObject *tp) {
  py::object cls = Utils::GetModuleAttr(mod_name, type_name);
  MS_EXCEPTION_IF_CHECK_FAIL(PyType_Check(cls.ptr()), "must be type");
  bool check_res = reinterpret_cast<PyObject *>(tp) == cls.ptr();
  if (!check_res && (check_sub_type)) {
    check_res |= (PyType_IsSubtype(tp, reinterpret_cast<PyTypeObject *>(cls.ptr())) != 0);
  }
  return check_res;
}

// sub-type check
template <>
bool IsGradOperationType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::prim::GradOperation, true>(tp);
}
template <>
bool IsVmapOperationType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::prim::VmapOperation, true>(tp);
}
template <>
bool IsShardType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::prim::Shard, true>(tp);
}
template <>
bool IsTensorType<true>(PyTypeObject *tp) {
  PyTypeObject *tar = mindspore::tensor::GetTensorPyType();
  return IsPybindType<mindspore::tensor::MetaTensor, true>(tp) || tp == tar || PyType_IsSubtype(tp, tar);
}
template <>
bool IsCellType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::Cell, true>(tp);
}
template <>
bool IsPrimitiveType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::PrimitivePyAdapter, true>(tp);
}
template <>
bool IsMetaFuncGraphType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::MetaFuncGraph, true>(tp);
}
template <>
bool IsMSDTypeType<true>(PyTypeObject *tp) {
  return IsPybindType<mindspore::Type, true>(tp);
}
// exact type check
template <>
bool IsCellListType<false>(PyTypeObject *tp) {
  return CheckType("mindspore.nn", "CellList", false, tp);
}

bool IsParameterObject(const py::handle &handle) {
  return parse::IsParameterObject(py::reinterpret_borrow<py::object>(handle));
}

bool CheckTensorDataInitialized(const py::object &py_tensor) {
  if (tensor::IsTensorPy(py_tensor)) {
    auto tensor = tensor::ConvertToTensor(py_tensor);
    return tensor->data().const_data() != nullptr;
  }

  return false;
}

bool FindTensorName(const std::string &name) {
  const auto &meth = pipeline::GetMethodMap().find(kObjectTypeTensorType)->second;
  if (meth.find(name) != meth.end()) {
    return true;
  }
  const auto &attr = pipeline::GetAttrMap().find(kObjectTypeTensorType)->second;
  if (attr.find(name) != attr.end()) {
    return true;
  }
  if (name == "device") {
    return true;
  }
  return false;
}

static AbstractBasePtr PyToAbs(py::handle handle) {
  py::object input = py::cast<py::object>(handle);
  ValuePtr value_ptr;
  if (!parse::ConvertStubData(input, &value_ptr) || value_ptr == nullptr) {
    MS_LOG(ERROR) << "can't convert argument to value ptr [" << std::string(py::str(input)) << "]";
    return nullptr;
  }
  return value_ptr->ToAbstract();
}

static std::unique_ptr<AbstractBasePtrList> MakeArgumentsAbstract(py::object callable_object, py::object args,
                                                                  py::object key_words) {
  // for cell construct
  auto callable_type = Py_TYPE(callable_object.ptr());
  if (IsCellType<true>(callable_type)) {
    callable_object = callable_object.attr("construct");
  }
  py::object signature = py::module::import("inspect").attr("signature")(callable_object).attr("bind");
  py::object bind_args = py::reinterpret_steal<py::object>(PyObject_Call(signature.ptr(), args.ptr(), key_words.ptr()));
  (void)bind_args.attr("apply_defaults")();
  args = py::tuple(bind_args.attr("args"));
  key_words = py::dict(bind_args.attr("kwargs"));

  AbstractBasePtrList list;
  for (auto value : args) {
    auto abs = PyToAbs(value);
    if (abs == nullptr) {
      return nullptr;
    }
    list.push_back(abs);
  }
  if (key_words.ptr() == nullptr) {
    return std::make_unique<AbstractBasePtrList>(std::move(list));
  }

  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(key_words.ptr(), &pos, &key, &value)) {
    auto abs = PyToAbs(value);
    if (abs == nullptr) {
      return nullptr;
    }
    list.push_back(std::make_shared<abstract::AbstractKeywordArg>(PyUnicode_AsUTF8(key), abs));
  }
  return std::make_unique<AbstractBasePtrList>(std::move(list));
}

const std::vector<Signature> &GetSignature(const ValuePtr &function) {
  static const auto empty = std::vector<Signature>();
  if (function->isa<Primitive>() && function->cast<PrimitivePtr>()->has_signature()) {
    return function->cast<PrimitivePtr>()->signatures();
  } else if (function->isa<MetaFuncGraph>()) {
    return function->cast<MetaFuncGraphPtr>()->signatures();
  }
  return empty;
}

void GetTypeInfo(const std::vector<TypePtr> &input_types, std::vector<TypeId> *args_type_id,
                 std::vector<bool> *args_has_tensor) {
  for (const auto &arg_type : input_types) {
    MS_EXCEPTION_IF_NULL(arg_type);
    if (arg_type->isa<Number>()) {
      (void)args_type_id->emplace_back(arg_type->cast<NumberPtr>()->type_id());
      (void)args_has_tensor->emplace_back(false);
    } else if (arg_type->isa<TensorType>()) {
      auto elem_type = arg_type->cast<TensorTypePtr>()->element();
      MS_EXCEPTION_IF_NULL(elem_type);
      (void)args_type_id->emplace_back(elem_type->type_id());
      (void)args_has_tensor->emplace_back(true);
    } else {
      (void)args_type_id->emplace_back(kTypeUnknown);
      (void)args_has_tensor->emplace_back(false);
    }
  }
}

SignatureEnumRW GetSignatureEnumRW(size_t index, const std::vector<Signature> &signature, bool has_var) {
  SignatureEnumRW sig = SignatureEnumRW::kRWDefault;
  // If sig_size is 0 use default.
  std::size_t sig_size = signature.size();
  if (index < sig_size) {
    sig = signature[index].rw;
  } else if (has_var && index >= sig_size) {
    sig = signature[sig_size - 1].rw;
  }
  return sig;
}

// Promote dtype of primitive's input args as op_def dtype_group
bool PromotePrimitiveInputsType(const ValuePtr &primitive, AbstractBasePtrList *inputs_abs_list) {
  auto &signature = GetSignature(primitive);
  std::size_t sig_size = signature.size();
  auto has_var = (sig_size > 0 && signature[sig_size - 1].kind == SignatureEnumKind::kKindVarPositional);
  std::vector<SignatureEnumDType> dtypes;
  (void)std::transform(signature.begin(), signature.end(), std::back_inserter(dtypes),
                       [](const Signature &sig) { return sig.dtype; });
  int64_t empty_dtype_count = std::count(dtypes.begin(), dtypes.end(), SignatureEnumDType::kDTypeEmptyDefaultValue);
  if (static_cast<int64_t>(dtypes.size()) == empty_dtype_count) {
    return true;
  }
  auto args_size = dtypes.size();
  if (args_size > inputs_abs_list->size()) {
    return false;
  }
  std::vector<TypePtr> input_types;
  std::set<size_t> write_indices;
  for (size_t i = 0; i < inputs_abs_list->size(); ++i) {
    input_types.push_back((*inputs_abs_list)[i]->BuildType());
    SignatureEnumRW sig = GetSignatureEnumRW(i, signature, has_var);
    if (sig == SignatureEnumRW::kRWWrite) {
      (void)write_indices.insert(i);
    }
  }
  std::vector<TypeId> source_type_id;
  std::vector<bool> source_is_tensor;
  GetTypeInfo(input_types, &source_type_id, &source_is_tensor);

  auto sig_type_map = GetSignatureTypeMap(dtypes, source_type_id, source_is_tensor, write_indices);
  for (size_t i = 0; i < args_size; ++i) {
    auto it = sig_type_map.find(dtypes[i]);
    if (it == sig_type_map.end()) {
      continue;
    }
    auto build_value = (*inputs_abs_list)[i]->BuildValue();
    if (build_value->isa<Scalar>()) {
      (*inputs_abs_list)[i] = pynative::CastUtils::ScalarToDstDtypeValue(build_value, it->second)->ToAbstract();
    } else if (build_value->isa<tensor::Tensor>() && it->second.second) {  // is tensor
      (*inputs_abs_list)[i] = pynative::CastUtils::TensorToDstDtypeValue(build_value, it->second.first)->ToAbstract();
    }
  }
  return true;
}

py::object EvalMSAPIValue(const py::object &ms_api, const py::object &args, const py::object &key_words) {
  py::object callable_object = ms_api;
  ValuePtr func_graph;
  if (!parse::ConvertData(callable_object, &func_graph) || func_graph == nullptr) {
    MS_LOG(ERROR) << "can't convert callable object to value ptr [" << std::string(py::str(callable_object)) << "]";
    return py::object();
  }

  auto inputs_ptr = MakeArgumentsAbstract(callable_object, args, key_words);
  if (inputs_ptr == nullptr) {
    return py::object();
  }

  AbstractBasePtrList inputs_abs_list = std::move(*inputs_ptr);
  AbstractBasePtr eval_result;
  if (func_graph->isa<Primitive>()) {
    if (!PromotePrimitiveInputsType(func_graph, &inputs_abs_list)) {
      return py::object();
    }
    auto eval_res = abstract::EvalOnePrim(func_graph->cast<PrimitivePtr>(), inputs_abs_list);
    eval_result = eval_res == nullptr ? nullptr : eval_res->abstract();
  } else if (func_graph->ToAbstract()->isa<abstract::AbstractFunction>()) {
    for (size_t i = 0, size = inputs_abs_list.size(); i != size; ++i) {
      inputs_abs_list[i] = inputs_abs_list[i]->Broaden();
    }
    try {
      auto analyze_res = pipeline::AbstractAnalyzeWithResourceClean(func_graph, inputs_abs_list);
      eval_result = analyze_res.eval_result == nullptr ? nullptr : analyze_res.eval_result->abstract();
    } catch (const std::exception &ex) {
      MS_LOG(ERROR) << "AbstractAnalyze failed for [" << func_graph->ToString() << "], error:" << ex.what();
    }
  }
  if (eval_result == nullptr) {
    MS_LOG(ERROR) << "eval callable object failed [" << std::string(py::str(callable_object)) << "]";
    return py::object();
  }
  py::object res = AbstractWrapper::ConvertToPyObject(eval_result);
  if (res.ptr() == nullptr) {
    MS_LOG(ERROR) << "can't convert AbstractBasePtr to PyObject [" << eval_result->ToString() << "]";
    return py::object();
  }
  return ConvertCppTensorToMsTensor(res);
}

}  // namespace pijit
}  // namespace mindspore
