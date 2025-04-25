/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/ps/static_analysis/prim_utils.h"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

#include "frontend/operator/composite/do_signature.h"
#include "frontend/operator/ops.h"
#include "frontend/operator/composite/functional_overload.h"
#include "include/common/utils/primfunc_utils.h"
#include "ir/core_ops_primitive.h"
#include "pipeline/jit/ps/fallback.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/jit/ps/parse/parse_base.h"
#include "utils/flags.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace abstract {

namespace {

py::dict ConvertAbstractToPython(const AbstractBasePtr &abs_base, bool only_convert_value = false);

py::object BuildPyObject(const ValuePtr &value_ptr) {
  if (value_ptr == nullptr) {
    return py::none();
  } else {
    return ValueToPyData(value_ptr);
  }
}

py::object AbstractTupleValueToPython(const AbstractTuple *tuple_abs) {
  MS_EXCEPTION_IF_NULL(tuple_abs);
  if (tuple_abs->dynamic_len()) {
    return py::none();
  }
  const auto &elements = tuple_abs->elements();
  size_t len = elements.size();
  py::tuple value_tuple(len);
  for (size_t i = 0; i < len; ++i) {
    value_tuple[i] = ConvertAbstractToPython(elements[i], true)[ATTR_VALUE];
  }
  return value_tuple;
}

py::dict AbstractTupleToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  auto arg_tuple = dyn_cast_ptr<AbstractTuple>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_tuple);
  auto dic = py::dict();
  if (only_convert_value) {
    dic[ATTR_VALUE] = AbstractTupleValueToPython(arg_tuple);
    return dic;
  }
  if (arg_tuple->dynamic_len()) {
    dic[ATTR_VALUE] = py::none();
    dic[ATTR_SHAPE] = ShapeVector{abstract::Shape::kShapeDimAny};
    dic[ATTR_DTYPE] = arg_tuple->BuildType();
    return dic;
  }
  size_t len = arg_tuple->size();
  py::tuple shape_tuple(len);
  py::tuple dtype_tuple(len);
  py::tuple value_tuple(len);
  std::vector<py::dict> res;

  for (size_t i = 0; i < len; i++) {
    py::dict out = ConvertAbstractToPython(arg_tuple->elements()[i]);
    res.push_back(out);
    shape_tuple[i] = out[ATTR_SHAPE];
    dtype_tuple[i] = out[ATTR_DTYPE];
    value_tuple[i] = out[ATTR_VALUE];
  }
  dic[ATTR_SHAPE] = shape_tuple;
  dic[ATTR_DTYPE] = dtype_tuple;
  dic[ATTR_VALUE] = value_tuple;

  return dic;
}

py::dict AbstractDictionaryToPython(const AbstractBasePtr &abs_base) {
  auto arg_dict = dyn_cast_ptr<AbstractDictionary>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_dict);

  size_t len = arg_dict->size();
  const auto &arg_dict_elements = arg_dict->elements();
  py::list shape_list(len);
  py::list dtype_list(len);
  py::dict value_dict = py::dict();

  for (size_t i = 0; i < len; ++i) {
    auto cur_attr = arg_dict_elements[i];
    auto cur_key = cur_attr.first;
    auto cur_value = cur_attr.second;

    py::dict cur_value_out = ConvertAbstractToPython(cur_value);
    shape_list[i] = cur_value_out[ATTR_SHAPE];
    dtype_list[i] = cur_value_out[ATTR_DTYPE];
    MS_EXCEPTION_IF_NULL(cur_key);
    value_dict[ValueToPyData(cur_key->BuildValue())] = cur_value_out[ATTR_VALUE];
  }

  py::dict dic = py::dict();
  dic[ATTR_SHAPE] = shape_list;
  dic[ATTR_DTYPE] = dtype_list;
  MS_EXCEPTION_IF_NULL(arg_dict->BuildValue());
  dic[ATTR_VALUE] = value_dict;
  return dic;
}

py::object AbstractKWArgsToPython(const AbstractBasePtr &abs_base) {
  MS_EXCEPTION_IF_NULL(abs_base);
  auto abs_keyword_arg = abs_base->cast_ptr<abstract::AbstractKeywordArg>();
  MS_EXCEPTION_IF_NULL(abs_keyword_arg);
  auto args_abs = abs_keyword_arg->get_arg();
  auto args_obj = BuildPyObject(args_abs->BuildValue());
  // if the args is none but the type is not none means the input is a variable.
  if (!args_abs->isa<AbstractNone>() && py::isinstance<py::none>(args_obj)) {
    return py::none();
  }
  return BuildPyObject(abs_base->BuildValue());
}

py::object AbstractListValueToPython(const AbstractList *list_abs) {
  MS_EXCEPTION_IF_NULL(list_abs);
  if (list_abs->dynamic_len()) {
    return py::none();
  }
  const auto &elements = list_abs->elements();
  size_t len = elements.size();
  py::list value_list(len);
  for (size_t i = 0; i < len; ++i) {
    value_list[i] = ConvertAbstractToPython(elements[i], true)[ATTR_VALUE];
  }
  return value_list;
}

py::dict AbstractListToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  auto arg_list = dyn_cast_ptr<AbstractList>(abs_base);
  MS_EXCEPTION_IF_NULL(arg_list);
  auto dic = py::dict();
  if (only_convert_value) {
    dic[ATTR_VALUE] = AbstractListValueToPython(arg_list);
    return dic;
  }
  if (arg_list->dynamic_len()) {
    auto elem_out = ConvertAbstractToPython(arg_list->dynamic_len_element_abs());
    dic[ATTR_VALUE] = py::none();
    dic[ATTR_SHAPE] = elem_out[ATTR_SHAPE];
    dic[ATTR_DTYPE] = elem_out[ATTR_DTYPE];
    return dic;
  }
  size_t len = arg_list->size();
  py::list shape_list(len);
  py::list dtype_list(len);
  py::list value_list(len);
  std::vector<py::dict> res;

  for (size_t i = 0; i < len; i++) {
    py::dict out = ConvertAbstractToPython(arg_list->elements()[i]);
    res.push_back(out);
    shape_list[i] = out[ATTR_SHAPE];
    dtype_list[i] = out[ATTR_DTYPE];
    value_list[i] = out[ATTR_VALUE];
  }

  dic[ATTR_SHAPE] = shape_list;
  dic[ATTR_DTYPE] = dtype_list;
  dic[ATTR_VALUE] = value_list;
  return dic;
}

void ConvertAbstractTensorToPython(const AbstractBasePtr &abs_base, bool only_convert_value, py::dict *dic) {
  auto arg_tensor = dyn_cast_ptr<AbstractTensor>(abs_base);
  MS_EXCEPTION_IF_NULL(dic);
  MS_EXCEPTION_IF_NULL(arg_tensor);
  if (only_convert_value) {
    (*dic)[ATTR_VALUE] = BuildPyObject(arg_tensor->BuildValue());
    return;
  }
  MS_EXCEPTION_IF_NULL(arg_tensor->shape());
  (*dic)[ATTR_SHAPE] = arg_tensor->shape()->shape();

  (*dic)[ATTR_DTYPE] = arg_tensor->BuildType();
  (*dic)[ATTR_VALUE] = BuildPyObject(arg_tensor->BuildValue());
}

namespace {
py::object GetPyObjForPrimitiveAbstract(const PrimitiveAbstractClosurePtr &prim_abs) {
  MS_EXCEPTION_IF_NULL(prim_abs);
  auto prim = prim_abs->BuildValue();
  if (prim == nullptr) {
    return py::none();
  }
  if (prim->isa<prim::DoSignaturePrimitive>()) {
    auto do_sig_prim = prim->cast_ptr<prim::DoSignaturePrimitive>();
    auto value = do_sig_prim->function();
    MS_EXCEPTION_IF_NULL(value);
    if (!value->isa<PrimitivePy>()) {
      return py::none();
    }
    auto prim_py = value->cast_ptr<PrimitivePy>();
    return prim_py->GetPyObj();
  }
  if (prim->isa<PrimitivePy>()) {
    auto prim_py = prim->cast_ptr<PrimitivePy>();
    return prim_py->GetPyObj();
  }
  return py::none();
}
}  // namespace

void ConvertAbstractFunctionToPython(const AbstractBasePtr &abs_base, py::dict *dic) {
  MS_EXCEPTION_IF_NULL(dic);
  MS_EXCEPTION_IF_NULL(abs_base);
  (*dic)[ATTR_SHAPE] = py::none();
  (*dic)[ATTR_DTYPE] = abs_base->BuildType();
  (*dic)[ATTR_VALUE] = py::none();
  if (abs_base->isa<PartialAbstractClosure>()) {
    auto partial_abs = abs_base->cast<PartialAbstractClosurePtr>();
    AbstractBasePtrList args = partial_abs->args();
    if (!args.empty()) {
      auto value = args[0]->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      auto value_obj = value->cast_ptr<parse::ClassType>();
      if (value_obj != nullptr) {
        (*dic)[ATTR_DTYPE] = std::make_shared<TypeType>();
        (*dic)[ATTR_VALUE] = value_obj->obj();
      }
    }
  }
  if (abs_base->isa<PrimitiveAbstractClosure>()) {
    (*dic)[ATTR_VALUE] = GetPyObjForPrimitiveAbstract(abs_base->cast<PrimitiveAbstractClosurePtr>());
  }
}

void UnknownAbstract(const AbstractBasePtr &abs_base) {
  auto value = abs_base->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  if ((*value == *kValueAny)) {
    auto value_desc = abs_base->value_desc();
    MS_EXCEPTION(TypeError) << "Unsupported parameter " << (value_desc.empty() ? "type" : value_desc)
                            << " for python primitive." << abs_base->ToString();
  }
  MS_EXCEPTION(TypeError) << "Unsupported parameter type for python primitive, the parameter value is "
                          << value->ToString();
}

py::dict ConvertAbstractToPython(const AbstractBasePtr &abs_base, bool only_convert_value) {
  MS_EXCEPTION_IF_NULL(abs_base);
  auto dic = py::dict();
  if (abs_base->isa<AbstractTensor>()) {
    ConvertAbstractTensorToPython(abs_base, only_convert_value, &dic);
  } else if (abs_base->isa<AbstractScalar>() || abs_base->isa<AbstractType>()) {
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(abs_base->BuildValue());
  } else if (abs_base->isa<AbstractTuple>()) {
    return AbstractTupleToPython(abs_base, only_convert_value);
  } else if (abs_base->isa<AbstractList>()) {
    return AbstractListToPython(abs_base, only_convert_value);
  } else if (abs_base->isa<AbstractDictionary>()) {
    return AbstractDictionaryToPython(abs_base);
  } else if (abs_base->isa<AbstractSlice>()) {
    auto arg_slice = dyn_cast_ptr<AbstractSlice>(abs_base);
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = arg_slice->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg_slice->BuildValue());
  } else if (abs_base->isa<AbstractRowTensor>()) {
    auto arg = dyn_cast_ptr<AbstractRowTensor>(abs_base);
    MS_EXCEPTION_IF_NULL(arg->shape());
    dic[ATTR_SHAPE] = arg->shape()->shape();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg->BuildValue());
  } else if (abs_base->isa<AbstractCOOTensor>()) {
    auto arg = dyn_cast_ptr<AbstractCOOTensor>(abs_base);
    MS_EXCEPTION_IF_NULL(arg->shape());
    AbstractBasePtrList sparse_shape = arg->shape()->elements();
    ShapeVector sparse_shape_vector;
    (void)std::transform(sparse_shape.begin(), sparse_shape.end(), std::back_inserter(sparse_shape_vector),
                         [](const AbstractBasePtr &e) -> int64_t {
                           MS_EXCEPTION_IF_NULL(e);
                           MS_EXCEPTION_IF_NULL(e->cast_ptr<AbstractScalar>());
                           ValuePtr value = e->cast_ptr<AbstractScalar>()->BuildValue();
                           return GetValue<int64_t>(value);
                         });
    dic[ATTR_SHAPE] = sparse_shape_vector;
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg->BuildValue());
  } else if (abs_base->isa<AbstractCSRTensor>()) {
    auto arg = dyn_cast_ptr<AbstractCSRTensor>(abs_base);
    MS_EXCEPTION_IF_NULL(arg->shape());
    AbstractBasePtrList sparse_shape = arg->shape()->elements();
    ShapeVector sparse_shape_vector;
    (void)std::transform(sparse_shape.begin(), sparse_shape.end(), std::back_inserter(sparse_shape_vector),
                         [](const AbstractBasePtr &e) -> int64_t {
                           MS_EXCEPTION_IF_NULL(e);
                           MS_EXCEPTION_IF_NULL(e->cast_ptr<AbstractScalar>());
                           ValuePtr value = e->cast_ptr<AbstractScalar>()->BuildValue();
                           return GetValue<int64_t>(value);
                         });
    dic[ATTR_SHAPE] = sparse_shape_vector;
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg->BuildValue());
  } else if (abs_base->isa<AbstractEllipsis>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::ellipsis();
    dic[ATTR_VALUE] = py::ellipsis();
  } else if (abs_base->isa<AbstractNone>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = py::none();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractFunction>()) {
    ConvertAbstractFunctionToPython(abs_base, &dic);
  } else if (abs_base->isa<AbstractClass>()) {
    auto arg_class = dyn_cast_ptr<AbstractClass>(abs_base);
    ShapeVector shape;
    dic[ATTR_SHAPE] = shape;
    dic[ATTR_DTYPE] = arg_class->BuildType();
    dic[ATTR_VALUE] = BuildPyObject(arg_class->BuildValue());
  } else if (abs_base->isa<AbstractUndetermined>()) {
    auto arg = dyn_cast_ptr<AbstractUndetermined>(abs_base);
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = arg->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractMonad>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = py::none();
  } else if (abs_base->isa<AbstractKeywordArg>()) {
    dic[ATTR_SHAPE] = py::none();
    dic[ATTR_DTYPE] = abs_base->BuildType();
    dic[ATTR_VALUE] = AbstractKWArgsToPython(abs_base);
  } else {
    UnknownAbstract(abs_base);
  }
  return dic;
}

void CheckCustomPrimOutputInferResult(const PrimitivePtr &prim, const AbstractBasePtr &res_spec) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(res_spec);
  const string kOutputNum = "output_num";
  if (prim->IsCustomPrim()) {
    // Raise error if output_num is not match the infer result.
    auto output_num_value = prim->GetAttr(kOutputNum);
    if (output_num_value == nullptr) {
      MS_LOG(DEBUG) << "The output num may no need to check";
      return;
    }
    int64_t output_num = GetValue<int64_t>(output_num_value);
    if (res_spec->isa<AbstractTensor>() && output_num != 1) {
      MS_LOG(EXCEPTION) << "Custom operator primitive[" << prim->ToString()
                        << "]'s attribute[output_num]: " << output_num << ", not matches the infer result "
                        << res_spec->ToString();
    } else if (res_spec->isa<AbstractTuple>() &&
               (res_spec->cast_ptr<AbstractTuple>()->size() != LongToSize(output_num))) {
      MS_LOG(EXCEPTION) << "Custom operator primitive[" << prim->ToString()
                        << "]'s attribute[output_num]: " << output_num << ", not matches the infer result "
                        << res_spec->ToString();
    }
  }
}

static bool IsMonadType(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    return type->isa<MonadType>();
  }
  return false;
}

AbstractBasePtr ToMonadAbstract(const py::object &type_obj) {
  if (py::isinstance<Type>(type_obj)) {
    auto type = type_obj.cast<Type *>();
    if (!type->isa<MonadType>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Not a monad type object: " << py::str(type_obj);
    }
    return abstract::MakeMonadAbstract(type->cast<MonadTypePtr>());
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Not a type object: " << py::str(type_obj);
}

py::object GetPyAbsItemOfTupleOut(const py::object &output, const size_t index) {
  auto out_dict = output.cast<py::dict>();
  auto type_obj = out_dict[ATTR_DTYPE];
  auto shape_obj = out_dict[ATTR_SHAPE];
  auto out_item = py::dict();
  auto shape_tuple = shape_obj.cast<py::tuple>();
  auto typeid_tuple = type_obj.cast<py::tuple>();
  out_item[ATTR_DTYPE] = typeid_tuple[index];
  out_item[ATTR_SHAPE] = shape_tuple[index];
  out_item[ATTR_VALUE] = py::none();
  return out_item;
}

AbstractBasePtr MakePyInferRes2AbstractTensor(const py::object &shape_obj, const py::object &type_obj) {
  auto res_vec = shape_obj.cast<ShapeVector>();
  auto res_dtype = type_obj.cast<TypePtr>();

  auto res_shape = std::make_shared<abstract::Shape>(res_vec);
  AbstractBasePtr tensor = MakeAbstractTensor(res_shape, res_dtype);
  return tensor;
}

AbstractBasePtr MakePyInferRes2Abstract(const py::object &output) {
  auto out_dict = output.cast<py::dict>();
  auto type_obj = out_dict[ATTR_DTYPE];
  auto shape_obj = out_dict[ATTR_SHAPE];
  if ((py::isinstance<py::list>(shape_obj) || py::isinstance<py::tuple>(shape_obj)) && py::isinstance<Type>(type_obj)) {
    auto res_vec = shape_obj.cast<ShapeVector>();
    auto res_dtype = type_obj.cast<TypePtr>();
    MS_EXCEPTION_IF_NULL(res_dtype);
    // if the size of shape list is empty, return an scalar abstract
    if (res_vec.empty() && (!res_dtype->isa<TensorType>())) {
      abstract::AbstractScalarPtr abs_scalar = std::make_shared<abstract::AbstractScalar>(kValueAny, res_dtype);
      return abs_scalar;
    }
    return MakePyInferRes2AbstractTensor(shape_obj, type_obj);
  } else if (py::isinstance<py::tuple>(shape_obj) && py::isinstance<py::tuple>(type_obj)) {
    auto typeid_tuple = type_obj.cast<py::tuple>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < typeid_tuple.size(); ++it) {
      auto output_it = GetPyAbsItemOfTupleOut(output, it);
      auto tensor_it = MakePyInferRes2Abstract(output_it);
      ptr_list.push_back(tensor_it);
    }
    auto tuple = std::make_shared<abstract::AbstractTuple>(ptr_list);
    return tuple;
  } else if (py::isinstance<py::list>(shape_obj) && py::isinstance<py::list>(type_obj)) {
    auto typeid_list = type_obj.cast<py::list>();
    AbstractBasePtrList ptr_list;
    for (size_t it = 0; it < typeid_list.size(); ++it) {
      auto output_it = GetPyAbsItemOfTupleOut(output, it);
      auto tensor_it = MakePyInferRes2Abstract(output_it);
      ptr_list.push_back(tensor_it);
    }
    auto list = std::make_shared<abstract::AbstractList>(ptr_list);
    return list;
  } else if (shape_obj.is_none() && type_obj.is_none()) {
    // AbstractNone indicates there is no output for this CNode node.
    auto abstract_none = std::make_shared<abstract::AbstractNone>();
    return abstract_none;
  } else if (IsMonadType(type_obj)) {
    // Return monad abstract if it is monad type.
    return ToMonadAbstract(type_obj);
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Python evaluator return invalid shape or type. " << py::str(type_obj);
  }
}
}  // namespace

py::tuple PreparePyInputs(const AbstractBasePtrList &args) {
  // The monad parameter is defined at the end of the parameter and needs to be ignored
  std::size_t args_size = args.size() - GetAbstractMonadNum(args);
  py::tuple py_args(args_size);
  for (size_t i = 0; i < args_size; i++) {
    py_args[i] = ConvertAbstractToPython(args[i]);
  }
  return py_args;
}

AbstractBasePtr PyInferRes2Abstract(const PrimitivePyPtr &prim_py, const py::dict &output) {
  // Convert to AbstractValue based on type and shape
  if (output[ATTR_VALUE].is_none()) {
    return MakePyInferRes2Abstract(output);
  }

  // Convert pyobject to Value, then to AbstractValue
  auto out_dtype = output[ATTR_DTYPE];
  TypePtr dtype = py::isinstance<Type>(out_dtype) ? out_dtype.cast<TypePtr>() : nullptr;
  ValuePtr converted_ret = nullptr;
  bool converted = parse::ConvertData(output[ATTR_VALUE], &converted_ret, false, dtype);
  if (!converted) {
    MS_LOG(INTERNAL_EXCEPTION) << "Convert data failed";
  }
  auto res_spec = FromValue(converted_ret);
  MS_EXCEPTION_IF_NULL(res_spec);
  if (res_spec->isa<AbstractTensor>()) {
    // Replace to tensor constant node in specialize
    auto res_tensor = res_spec->cast<AbstractTensorPtr>();
    res_tensor->set_value(converted_ret);
  }
  CheckCustomPrimOutputInferResult(prim_py, res_spec);
  return res_spec;
}

AnfNodePtrList GetPrimitiveInitArgs(const PrimitivePyPtr &prim_py, const ops::OpDef *op_def) {
  MS_EXCEPTION_IF_NULL(prim_py);
  MS_EXCEPTION_IF_NULL(op_def);

  std::vector<AnfNodePtr> prim_init_arg_nodes;
  auto obj = prim_py->GetPyObj();

  for (const auto &op_arg : op_def->args_) {
    if (op_arg.as_init_arg_) {
      auto arg_name = op_arg.arg_name_;
      py::object arg_value = py::getattr(obj, common::SafeCStr(arg_name));
      ValuePtr converted_ret = nullptr;
      bool converted = parse::ConvertData(arg_value, &converted_ret);
      if (!converted) {
        MS_LOG(INTERNAL_EXCEPTION) << "Cannot convert initialization arg: (" << arg_name << ": " << py::str(arg_value)
                                   << ") in Primitive '" << prim_py->name() << "'.";
      }
      (void)prim_init_arg_nodes.emplace_back(NewValueNode(converted_ret));
    }
  }
  MS_LOG(DEBUG) << "PrimitivePy " << prim_py->name() << " has " << prim_init_arg_nodes.size() << " __init__() args";
  return prim_init_arg_nodes;
}

bool ValidateArgOptional(const AbstractBasePtr &abs_arg, const ops::OpInputArg &input_arg) {
  if (!input_arg.is_optional_) {
    return false;
  }

  auto abs_type = abs_arg->BuildType();
  MS_EXCEPTION_IF_NULL(abs_type);
  return abs_type->isa<TypeNone>();
}

bool ValidateArgSpecialType(const std::string &op_name, const AbstractBasePtr &abs, const ops::OpInputArg &op_arg) {
  if (abs->isa<abstract::AbstractKeywordArg>()) {
    MS_EXCEPTION(TypeError) << "For Primitive[" << op_name
                            << "], only positional arguments as inputs are supported, but got " << abs->ToString();
  }
  return fallback::ContainsSequenceAnyType(abs) || ValidateArgOptional(abs, op_arg) ||
         ops::ValidateArgsType(abs, op_arg.arg_dtype_);
}

void GetKeywordArgsMap(const AbstractBasePtr &input_abs, const std::vector<ops::OpInputArg> &op_args,
                       const AnfNodePtr &input, const FuncGraphPtr &graph, std::map<std::string, AnfNodePtr> *key_map) {
  auto input_kwarg_abs = input_abs->cast<AbstractKeywordArgPtr>();
  const auto &key = input_kwarg_abs->get_key();
  bool is_key_valid = std::any_of(op_args.begin(), op_args.end(),
                                  [&key](const ops::OpInputArg &op_arg) { return key == op_arg.arg_name_; });
  if (is_key_valid) {
    const auto &kwarg_value = graph->NewCNode({NewValueNode(prim::kPrimExtractKeywordArg), NewValueNode(key), input});
    (*key_map)[key] = kwarg_value;
  } else {
    MS_LOG(EXCEPTION) << "Got an unexpected keyword argument '" << key << "'.";
  }
}

AnfNodePtrList GeneratePrimitiveDefaultArgs(const std::string &op_name, const std::vector<AnfNodePtr> &args_list,
                                            const std::vector<ops::OpInputArg> &op_args,
                                            const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                            const FuncGraphPtr &graph) {
  size_t args_size = args_list.size();
  AnfNodePtrList nodes;
  std::map<std::string, AnfNodePtr> key_map;
  for (size_t idx = 0; idx < args_list.size(); ++idx) {
    auto input = args_list[idx];
    if (IsMonad(input)) {
      --args_size;
      continue;
    }
    auto input_abs = eval_func(input);
    if (input_abs->isa<AbstractKeywordArg>()) {
      GetKeywordArgsMap(input_abs, op_args, input, graph, &key_map);
    } else {
      (void)nodes.emplace_back(input);
      continue;
    }
  }
  args_size -= key_map.size();
  if (args_size < op_args.size()) {
    for (size_t i = args_size; i < op_args.size(); ++i) {
      auto arg_name = op_args[i].arg_name_;
      auto iter = key_map.find(arg_name);
      if (iter != key_map.end()) {
        MS_LOG(DEBUG) << "Get args for Primitive[" << op_name << "]: " << iter->second->DebugString();
        (void)nodes.emplace_back(iter->second);
        (void)key_map.erase(arg_name);
      } else {
        auto default_arg = parse::GetArgDefaultValue(op_name, arg_name);
        if (default_arg == nullptr) {
          break;
        }
        MS_LOG(DEBUG) << "Get the default value of '" << arg_name << "' attribute of Primitive[" << op_name
                      << "], which is " << default_arg->ToString() << ".";
        (void)nodes.emplace_back(NewValueNode(default_arg));
      }
    }
  }

  if (nodes.size() != op_args.size()) {
    std::string args_type_str = (op_args.size() != 0 && op_args[0].as_init_arg_) ? "init arguments" : "inputs";
    MS_EXCEPTION(TypeError) << "For Operator[" << op_name << "], the number of " << args_type_str
                            << " (including default arguments) should be " << op_args.size()
                            << ", but the actual number of inputs is not satisfied, which is " << args_size << ".";
  }
  return nodes;
}

namespace {

inline int64_t OpDtypeToInt(ops::OP_DTYPE dtype) { return static_cast<int64_t>(dtype); }

AnfNodePtr GetNodeAfterTypeConversion(const AnfNodePtr &node, const ops::OpInputArg &op_arg, const FuncGraphPtr &fg) {
  MS_EXCEPTION_IF_NULL(fg);
  // If src_cast_dtype is empty, do no need to do type conversion.
  if (op_arg.cast_dtype_.empty()) {
    return node;
  }
  const auto convert_func =
    prim::GetPythonOps(parse::PYTHON_MOD_PRIMITIVE_OP_TYPE_CAST, parse::PYTHON_MOD_PRIMITIVE_ARG_DTYPE_CAST_MODULE);
  auto convert_fg = dyn_cast<FuncGraph>(convert_func);
  MS_EXCEPTION_IF_NULL(convert_fg);
  convert_fg->set_manager(fg->manager());
  auto res = fg->NewCNodeInOrder({NewValueNode(convert_fg), node, NewValueNode(OpDtypeToInt(op_arg.arg_dtype_))});
  res->set_debug_info(node->debug_info());
  return res;
}

bool ValidateAndConvertArgsType(const std::string &op_name, const std::vector<ops::OpInputArg> &op_args,
                                const AbstractBasePtrList &abs_list, const FuncGraphPtr &fg,
                                std::vector<AnfNodePtr> *nodes) {
  bool exist_undetermined_arg = false;
  for (size_t i = 0; i < op_args.size(); ++i) {
    auto op_arg = op_args[i];
    auto abs_arg = abs_list[i];
    if (HasAbstractType<AbstractUndetermined>(abs_arg)) {
      exist_undetermined_arg = true;
    }
    if (ValidateArgSpecialType(op_name, abs_arg, op_arg)) {
      continue;
    }
    bool match = false;
    auto cast_dtypes = op_arg.cast_dtype_;
    for (size_t j = 0; j < cast_dtypes.size(); ++j) {
      if (ops::ValidateArgsType(abs_arg, cast_dtypes[j])) {
        (*nodes)[i] = GetNodeAfterTypeConversion((*nodes)[i], op_arg, fg);
        match = true;
        break;
      }
    }
    if (!match && !exist_undetermined_arg) {
      return false;
    }
  }
  return true;
}

AnfNodePtr GetNodeAfterArgHandler(const AnfNodePtr &node, const std::string &op_name, const ops::OpInputArg &op_arg,
                                  const AbstractBasePtr &abs, const FuncGraphPtr &fg) {
  if (op_arg.arg_handler_.empty()) {
    return node;
  }
  if (op_arg.is_optional_ && abs->isa<AbstractNone>()) {
    return node;
  }
  const auto arg_handler_func = prim::GetPythonOps(op_arg.arg_handler_, parse::PYTHON_MOD_PRIMITIVE_ARG_HANDLER_MODULE);
  MS_LOG(DEBUG) << "The arg handler function for '" << op_arg.arg_name_ << "' of Primitive[" << op_name << "] is "
                << arg_handler_func->ToString() << ".";
  if (arg_handler_func->isa<Primitive>()) {
    auto arg_handler_fg = dyn_cast<Primitive>(arg_handler_func);
    MS_EXCEPTION_IF_NULL(arg_handler_fg);
    auto res =
      fg->NewCNodeInOrder({NewValueNode(arg_handler_fg), NewValueNode(op_name), NewValueNode(op_arg.arg_name_), node});
    res->set_debug_info(node->debug_info());
    return res;
  }
  auto arg_handler_fg = dyn_cast<FuncGraph>(arg_handler_func);
  MS_EXCEPTION_IF_NULL(arg_handler_fg);
  arg_handler_fg->set_manager(fg->manager());
  auto res =
    fg->NewCNodeInOrder({NewValueNode(arg_handler_fg), NewValueNode(op_name), NewValueNode(op_arg.arg_name_), node});
  res->set_debug_info(node->debug_info());
  return res;
}

CNodePtr CheckAndConvertPrimitiveArgs(const PrimitivePtr &prim, const FuncGraphPtr &graph,
                                      const std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> &args_pair,
                                      const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func,
                                      bool is_preprocessed, const AnfNodePtr &old_cnode = nullptr) {
  auto init_args_list = args_pair.first;
  auto call_args_list = args_pair.second;
  auto prim_name = prim->name();
  auto op_def = mindspore::ops::GetOpDef(prim_name);
  MS_EXCEPTION_IF_NULL(op_def);
  MS_EXCEPTION_IF_NULL(graph);
  // Check args size.
  std::vector<ops::OpInputArg> op_call_args;
  std::vector<ops::OpInputArg> op_init_args;
  auto op_args = op_def->args_;
  for (const auto &op_arg : op_args) {
    if (op_arg.as_init_arg_) {
      (void)op_init_args.emplace_back(op_arg);
    } else {
      (void)op_call_args.emplace_back(op_arg);
    }
  }

  MS_LOG(DEBUG) << "For Primitive[" << prim_name << "], the number of init args is expected to be "
                << op_init_args.size() << ", and the number of call args is expected to be " << op_call_args.size();
  // Generate primitive default args.
  MS_LOG(DEBUG) << "For Primitive[ " << prim_name << "], before processing default args, the number of init args is "
                << init_args_list.size() << " and the number of call args is " << call_args_list.size();
  auto call_nodes = GeneratePrimitiveDefaultArgs(prim_name, call_args_list, op_call_args, eval_func, graph);
  auto init_nodes = GeneratePrimitiveDefaultArgs(prim_name, init_args_list, op_init_args, eval_func, graph);
  MS_LOG(DEBUG) << "For Primitive[ " << prim_name << "], after processing default args, the number of init args is "
                << init_args_list.size() << " and the number of call args is " << call_args_list.size();
  // If it is not preprocessed, signatures and need to be processed.
  if (!is_preprocessed) {
    // Process signatures.
    MS_LOG(DEBUG) << "Process signatures for Primitive[" << prim_name << "].";
    AbstractBasePtrList call_abs_list;
    (void)std::transform(call_nodes.cbegin(), call_nodes.cend(), std::back_inserter(call_abs_list), eval_func);
    call_nodes = prim::GetNewInputsBySignatures(graph, prim_name, prim, call_abs_list, call_nodes, old_cnode);
    // Process arg_handler.
    for (size_t i = 0; i < op_init_args.size(); ++i) {
      auto abs_node = eval_func(init_nodes[i]);
      if (!prim->HasAttr("Converted")) {
        init_nodes[i] = GetNodeAfterArgHandler(init_nodes[i], prim_name, op_init_args[i], abs_node, graph);
      }
    }
  }
  for (size_t i = 0; i < op_call_args.size(); ++i) {
    auto abs_node = eval_func(call_nodes[i]);
    if (!prim->HasAttr("Converted")) {
      call_nodes[i] = GetNodeAfterArgHandler(call_nodes[i], prim_name, op_call_args[i], abs_node, graph);
    }
  }

  // Check args type and do type conversion.
  AbstractBasePtrList call_abs_list;
  AbstractBasePtrList init_abs_list;
  (void)std::transform(call_nodes.cbegin(), call_nodes.cend(), std::back_inserter(call_abs_list), eval_func);
  (void)std::transform(init_nodes.cbegin(), init_nodes.cend(), std::back_inserter(init_abs_list), eval_func);
  MS_LOG(DEBUG) << "For Primitive[" << prim_name << "], the number of init args is " << init_nodes.size()
                << " and the number of call args is " << call_nodes.size();
  if (!ValidateAndConvertArgsType(prim_name, op_call_args, call_abs_list, graph, &call_nodes) ||
      !ValidateAndConvertArgsType(prim_name, op_init_args, init_abs_list, graph, &init_nodes)) {
    std::vector<std::string> op_type_list;
    (void)std::transform(call_abs_list.cbegin(), call_abs_list.cend(), std::back_inserter(op_type_list),
                         [](const AbstractBasePtr &op_abs) { return prim::BuildArgsTypeString(op_abs->BuildType()); });
    (void)std::transform(init_abs_list.cbegin(), init_abs_list.cend(), std::back_inserter(op_type_list),
                         [](const AbstractBasePtr &op_abs) { return prim::BuildArgsTypeString(op_abs->BuildType()); });
    MS_EXCEPTION(TypeError) << ops::BuildOpErrorMsg(op_def, op_type_list);
  }

  // Create New node.
  AnfNodePtrList input_nodes{NewValueNode(prim)};
  (void)std::copy(call_nodes.cbegin(), call_nodes.cend(), std::back_inserter(input_nodes));
  (void)std::copy(init_nodes.cbegin(), init_nodes.cend(), std::back_inserter(input_nodes));
  auto new_cnode = graph->NewCNodeInOrder(input_nodes);
  return new_cnode;
}
}  // namespace

AnfNodePtr CheckAndConvertPrimitiveArgs(const PrimitivePtr &prim,
                                        const std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr>> &args_pair,
                                        const AnalysisEnginePtr &engine, const AnfNodeConfigPtr &out_conf,
                                        bool is_preprocessed) {
  auto node = out_conf->node();
  MS_EXCEPTION_IF_NULL(node);
  auto graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);

  auto eval_func = [&engine, &out_conf](const AnfNodePtr &node) {
    AnfNodeConfigPtr config = engine->MakeConfig(node, out_conf->context(), out_conf->func_graph());
    MS_EXCEPTION_IF_NULL(config);
    const auto &eval_result = config->ObtainEvalResult();
    MS_EXCEPTION_IF_NULL(eval_result);
    return eval_result->abstract();
  };

  auto new_cnode = CheckAndConvertPrimitiveArgs(prim, graph, args_pair, eval_func, is_preprocessed, node);
  MS_LOG(DEBUG) << "Convert primitive args: " << prim->name() << ". node: " << node->DebugString()
                << ", new_node: " << new_cnode->DebugString();
  new_cnode->set_debug_info(node->debug_info());
  return new_cnode;
}

CNodePtr GeneratePrimitiveCNode(const PrimitivePtr &primitive, const ops::OpDef *op_def, const FuncGraphPtr &graph,
                                const AnfNodePtrList &init_args_nodes, const AnfNodePtrList &call_args_nodes,
                                const std::function<AbstractBasePtr(const AnfNodePtr &)> &eval_func) {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(op_def);

  auto args_pair = std::make_pair(init_args_nodes, call_args_nodes);

  // Follow the implementations in PrimitiveArgsToInputsEvaluator, convert to base Primitive, and is_preprocessed=true
  auto new_prim = std::make_shared<Primitive>(*primitive);
  auto new_cnode = CheckAndConvertPrimitiveArgs(new_prim, graph, args_pair, eval_func, true);

  MS_LOG(INFO) << "Convert primitive args: " << primitive->name() << ", new node: " << new_cnode->DebugString();
  return new_cnode;
}

std::shared_ptr<Functional> BuildMethodFunctional(const std::string &name) {
  auto functional = std::make_shared<Functional>(name);
  functional->set_is_method(true);
  return functional;
}

namespace {
bool IsSubtypeTuple(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_tuple = dyn_cast_ptr<AbstractTuple>(x);
  auto model_tuple = dyn_cast_ptr<Tuple>(model);

  if (x_tuple == nullptr || model_tuple == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  if (x_tuple->size() != model_tuple->size()) {
    return false;
  }

  for (size_t i = 0; i < x_tuple->size(); i++) {
    bool is_subtype = IsSubtype((*x_tuple)[i], (*model_tuple)[i]);
    if (!is_subtype) {
      return false;
    }
  }
  return true;
}

bool IsSubtypeArray(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_tensor = dyn_cast_ptr<AbstractTensor>(x);
  auto model_tensor = dyn_cast_ptr<TensorType>(model);

  if (x_tensor == nullptr || model_tensor == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  return IsSubtype(x_tensor->element(), model_tensor->element());
}

bool IsSubtypeList(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  auto x_list = dyn_cast_ptr<AbstractList>(x);
  auto model_list = dyn_cast_ptr<List>(model);

  if (x_list == nullptr || model_list == nullptr) {
    return false;
  }

  if (model->IsGeneric()) {
    return true;
  }

  if (x_list->size() != model_list->size()) {
    return false;
  }

  bool is_subtype = true;
  for (size_t i = 0; i < x_list->size(); i++) {
    is_subtype = IsSubtype((*x_list)[i], (*model_list)[i]);
    if (!is_subtype) {
      return false;
    }
  }
  return is_subtype;
}

inline bool IsSubtypeScalar(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  if (dyn_cast_ptr<AbstractScalar>(x) == nullptr) {
    return false;
  }
  auto &x_type = x->GetTypeTrack();
  return IsSubType(x_type, model);
}
}  // namespace

bool IsSubtype(const AbstractBasePtr x, const TypePtr model) {
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(model);
  TypeId model_typeid = model->type_id();
  switch (model_typeid) {
    case kMetaTypeObject:
      return true;
    case kObjectTypeTuple:
      return IsSubtypeTuple(x, model);
    case kObjectTypeTensorType:
      return IsSubtypeArray(x, model);
    case kObjectTypeList:
      return IsSubtypeList(x, model);
    default:
      if (IsSubType(model, std::make_shared<Number>())) {
        return IsSubtypeScalar(x, model);
      }
      MS_LOG(EXCEPTION) << "Invalid model type: " << model->ToString() << ".";
  }
}

template <typename T>
bool HasAbstractType(const AbstractBasePtr &abs) {
  if (abs->isa<AbstractSequence>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    return std::any_of(abs_seq->elements().cbegin(), abs_seq->elements().cend(), HasAbstractType<T>);
  }
  return abs->IsSameTypeId(T::kTypeId);
}

}  // namespace abstract
}  // namespace mindspore
