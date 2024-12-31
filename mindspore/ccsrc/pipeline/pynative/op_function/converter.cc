/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#include "pipeline/pynative/op_function/converter.h"
#include <unordered_map>
#include "include/common/utils/convert_utils_py.h"
#include "frontend/operator/composite/functional_overload.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pipeline/pynative/pynative_utils.h"
#include "pipeline/pynative/op_function/auto_generate/functional_map.h"

namespace mindspore {
namespace pynative {

namespace {
using OP_DTYPE = mindspore::ops::OP_DTYPE;
template <typename T, typename U>
std::shared_ptr<U> PyCast(const py::object &obj) {
  return std::make_shared<U>(py::cast<T>(obj));
}

BoolImmPtr ConvertBool(const py::object &obj) {
  if (!py::isinstance<py::bool_>(obj)) {
    // The mutable _Bool class inherits from int, because base class 'bool' is a marked final.
    if (py::isinstance<py::int_>(obj) && py::hasattr(obj, "__ms_mutable_bool__")) {
      auto obj_int64 = py::cast<int64_t>(obj);
      bool obj_bool = obj_int64 != 0;
      return std::make_shared<BoolImm>(obj_bool);
    }
    return nullptr;
  }
  return PyCast<bool, BoolImm>(obj);
}

Int64ImmPtr ConvertInt(const py::object &obj) {
  // bool is also an instance of py::int_
  if (py::isinstance<py::bool_>(obj) || !py::isinstance<py::int_>(obj)) {
    return nullptr;
  }
  return PyCast<int64_t, Int64Imm>(obj);
}

FP32ImmPtr ConvertFloat(const py::object &obj) {
  if (!py::isinstance<py::float_>(obj)) {
    return nullptr;
  }
  return PyCast<double, FP32Imm>(obj);
}

ScalarPtr ConvertNumber(const py::object &obj) {
  if (py::isinstance<py::float_>(obj)) {
    return std::make_shared<FP32Imm>(py::cast<double>(obj));
  }
  if (py::isinstance<py::bool_>(obj)) {
    return std::make_shared<BoolImm>(py::cast<bool>(obj));
  }
  if (py::isinstance<py::int_>(obj)) {
    return std::make_shared<Int64Imm>(py::cast<int64_t>(obj));
  }
  return nullptr;
}

StringImmPtr ConvertStr(const py::object &obj) {
  if (!py::isinstance<py::str>(obj)) {
    return nullptr;
  }
  return PyCast<string, StringImm>(obj);
}

template <typename T, typename U, typename N>
ValueTuplePtr ConvertList(const py::object &obj) {
  if (!py::isinstance<T>(obj)) {
    return nullptr;
  }
  auto seq = py::cast<T>(obj);
  size_t size = seq.size();
  std::vector<ValuePtr> convert(size);
  for (size_t i = 0; i < size; ++i) {
    if (!py::isinstance<U>(seq[i])) {
      return nullptr;
    }
    auto out = PyCast<U, N>(seq[i]);
    if (out == nullptr) {
      return nullptr;
    }
    convert[i] = out;
  }
  return std::make_shared<ValueTuple>(std::move(convert));
}

template <typename T>
ValueTuplePtr ConvertIntList(const py::object &obj) {
  if (!py::isinstance<T>(obj)) {
    return nullptr;
  }
  auto seq = py::cast<T>(obj);
  size_t size = seq.size();
  std::vector<ValuePtr> convert(size);
  for (size_t i = 0; i < size; ++i) {
    // bool is also an instance of py::int_
    if (py::isinstance<py::bool_>(seq[i]) || !py::isinstance<py::int_>(seq[i])) {
      return nullptr;
    }
    auto out = PyCast<py::int_, Int64Imm>(seq[i]);
    if (out == nullptr) {
      return nullptr;
    }
    convert[i] = out;
  }
  return std::make_shared<ValueTuple>(std::move(convert));
}

template <>
ValueTuplePtr ConvertList<py::tuple, py::int_, Int64Imm>(const py::object &obj) {
  return ConvertIntList<py::tuple>(obj);
}

template <>
ValueTuplePtr ConvertList<py::list, py::int_, Int64Imm>(const py::object &obj) {
  return ConvertIntList<py::list>(obj);
}

void EnablePipelineForTupleTensor(const ValueTuplePtr &tuple) {
  const auto &values = tuple->value();
  for (auto &value : values) {
    if (value->isa<BaseTensor>()) {
      auto t = value->cast<BaseTensorPtr>();
      t->set_need_pipeline_sync(true);
    }
  }
}
}  // namespace

Converter::Converter(ops::OpDef *op_def)
    : op_def_(op_def), source_type_(std::vector<ops::OP_DTYPE>(op_def->args_.size())) {}

void Converter::Parse(const py::list &python_args) {
  if (op_def_->args_.size() != python_args.size()) {
    MS_LOG(EXCEPTION) << "For operator " << op_def_->name_ << ", it requires " << op_def_->args_.size()
                      << "parameters, bug got " << python_args.size() << "parameters!";
  }
}

ValuePtr Converter::ToTensor(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = (python_args)[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto tensor = parse::ConvertTensor(obj);
  if (tensor != nullptr) {
    if (tensor->isa<tensor::BaseTensor>()) {
      tensor->cast<tensor::BaseTensorPtr>()->set_need_pipeline_sync(true);
    }
    return tensor;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert = ConvertByCastDtype(obj, op_arg, i);
    if (convert != nullptr && convert->isa<tensor::BaseTensor>()) {
      return convert->cast<tensor::BaseTensorPtr>();
    }
  }

  PyNativeAlgo::PyParser::PrintTypeCastError(op_def_, python_args, i);
  return nullptr;
}

std::optional<ValuePtr> Converter::ToTensorOptional(const py::list &python_args, size_t i) {
  const py::object &obj = (python_args)[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToTensor(python_args, i));
}

template <typename T>
ValueTuplePtr Converter::ToTensorList(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto val_seq = parse::ConvertSequence<py::tuple, ValueTuple, parse::ConvertTensor>(obj);
  if (val_seq != nullptr && val_seq->isa<ValueTuple>()) {
    EnablePipelineForTupleTensor(val_seq->cast<ValueTuplePtr>());
    return val_seq->cast<ValueTuplePtr>();
  }
  return ConvertValueTupleByCastDtype(python_args, op_arg, i);
}

template <typename T>
std::optional<ValueTuplePtr> Converter::ToTensorListOptional(const py::list &python_args, size_t i) {
  const py::object &obj = (python_args)[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToTensorList<T>(python_args, i));
}

Int64ImmPtr Converter::ToInt(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertInt(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<Int64Imm>()) {
      return convert_value->cast<Int64ImmPtr>();
    }
  }
  PyNativeAlgo::PyParser::PrintTypeCastError(op_def_, python_args, i);
  return nullptr;
}

std::optional<Int64ImmPtr> Converter::ToIntOptional(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToInt(python_args, i));
}

template <typename T>
ValueTuplePtr Converter::ToIntList(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = python_args[i];
  ValueTuplePtr convert = ConvertList<T, py::int_, Int64Imm>(obj);
  if (convert != nullptr) {
    return convert;
  }
  return ConvertValueTupleByCastDtype(python_args, op_arg, i);
}

template <typename T>
std::optional<ValueTuplePtr> Converter::ToIntListOptional(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToIntList<T>(python_args, i));
}

BoolImmPtr Converter::ToBool(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertBool(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<BoolImm>()) {
      return convert_value->cast<BoolImmPtr>();
    }
  }
  PyNativeAlgo::PyParser::PrintTypeCastError(op_def_, python_args, i);
  return nullptr;
}

std::optional<BoolImmPtr> Converter::ToBoolOptional(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToBool(python_args, i));
}

template <typename T>
ValueTuplePtr Converter::ToBoolList(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  ValueTuplePtr convert = ConvertList<T, py::bool_, BoolImm>(obj);
  if (convert != nullptr) {
    return convert;
  }
  return ConvertValueTupleByCastDtype(python_args, op_arg, i);
}

template <typename T>
std::optional<ValueTuplePtr> Converter::ToBoolListOptional(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToBoolList<T>(python_args, i));
}

FP32ImmPtr Converter::ToFloat(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertFloat(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<FP32Imm>()) {
      return convert_value->cast<FP32ImmPtr>();
    }
  }
  PyNativeAlgo::PyParser::PrintTypeCastError(op_def_, python_args, i);
  return nullptr;
}

std::optional<FP32ImmPtr> Converter::ToFloatOptional(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToFloat(python_args, i));
}

template <typename T>
ValueTuplePtr Converter::ToFloatList(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  ValueTuplePtr convert = ConvertList<T, py::float_, FP32Imm>(obj);
  if (convert != nullptr) {
    return convert;
  }
  return ConvertValueTupleByCastDtype(python_args, op_arg, i);
}

template <typename T>
std::optional<ValueTuplePtr> Converter::ToFloatListOptional(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToFloatList<T>(python_args, i));
}

ScalarPtr Converter::ToScalar(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertNumber(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<Scalar>()) {
      return convert_value->cast<ScalarPtr>();
    }
  }
  PyNativeAlgo::PyParser::PrintTypeCastError(op_def_, python_args, i);
  return nullptr;
}

std::optional<ScalarPtr> Converter::ToScalarOptional(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToScalar(python_args, i));
}

StringImmPtr Converter::ToString(const py::list &python_args, size_t i) {
  const auto &op_arg = op_def_->args_[i];
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertStr(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(obj, op_arg, i);
    if (convert_value != nullptr && convert_value->isa<StringImm>()) {
      return convert_value->cast<StringImmPtr>();
    }
  }
  PyNativeAlgo::PyParser::PrintTypeCastError(op_def_, python_args, i);
  return nullptr;
}

std::optional<StringImmPtr> Converter::ToStringOptional(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToString(python_args, i));
}

Int64ImmPtr Converter::ToDtype(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  source_type_[i] = OP_DTYPE::DT_BEGIN;
  auto convert = ConvertInt(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    TypePtr type = py::cast<mindspore::TypePtr>(obj);
    return std::make_shared<Int64Imm>(static_cast<int>(type->type_id()));
  }
  PyNativeAlgo::PyParser::PrintTypeCastError(op_def_, python_args, i);
  return nullptr;
}

std::optional<Int64ImmPtr> Converter::ToDtypeOptional(const py::list &python_args, size_t i) {
  const py::object &obj = python_args[i];
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  return std::make_optional(ToDtype(python_args, i));
}

ValuePtr Converter::ConvertByCastDtype(const py::object &input, const ops::OpInputArg &op_arg, size_t index) {
  for (auto &cast_dtype : op_arg.cast_dtype_) {
    auto convert_func = parse::GetConverterByType(parse::CombineTypesForTypeCast(cast_dtype, op_arg.arg_dtype_));
    if (convert_func == nullptr) {
      MS_LOG(EXCEPTION) << "Can't find convert function for src_dtype[" << cast_dtype << "] and dst_type"
                        << op_arg.arg_dtype_ << "].";
    }
    auto value = convert_func(input);
    if (value != nullptr) {
      source_type_[index] = cast_dtype;
      return value;
    }
  }
  return nullptr;
}

ValueTuplePtr Converter::ConvertValueTupleByCastDtype(const py::list &python_args, const ops::OpInputArg &op_arg,
                                                      size_t index) {
  const auto &input = python_args[index];
  if (!op_arg.cast_dtype_.empty()) {
    auto convert_value = ConvertByCastDtype(input, op_arg, index);
    if (convert_value != nullptr && convert_value->isa<ValueTuple>()) {
      EnablePipelineForTupleTensor(convert_value->cast<ValueTuplePtr>());
      return convert_value->cast<ValueTuplePtr>();
    }
  }
  PyNativeAlgo::PyParser::PrintTypeCastError(op_def_, python_args, index);
  return nullptr;
}

PythonArgParser::PythonArgParser(std::vector<std::string> fmts, const std::string &function_name)
    : function_name_(function_name), max_args_(0) {
  int index = 0;
  for (auto &stmt : fmts) {
    signatures_.emplace_back(stmt, index);
    index++;
  }
  for (auto &signature : signatures_) {
    if (signature.max_args_ > max_args_) {
      max_args_ = signature.max_args_;
    }
  }
}

std::string PythonArgParser::ParseError(const py::list &args, const py::dict &kwargs, const bool &is_method) {
  std::vector<std::string> type_list;
  for (const auto &py_arg : args) {
    (void)type_list.emplace_back(
      PyNativeAlgo::PyParser::BuilidPyInputTypeString(py::reinterpret_borrow<py::object>(py_arg)));
  }
  for (const auto &py_kwarg : kwargs) {
    std::string kwarg_info = py::str(py_kwarg.first);
    kwarg_info += "=";
    kwarg_info += PyNativeAlgo::PyParser::BuilidPyInputTypeString(py::reinterpret_borrow<py::object>(py_kwarg.second));
    (void)type_list.emplace_back(kwarg_info);
  }
  return prim::BuildFunctionalErrorMsg(function_name_, type_list, is_method);
}

template <typename T, typename U>
bool CheckListType(const py::object &obj) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = py::cast<T>(obj);
  size_t size = seq.size();
  for (size_t i = 0; i < size; ++i) {
    if (!py::isinstance<U>(seq[i])) {
      return false;
    }
  }
  return true;
}

template <typename T, typename U>
bool CheckItemType(const py::object &obj, const size_t &index) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = py::cast<T>(obj);
  if (!py::isinstance<U>(seq[index])) {
    return false;
  }
  return true;
}

bool CheckArgsAsIntlist(const py::object &obj, bool as_intlist) {
  return as_intlist && (py::isinstance<py::int_>(obj) || CheckItemType<py::list, py::int_>(obj, kIndex0) ||
                        CheckItemType<py::tuple, py::int_>(obj, kIndex0));
}

bool FunctionSignature::CheckParamValid(const py::object &obj, const FunctionParameter &param) {
  if (param.is_any_) {
    // only when py_method dispatch to skip type check
    return true;
  }
  if (py::isinstance<py::none>(obj)) {
    if (!param.allow_none_) {
      return false;
    }
    return true;
  } else if (param.Check(obj)) {
    return true;
  }
  return false;
}

bool FunctionSignature::Parse(const py::list &args, const py::dict &kwargs, py::list *python_args) {
  size_t nargs = args ? args.size() : 0;
  size_t nkwargs = kwargs ? kwargs.size() : 0;
  size_t arg_pos = 0;

  if (nargs > max_args_ && !allow_int_as_list_) {
    return false;
  }
  for (auto &param : params_) {
    bool is_kwd = false;
    param.is_any_ = param.type_ == OP_DTYPE::DT_ANY;
    py::object obj;
    if (arg_pos < nargs) {
      obj = (args)[arg_pos++];
      if (param.kw_only_) {
        return false;
      }
    } else if (kwargs) {
      is_kwd = true;
      py::str key_object(param.name_);
      if (PyDict_Contains(kwargs.ptr(), key_object.ptr())) {
        obj = py::reinterpret_borrow<py::object>(PyDict_GetItem(kwargs.ptr(), key_object.ptr()));
        nkwargs--;
      }
    }
    bool check_arg_as_intlist = !is_kwd && (arg_pos == kIndex1) && param.allow_vararg_;
    if (!obj) {
      if (!param.optional_) {
        return false;
      }
      python_args->append(param.GetDefaultValue());
    } else if (CheckParamValid(obj, param)) {
      python_args->append(obj);
    } else if (CheckArgsAsIntlist(args, check_arg_as_intlist)) {
      // tensor.reshape(1, 2, 3) as tensor.reshape((1, 2, 3))
      python_args->append(args);
      arg_pos = nargs;
    } else {
      return false;
    }
  }
  return nkwargs == 0;
}

FunctionSignature::FunctionSignature(const std::string &fmt, int index)
    : max_args_(0), allow_int_as_list_(false), index_(index) {
  auto open_paren = fmt.find('(');
  if (open_paren == std::string::npos) {
    MS_LOG(EXCEPTION) << "parse failed";
  }
  name_ = fmt.substr(0, open_paren);

  auto last_offset = open_paren + 1;
  bool done = false;
  bool is_kwonlyargs = false;
  while (!done) {
    auto offset = fmt.find(", ", last_offset);
    auto next_offset = offset + 2;
    if (offset == std::string::npos) {
      offset = fmt.find(')', last_offset);
      done = true;
      next_offset = offset + 1;
      if (offset == last_offset) {
        last_offset = next_offset;
        break;
      }
    }

    if (offset == std::string::npos || offset == last_offset) {
      MS_LOG(EXCEPTION) << "parse failed";
    }

    auto param_str = fmt.substr(last_offset, offset - last_offset);
    if (param_str.compare("*") != 0) {
      if (!is_kwonlyargs) {
        max_args_++;
      }
      params_.emplace_back(param_str, is_kwonlyargs);
      allow_int_as_list_ |= params_.back().allow_vararg_;
    } else {
      is_kwonlyargs = true;
    }
    last_offset = next_offset;
  }
}

ops::OP_DTYPE GetOpDtype(const std::string &type_str) {
  auto it = type_str_map.find(type_str);
  if (it == type_str_map.end()) {
    it = type_not_in_yaml_str_map.find(type_str);
    if (it == type_not_in_yaml_str_map.end()) {
      MS_LOG(EXCEPTION) << "Parse function parameter failed! invalid type string:" << type_str;
    }
  }
  return it->second;
}

FunctionParameter::FunctionParameter(const std::string &fmt, bool is_kw_only) {
  kw_only_ = is_kw_only;
  auto space = fmt.find(' ');
  if (space == std::string::npos) {
    MS_LOG(EXCEPTION) << "Parse function parameter failed! missing type:" << fmt;
  }
  auto types_str = fmt.substr(0, space);
  cast_types_ = std::vector<ops::OP_DTYPE>{};
  std::istringstream iss(types_str);
  std::string substring;
  bool first_str = true;
  while (std::getline(iss, substring, '|')) {
    if (first_str) {
      type_ = GetOpDtype(substring);
      first_str = false;
    } else {
      cast_types_.emplace_back(GetOpDtype(substring));
    }
  }

  auto name_str = fmt.substr(space + 1);
  auto eq = name_str.find('=');
  if (eq != std::string::npos) {
    name_ = name_str.substr(0, eq);
    optional_ = true;
    auto type_str = name_str.substr(eq + 1);
    if (type_str == "None") {
      allow_none_ = true;
      default_obj_ = py::none();
    } else {
      SetDefaultObj(type_str);
    }
  } else {
    optional_ = false;
    name_ = name_str;
  }
  auto varargs = name_str.find('*');
  if (varargs != std::string::npos) {
    allow_vararg_ = true;
    name_ = name_.substr(1);
  }
}

bool IsTensor(const py::object &obj, bool is_optional) {
  if (py::isinstance<py::none>(obj) && is_optional) {
    return true;
  }
  if (py::isinstance<mindspore::tensor::Tensor>(obj)) {
    return true;
  }
  if (IsStubTensor(obj)) {
    return true;
  }
  return false;
}

template <typename T>
bool IsTensorList(const py::object &obj) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = obj.cast<T>();
  for (size_t it = 0; it < seq.size(); ++it) {
    if (!IsTensor(seq[it], false)) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool CheckBoolList(const py::object &obj) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = py::cast<T>(obj);
  size_t size = seq.size();
  for (size_t i = 0; i < size; ++i) {
    if (!(py::isinstance<py::bool_>(seq[i]) ||
          (py::isinstance<py::int_>(seq[i]) && py::hasattr(seq[i], "__ms_mutable_bool__")))) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool IsScalarList(const py::object &obj) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = py::cast<T>(obj);
  size_t size = seq.size();
  for (size_t i = 0; i < size; ++i) {
    if (!(py::isinstance<py::float_>(seq[i]) || py::isinstance<py::bool_>(seq[i]) ||
          py::isinstance<py::int_>(seq[i]))) {
      return false;
    }
  }
  return true;
}

static inline std::vector<int64_t> ParseListInt(const std::string &s) {
  if (s.empty()) return std::vector<int64_t>();
  if (s[0] != '[' && s[0] != '(') {
    return std::vector<int64_t>{std::stol(s)};
  }
  auto args = std::vector<int64_t>();
  std::istringstream ss(s.substr(1, s.length() - kIndex2));
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    args.emplace_back(std::stol(tok));
  }
  return args;
}

bool CheckListType(const py::object &obj, const ops::OP_DTYPE &type) {
  switch (type) {
    case OP_DTYPE::DT_ANY:
      return true;
    case OP_DTYPE::DT_LIST_TENSOR:
      return IsTensorList<py::list>(obj);
    case OP_DTYPE::DT_LIST_ANY:
      return py::isinstance<py::list>(obj);
    case OP_DTYPE::DT_LIST_INT:
      return CheckListType<py::list, py::int_>(obj);
    case OP_DTYPE::DT_LIST_FLOAT:
      return CheckListType<py::list, py::float_>(obj);
    case OP_DTYPE::DT_LIST_BOOL:
      return CheckBoolList<py::list>(obj);
    case OP_DTYPE::DT_LIST_STR:
      return CheckListType<py::list, py::str>(obj);
    case OP_DTYPE::DT_LIST_NUMBER:
      return IsScalarList<py::list>(obj);
    case OP_DTYPE::DT_TUPLE_ANY:
      return py::isinstance<py::tuple>(obj);
    case OP_DTYPE::DT_TUPLE_INT:
      return CheckListType<py::tuple, py::int_>(obj);
    case OP_DTYPE::DT_TUPLE_FLOAT:
      return CheckListType<py::tuple, py::float_>(obj);
    case OP_DTYPE::DT_TUPLE_BOOL:
      return CheckBoolList<py::tuple>(obj);
    case OP_DTYPE::DT_TUPLE_TENSOR:
      return IsTensorList<py::tuple>(obj);
    case OP_DTYPE::DT_TUPLE_NUMBER:
      return IsScalarList<py::tuple>(obj);
    default:
      MS_LOG(EXCEPTION) << "Unknown param type:" << type;
  }
  return false;
}

bool TypeCheck(const py::object &obj, const ops::OP_DTYPE &type, bool optional) {
  switch (type) {
    case OP_DTYPE::DT_TENSOR:
      return IsTensor(obj, optional);
    case OP_DTYPE::DT_NUMBER:
      return py::isinstance<py::float_>(obj) || py::isinstance<py::bool_>(obj) || py::isinstance<py::int_>(obj);
    case OP_DTYPE::DT_FLOAT:
      return py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj);
    case OP_DTYPE::DT_INT:
      return py::isinstance<py::int_>(obj);
    case OP_DTYPE::DT_BOOL:
      return py::isinstance<py::bool_>(obj) ||
             (py::isinstance<py::int_>(obj) && py::hasattr(obj, "__ms_mutable_bool__"));
    case OP_DTYPE::DT_TYPE:
      return py::isinstance<py::int_>(obj) || py::isinstance<mindspore::Type>(obj);
    case OP_DTYPE::DT_STR:
      return py::isinstance<py::str>(obj);
    default:
      return CheckListType(obj, type);
  }
  return false;
}

bool FunctionParameter::Check(const py::object &obj) const {
  if (!TypeCheck(obj, type_, optional_)) {
    bool res = std::accumulate(cast_types_.begin(), cast_types_.end(), false,
                               [&](bool acc, const auto &type) { return acc || TypeCheck(obj, type, optional_); });
    return res;
  }
  return true;
}

template <typename T>
py::object GetPyListInt(const std::vector<int64_t> &vec) {
  T list_py(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    list_py[i] = py::int_(vec[i]);
  }
  return list_py;
}

std::optional<py::int_> ParseIntStr(const std::string &str) {
  char *str_end;
  auto defalut_int = strtol(str.c_str(), &str_end, 0);
  return (*str_end == 0) ? std::optional<py::int_>(py::int_(defalut_int)) : std::nullopt;
}

std::optional<py::bool_> ParseBoolStr(const std::string &str) {
  if (str == "True" || str == "true" || str == "False" || str == "false") {
    return std::optional<py::bool_>(py::bool_((str == "True" || str == "true")));
  }
  return std::nullopt;
}

py::object ParseNumber(const std::string &str) {
  auto cast_bool = ParseBoolStr(str);
  if (cast_bool.has_value()) {
    return cast_bool.value();
  }
  auto cast_int = ParseIntStr(str);
  if (cast_int.has_value()) {
    return cast_int.value();
  }
  return py::float_(stof(str));
}

std::string RemoveQuotes(const std::string &str) {
  if (str.size() >= kIndex2 && str.front() == '\'' && str.back() == '\'') {
    return str.substr(1, str.size() - kIndex2);
  }
  return str;
}

void FunctionParameter::SetDefaultObj(const std::string &str) {
  switch (type_) {
    case ops::OP_DTYPE::DT_INT:
      default_obj_ = py::int_(stol(str));
      break;
    case ops::OP_DTYPE::DT_FLOAT:
      default_obj_ = py::float_(stof(str));
      break;
    case ops::OP_DTYPE::DT_BOOL:
      default_obj_ = py::bool_((str == "True" || str == "true"));
      break;
    case ops::OP_DTYPE::DT_NUMBER:
      default_obj_ = ParseNumber(str);
      break;
    case ops::OP_DTYPE::DT_TUPLE_INT:
      default_obj_ = GetPyListInt<py::tuple>(ParseListInt(str));
      break;
    case ops::OP_DTYPE::DT_TUPLE_TENSOR:
      // now only support default=None
      if (str != "None") {
        MS_LOG(EXCEPTION) << "default value for Tensor must be none, got: " << str;
      }
      default_obj_ = py::none();
      break;
    case ops::OP_DTYPE::DT_STR:
      default_obj_ = py::str(RemoveQuotes(str));
      break;
    case ops::OP_DTYPE::DT_TENSOR:
      if (str != "None") {
        MS_LOG(EXCEPTION) << "default value for Tensor must be None, but got: " << str;
      }
      default_obj_ = py::none();
      break;
    case ops::OP_DTYPE::DT_LIST_INT:
      default_obj_ = GetPyListInt<py::list>(ParseListInt(str));
      break;
    case ops::OP_DTYPE::DT_LIST_FLOAT:
      if (str != "None") {
        MS_LOG(EXCEPTION) << "Defaults not supported for float[]";
      }
      default_obj_ = py::none();
      break;
    default:
      MS_LOG(EXCEPTION) << "The" << type_ << " is an unknown type "
                        << ", or the default value cannot be set.";
      break;
  }
}

// Declare template to compile corresponding method.
template ValueTuplePtr Converter::ToTensorList<py::tuple>(const py::list &python_args, size_t i);
template ValueTuplePtr Converter::ToTensorList<py::list>(const py::list &python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToTensorListOptional<py::tuple>(const py::list &python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToTensorListOptional<py::list>(const py::list &python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToIntListOptional<py::tuple>(const py::list &python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToIntListOptional<py::list>(const py::list &python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToBoolListOptional<py::tuple>(const py::list &python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToBoolListOptional<py::list>(const py::list &python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToFloatListOptional<py::tuple>(const py::list &python_args, size_t i);
template std::optional<ValueTuplePtr> Converter::ToFloatListOptional<py::list>(const py::list &python_args, size_t i);

}  // namespace pynative
}  // namespace mindspore
