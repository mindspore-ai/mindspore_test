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
#include "pynative/op_function/converter.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include "include/common/utils/convert_utils_py.h"
#include "frontend/operator/composite/functional_overload.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "pynative/pynative_utils.h"
#include "include/common/utils/tensor_py.h"
#include "frontend/operator/composite/auto_generate/functional_map.h"

namespace mindspore {
namespace pynative {
#define RAISE_PARSE_ERROR(out_error_msg, raise_error, msg, func_name) \
  if (out_error_msg || raise_error) {                                 \
    std::string error_msg = msg;                                      \
    if (raise_error) {                                                \
      MS_EXCEPTION(TypeError) << func_name << "()" << error_msg;      \
    } else if (out_error_msg) {                                       \
      out_error_msg->append(error_msg);                               \
    }                                                                 \
  }
using OpDefConvertFunc = std::function<ValuePtr(const py::object &obj)>;

namespace {
using OP_DTYPE = mindspore::ops::OP_DTYPE;
template <typename T, typename U>
std::shared_ptr<U> PyCast(const py::object &obj) {
  return std::make_shared<U>(py::cast<T>(obj));
}

template <>
std::shared_ptr<FP32Imm> PyCast<double, FP32Imm>(const py::object &obj) {
  auto obj_float32 = py::cast<float>(obj);
  auto ret = std::make_shared<FP32Imm>(obj_float32);
  ret->set_prim_value(py::cast<double>(obj));
  return ret;
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
    return PyCast<double, FP32Imm>(obj);
  }
  if (py::isinstance<py::bool_>(obj)) {
    return PyCast<bool, BoolImm>(obj);
  }
  if (py::isinstance<py::int_>(obj)) {
    return PyCast<int64_t, Int64Imm>(obj);
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
    signatures_.emplace_back(std::make_shared<FunctionSignature>(stmt, index, function_name_));
    index++;
  }
  for (auto &signature : signatures_) {
    if (signature->max_args_ > max_args_) {
      max_args_ = signature->max_args_;
    }
  }
}

const std::vector<std::string> PythonArgParser::GetParseTypeListString(const py::list &args, const py::dict &kwargs) {
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
  return type_list;
}

template <typename T, typename U>
bool CheckListType(const py::object &obj, int &idx, bool fullcheck = false) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = py::cast<T>(obj);
  size_t size = seq.size();
  if (size == 0) {
    return true;
  }
  size = fullcheck ? size : 1;
  for (size_t i = 0; i < size; ++i) {
    if (!py::isinstance<U>(seq[i])) {
      idx = i;
      return false;
    }
  }
  return true;
}

bool IsTensor(const py::object &obj) {
  if (IsStubTensor(obj)) {
    return true;
  }
  if (mindspore::tensor::IsTensorPy(obj)) {
    return true;
  }
  return false;
}

template <typename T>
bool CheckListInt(const py::object &obj, int &idx, bool fullcheck = false) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = py::cast<T>(obj);
  size_t size = seq.size();
  if (size == 0) {
    return true;
  }
  size = fullcheck ? size : 1;
  for (size_t i = 0; i < size; ++i) {
    if (!py::isinstance<py::int_>(seq[i]) && !IsTensor(seq[i])) {
      idx = i;
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

std::string GetTypeErrorMsg(bool is_kwd, int error_idx, const py::object &obj, const FunctionParameter &param,
                            size_t arg_pos) {
  std::string error_msg;
  if (is_kwd) {
    error_msg = ": argument '" + param.name_ + "' must be " + ops::EnumToString(param.type_) + " but got " +
                py::str(obj.get_type().attr("__name__")).cast<std::string>() + ".";
  } else {
    error_msg = ": argument '" + param.name_ + "' (position " + std::to_string(arg_pos) + ")" + " must be " +
                ops::EnumToString(param.type_);
    error_msg += (error_idx >= 0)
                   ? " but found type of " + py::str(obj.get_type().attr("__name__")).cast<std::string>() + " at pos " +
                       std::to_string(error_idx) + "."
                   : ", not " + py::str(obj.get_type().attr("__name__")).cast<std::string>() + ".";
  }
  return error_msg;
}

bool FunctionSignature::CheckParamValid(const py::object &obj, const FunctionParameter &param, bool raise_error,
                                        std::string *out_error_msg, ConvertPair &convert_type, int &error_idx) {
  if (param.is_any_) {
    // only when py_method dispatch to skip type check
    return true;
  }
  if (py::isinstance<py::none>(obj)) {
    if (!param.allow_none_) {
      RAISE_PARSE_ERROR(out_error_msg, raise_error, ": missing 1 required positional argument: " + param.name_ + ".",
                        name_);
      return false;
    }
    return true;
  } else if (param.Check(obj, convert_type, error_idx)) {
    return true;
  }
  return false;
}

bool FunctionSignature::Parse(const py::list &args, const py::dict &kwargs, ParserArgs &parser_args, bool raise_error,
                              std::string *out_error_msg) {
  size_t nargs = args ? args.size() : 0;
  size_t nkwargs = kwargs ? kwargs.size() : 0;
  size_t arg_pos = 0;
  size_t out_arglist_index = 0;

  if (nargs > max_pos_args_ && !allow_int_as_list_) {
    RAISE_PARSE_ERROR(out_error_msg, raise_error,
                      " takes " + std::to_string(max_pos_args_) + " positional arguments but " + std::to_string(nargs) +
                        (nargs > 1 ? " were" : " was") + " given.",
                      name_);
    return false;
  }
  for (auto &param : params_) {
    bool is_kwd = false;
    param.is_any_ = param.type_ == OP_DTYPE::DT_ANY;
    py::object obj;
    if (arg_pos < nargs) {
      obj = args[arg_pos++];
      if (param.kw_only_) {
        RAISE_PARSE_ERROR(out_error_msg, raise_error, " got extra positional args.", name_);
        return false;
      }
    } else if (kwargs) {
      is_kwd = true;
      py::str key_object(param.name_);
      if (kwargs.contains(key_object)) {
        obj = kwargs[key_object];
        nkwargs--;
      }
    }
    bool check_arg_as_intlist = !is_kwd && (arg_pos == kIndex1) && param.allow_vararg_;
    int error_idx = check_arg_as_intlist ? kIndex0 : -1;
    ConvertPair convert_type({OP_DTYPE::DT_BEGIN, param.type_});
    if (!obj) {
      if (!param.optional_) {
        RAISE_PARSE_ERROR(out_error_msg, raise_error, " missing 1 required positional argument: " + param.name_ + ".",
                          name_);
        return false;
      }
      parser_args.SetArg(param.GetDefaultValue(), convert_type, out_arglist_index++);
    } else if (CheckParamValid(obj, param, raise_error, out_error_msg, convert_type, error_idx)) {
      parser_args.SetArg(obj, convert_type, out_arglist_index++);
    } else if (CheckArgsAsIntlist(args, check_arg_as_intlist)) {
      // tensor.reshape(1, 2, 3) as tensor.reshape((1, 2, 3))
      parser_args.SetArg(args, {OP_DTYPE::DT_LIST_INT, param.type_}, out_arglist_index++);
      arg_pos = nargs;
    } else {
      RAISE_PARSE_ERROR(out_error_msg, raise_error, GetTypeErrorMsg(is_kwd, error_idx, obj, param, arg_pos), name_);
      return false;
    }
  }
  return RaiseParseKeywordArgsError(nkwargs, raise_error, out_error_msg, nargs, kwargs);
}

bool FunctionSignature::RaiseParseKeywordArgsError(size_t nkwargs, bool raise_error, std::string *out_error_msg,
                                                   size_t nargs, const py::dict &kwargs) {
  std::string error_msg;
  if (nkwargs == 0) {
    return true;
  }
  if (raise_error || out_error_msg) {
    for (const auto &kw_obj : kwargs) {
      int64_t pos = -1;
      for (size_t i = 0; i < params_.size(); ++i) {
        if (py::str(kw_obj.first).cast<std::string>() == params_[i].name_) {
          pos = i;
        }
      }
      if (pos < 0) {
        error_msg = " got an unexpected keyword argument '" + py::str(kw_obj.first).cast<std::string>() + "'.";
      } else if (pos < (int64_t)nargs) {
        error_msg = " got multiple values for argument '" + py::str(kw_obj.first).cast<std::string>() + "'.";
      }
    }
    if (error_msg.empty()) {
      error_msg = "(): invalid keyword arguments.";
    }
    if (out_error_msg) {
      out_error_msg->append(error_msg);
    }
    if (raise_error) {
      MS_EXCEPTION(TypeError) << name_ << "()" << error_msg;
    }
  }
  return false;
}

FunctionSignature::FunctionSignature(const std::string &fmt, int index, const std::string &name)
    : name_(name), max_pos_args_(0), max_args_(0), min_args_(0), allow_int_as_list_(false), index_(index) {
  auto open_paren = fmt.find('(');
  if (open_paren == std::string::npos) {
    MS_LOG(EXCEPTION) << "parse failed";
  }

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
        max_pos_args_++;
      }
      params_.emplace_back(param_str, is_kwonlyargs);
      allow_int_as_list_ |= params_.back().allow_vararg_;
      if (!params_.back().optional_) {
        min_args_++;
      }
      max_args_++;
    } else {
      is_kwonlyargs = true;
    }
    last_offset = next_offset;
  }
}

std::string FunctionSignature::ToString() {
  std::stringstream param_ss;
  bool kw_only_flag = false;
  for (auto &param : params_) {
    if (param.kw_only_ && !kw_only_flag) {
      kw_only_flag = true;
      param_ss << "*, ";
    }
    std::vector<std::string> type_list = {ops::EnumToString(param.type_)};
    std::transform(param.cast_types_.begin(), param.cast_types_.end(), std::back_inserter(type_list),
                   [](const auto &type) { return ops::EnumToString(type); });
    if (param.allow_none_) {
      type_list.emplace_back("None");
    }
    std::sort(type_list.begin(), type_list.end());
    std::string type_list_str = std::accumulate(
      type_list.begin(), type_list.end(), std::string(),
      [](const std::string &a, const std::string &b) -> std::string { return a.empty() ? b : a + ", " + b; });
    param_ss << param.name_ << "=<" << type_list_str << ">, ";
  }
  auto type_str = param_ss.str().substr(0, param_ss.str().length() - 2);
  return "(" + type_str + ")";
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
    auto value_str = name_str.substr(eq + 1);
    if (value_str == "None") {
      allow_none_ = true;
    }
    default_str_.assign(substring).append(",").append(value_str);
    ParserDefaultObjects::GetInstance().Set(type_, value_str, default_str_);
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

template <typename T>
bool IsTensorList(const py::object &obj, int &idx, bool fullcheck = false) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = obj.cast<T>();
  size_t size = seq.size();
  if (size == 0) {
    return true;
  }
  size = fullcheck ? size : 1;
  for (size_t i = 0; i < size; ++i) {
    if (!IsTensor(seq[i])) {
      idx = i;
      return false;
    }
  }
  return true;
}

template <typename T>
bool CheckBoolList(const py::object &obj, int &idx, bool fullcheck = false) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = py::cast<T>(obj);
  size_t size = seq.size();
  if (size == 0) {
    return true;
  }
  size = fullcheck ? size : 1;
  for (size_t i = 0; i < size; ++i) {
    if (!(py::isinstance<py::bool_>(seq[i]) ||
          (py::isinstance<py::int_>(seq[i]) && py::hasattr(seq[i], "__ms_mutable_bool__")))) {
      idx = i;
      return false;
    }
  }
  return true;
}

template <typename T>
bool IsScalarList(const py::object &obj, int &idx, bool fullcheck = false) {
  if (!py::isinstance<T>(obj)) {
    return false;
  }
  auto seq = py::cast<T>(obj);
  size_t size = seq.size();
  if (size == 0) {
    return true;
  }
  size = fullcheck ? size : 1;
  for (size_t i = 0; i < size; ++i) {
    if (!(py::isinstance<py::float_>(seq[i]) || py::isinstance<py::bool_>(seq[i]) ||
          py::isinstance<py::int_>(seq[i]))) {
      idx = i;
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

bool ListTypeCheck(const py::object &obj, const ops::OP_DTYPE &type, int &idx, bool fullcheck = false) {
  switch (type) {
    case OP_DTYPE::DT_ANY:
      return true;
    case OP_DTYPE::DT_LIST_TENSOR:
      return IsTensorList<py::list>(obj, idx, fullcheck);
    case OP_DTYPE::DT_LIST_ANY:
      return py::isinstance<py::list>(obj);
    case OP_DTYPE::DT_LIST_INT:
      return CheckListInt<py::list>(obj, idx, fullcheck);
    case OP_DTYPE::DT_LIST_FLOAT:
      return CheckListType<py::list, py::float_>(obj, idx, fullcheck);
    case OP_DTYPE::DT_LIST_BOOL:
      return CheckBoolList<py::list>(obj, idx, fullcheck);
    case OP_DTYPE::DT_LIST_STR:
      return CheckListType<py::list, py::str>(obj, idx, fullcheck);
    case OP_DTYPE::DT_LIST_NUMBER:
      return IsScalarList<py::list>(obj, idx, fullcheck);
    case OP_DTYPE::DT_TUPLE_ANY:
      return py::isinstance<py::tuple>(obj);
    case OP_DTYPE::DT_TUPLE_INT:
      return CheckListInt<py::tuple>(obj, idx, fullcheck);
    case OP_DTYPE::DT_TUPLE_FLOAT:
      return CheckListType<py::tuple, py::float_>(obj, idx, fullcheck);
    case OP_DTYPE::DT_TUPLE_BOOL:
      return CheckBoolList<py::tuple>(obj, idx, fullcheck);
    case OP_DTYPE::DT_TUPLE_TENSOR:
      return IsTensorList<py::tuple>(obj, idx, fullcheck);
    case OP_DTYPE::DT_TUPLE_NUMBER:
      return IsScalarList<py::tuple>(obj, idx, fullcheck);
    default:
      MS_LOG(EXCEPTION) << "Unknown param type:" << type;
  }
  return false;
}

bool TypeCheck(const py::object &obj, const ops::OP_DTYPE &type, int &idx, ConvertPair &convert_type) {
  switch (type) {
    case OP_DTYPE::DT_TENSOR:
      if (IsStubTensor(obj)) {
        convert_type.first = OP_DTYPE::DT_TENSOR;
        return true;
      }
      return IsTensor(obj);
    case OP_DTYPE::DT_NUMBER:
      return py::isinstance<py::float_>(obj) || py::isinstance<py::bool_>(obj) || py::isinstance<py::int_>(obj);
    case OP_DTYPE::DT_FLOAT:
      return py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj);
    case OP_DTYPE::DT_INT:
      return py::isinstance<py::int_>(obj);
    case OP_DTYPE::DT_BOOL:
      if (py::isinstance<py::bool_>(obj)) {
        return true;
      }
      if (py::isinstance<py::int_>(obj) && py::hasattr(obj, "__ms_mutable_bool__")) {
        convert_type.first = OP_DTYPE::DT_BOOL;
        return true;
      }
      return false;
    case OP_DTYPE::DT_TYPE:
      return py::isinstance<mindspore::Type>(obj);
    case OP_DTYPE::DT_STR:
      return py::isinstance<py::str>(obj);
    default:
      return ListTypeCheck(obj, type, idx);
  }
  return false;
}

bool FunctionParameter::Check(const py::object &obj, ConvertPair &convert_type, int &error_idx) const {
  if (!TypeCheck(obj, type_, error_idx, convert_type)) {
    auto it = std::find_if(cast_types_.begin(), cast_types_.end(), [&](const ops::OP_DTYPE &cast_type) {
      return TypeCheck(obj, cast_type, error_idx, convert_type);
    });
    if (it != cast_types_.end() && convert_type.first == OP_DTYPE::DT_BEGIN) {
      convert_type.first = *it;
      return true;
    }
    return false;
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

py::object ParserDefaultObjects::StrToPyObj(const ops::OP_DTYPE &type, const std::string &str) {
  if (str == "None") {
    return py::none();
  }
  switch (type) {
    case ops::OP_DTYPE::DT_INT:
      return py::int_(stol(str));
    case ops::OP_DTYPE::DT_FLOAT:
      return py::float_(stof(str));
    case ops::OP_DTYPE::DT_BOOL:
      return py::bool_((str == "True" || str == "true"));
    case ops::OP_DTYPE::DT_NUMBER:
      return ParseNumber(str);
    case ops::OP_DTYPE::DT_TUPLE_INT:
      return GetPyListInt<py::tuple>(ParseListInt(str));
    case ops::OP_DTYPE::DT_TUPLE_TENSOR:
      // now only support default=None
      if (str != "None") {
        MS_LOG(EXCEPTION) << "default value for Tensor must be none, got: " << str;
      }
      return py::none();
    case ops::OP_DTYPE::DT_STR:
      return py::str(RemoveQuotes(str));
    case ops::OP_DTYPE::DT_TENSOR:
      if (str != "None") {
        MS_LOG(EXCEPTION) << "default value for Tensor must be None, but got: " << str;
      }
      return py::none();
    case ops::OP_DTYPE::DT_LIST_INT:
      return GetPyListInt<py::list>(ParseListInt(str));
    case ops::OP_DTYPE::DT_LIST_FLOAT:
      if (str != "None") {
        MS_LOG(EXCEPTION) << "Defaults not supported for float[]";
      }
      return py::none();
    default:
      MS_LOG(EXCEPTION) << "The" << type << " is an unknown type "
                        << ", or the default value cannot be set.";
      break;
  }
}

ValuePtr ConvertSimpleBool(const py::object &obj) { return std::make_shared<BoolImm>(py::cast<bool>(obj)); }

ValuePtr ConvertMutableBool(const py::object &obj) {
  auto obj_int64 = py::cast<int64_t>(obj);
  bool obj_bool = obj_int64 != 0;
  return std::make_shared<BoolImm>(obj_bool);
}

ValuePtr ConvertStubTensor(const py::object &obj) {
  auto tensor = PyStubNodeCast(obj);
  if (tensor != nullptr) {
    if (tensor->isa<tensor::BaseTensor>()) {
      tensor->cast<tensor::BaseTensorPtr>()->set_need_pipeline_sync(true);
    }
    return tensor;
  }
  return tensor;
}

ValuePtr ConvertSimpleTensor(const py::object &obj) {
  auto tensor = tensor::ConvertToTensor(obj);
  if (tensor != nullptr) {
    if (tensor->isa<tensor::BaseTensor>()) {
      tensor->cast<tensor::BaseTensorPtr>()->set_need_pipeline_sync(true);
    }
    return tensor;
  }
  return tensor;
}

ValuePtr ConvertTensorList(const py::object &obj) {
  auto val_seq = parse::ConvertSequence<py::tuple, ValueTuple, parse::ConvertTensor>(obj);
  if (val_seq != nullptr && val_seq->isa<ValueTuple>()) {
    EnablePipelineForTupleTensor(val_seq->cast<ValueTuplePtr>());
    return val_seq;
  }
  return val_seq;
}

static const std::unordered_map<int32_t, OpDefConvertFunc> kParseConverters = {
  {parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_TENSOR), ConvertSimpleTensor},
  {parse::CombineTypesForTypeCast(mindspore::ops::DT_TENSOR, mindspore::ops::DT_TENSOR), ConvertStubTensor},
  {parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_TUPLE_TENSOR), ConvertTensorList},
  {parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_LIST_TENSOR), ConvertTensorList},
  {parse::CombineTypesForTypeCast(mindspore::ops::DT_BOOL, mindspore::ops::DT_BOOL), ConvertMutableBool},
  {parse::CombineTypesForTypeCast(mindspore::ops::DT_BEGIN, mindspore::ops::DT_BOOL), ConvertSimpleBool}};

OpDefConvertFunc GetSimpleConverterByType(int32_t dtype) {
  auto it = kParseConverters.find(dtype);
  if (it == kParseConverters.end()) {
    return nullptr;
  }
  return it->second;
}

ValuePtr ParserArgs::ConvertByParseDtype(size_t index) {
  auto src = src_types_[index];
  auto dst = dst_types_[index];
  OpDefConvertFunc convert_func = GetSimpleConverterByType(parse::CombineTypesForTypeCast(src, dst));
  if (convert_func == nullptr) {
    convert_func =
      parse::GetConverterByType(src == OP_DTYPE::DT_BEGIN ? dst : parse::CombineTypesForTypeCast(src, dst));
    if (convert_func == nullptr) {
      MS_EXCEPTION(NotImplementedError) << "Can't find convert function for src_dtype[" << src << "] and dst_type"
                                        << dst << "].";
    }
  } else {
    src_types_[index] = mindspore::ops::DT_BEGIN;
  }
  auto value = convert_func(arg_list_[index]);
  if (value != nullptr) {
    return value;
  }
  return nullptr;
}

void ParserArgs::InsertInputTensor(size_t index, const py::object &input) {
  arg_list_.insert(arg_list_.begin() + index, input);
  src_types_.insert(src_types_.begin() + index, ops::OP_DTYPE::DT_BEGIN);
  dst_types_.insert(dst_types_.begin() + index, ops::OP_DTYPE::DT_TENSOR);
}

ValuePtr UnpackTensor(const py::object &input, const std::string &func_name) {
  if (IsStubTensor(input)) {
    return ConvertStubTensor(input);
  } else if (tensor::IsTensorPy(input)) {
    return ConvertSimpleTensor(input);
  } else {
    MS_EXCEPTION(TypeError) << "Tensor." << func_name << "() doesn't apply to '"
                            << PyNativeAlgo::PyParser::BuilidPyInputTypeString((input)) << "' object.";
  }
  return nullptr;
}

void ParserArgs::SetArg(const py::object &arg, const ConvertPair &convert_type, size_t index) {
  if (index > arg_list_.size()) {
    MS_LOG(EXCEPTION) << "Invalid argument index.";
  }
  arg_list_[index] = arg;
  src_types_[index] = convert_type.first;
  dst_types_[index] = convert_type.second;
}

void ParserArgs::ClearArgs() {
  arg_list_.clear();
  src_types_.clear();
  dst_types_.clear();
}

void ParserArgs::PrintConvertError(size_t index) {
  const auto &obj = arg_list_[index];
  std::stringstream ss;
  size_t param_idx = index;
  // In tensor api, 'input' is not included in the signature's parameter list
  if (arg_list_.size() > signature_->params_.size()) {
    if (param_idx == 0) {
      MS_LOG(EXCEPTION) << "Invalid param idx, please check.";
    }
    param_idx -= 1;
  }
  ss << signature_->name_ << "():";
  ss << " argument \'" << signature_->params_[param_idx].name_ << "\'(position " << param_idx << ") should be "
     << ops::EnumToString(dst_types_[index]);
  if (!py::isinstance<py::tuple>(obj) && !py::isinstance<py::list>(obj)) {
    ss << ", but got " << PyNativeAlgo::PyParser::BuilidPyInputTypeString(py::reinterpret_borrow<py::object>(obj))
       << ".";
  } else {
    int error_pos = 0;
    py::object element;
    ListTypeCheck(obj, src_types_[index], error_pos, true);
    if (py::isinstance<py::tuple>(obj)) {
      element = py::cast<py::tuple>(obj)[error_pos];
    } else {
      element = py::cast<py::list>(obj)[error_pos];
    }
    ss << ", but unpack failed at pos " << error_pos << " (got "
       << PyNativeAlgo::PyParser::BuilidPyInputTypeString(py::reinterpret_borrow<py::object>(element)) << ").";
  }
  MS_EXCEPTION(TypeError) << ss.str();
}

std::vector<std::string> GetInvalidKwargsName(const py::dict &kwargs, const std::vector<FunctionParameter> &params) {
  std::vector<std::string> invalid_names;
  const size_t kw_start_idx = params.size() - kwargs.size();
  for (const auto &arg : kwargs) {
    bool is_vailed = true;
    auto arg_name = arg.first.cast<std::string>();
    for (auto idx = kw_start_idx; idx < params.size(); ++idx) {
      if (arg_name == params[idx].name_) {
        is_vailed = false;
        break;
      }
    }
    if (is_vailed) {
      invalid_names.emplace_back(arg_name);
    }
  }
  return invalid_names;
}

std::vector<std::string> ParamMatchInfo(const py::list &args, const py::dict &kwargs,
                                        const std::vector<FunctionParameter> &params) {
  size_t argpos = 0;
  std::string type_info = "    match failed because invalid types: (";
  std::string guide_line(type_info.size(), ' ');
  while (argpos < params.size()) {
    bool is_kwd = argpos >= args.size();
    py::object obj;
    if (is_kwd) {
      py::str key_object(params[argpos].name_);
      if (PyDict_Contains(kwargs.ptr(), key_object.ptr())) {
        obj = py::reinterpret_borrow<py::object>(PyDict_GetItem(kwargs.ptr(), key_object.ptr()));
      }
    } else {
      obj = args[argpos];
    }
    if (!obj) {
      if (!params[argpos].optional_) {
        return {"    missing required argument: " + params[argpos].name_};
      }
    } else {
      auto input_type_str =
        PyNativeAlgo::PyParser::BuilidPyInputTypeString(py::reinterpret_borrow<py::object>(obj)) + ", ";
      if (is_kwd) {
        input_type_str = params[argpos].name_ + "=" + input_type_str;
      }
      type_info += input_type_str;
      ConvertPair convert_type({params[argpos].type_, params[argpos].type_});
      int error_idx = 0;
      if ((!is_kwd && params[argpos].kw_only_) || (py::isinstance<py::none>(obj) && !params[argpos].allow_none_) ||
          !params[argpos].Check(obj, convert_type, error_idx)) {
        guide_line += std::string(input_type_str.size(), '~');
      } else {
        guide_line += std::string(input_type_str.size(), ' ');
      }
    }
    ++argpos;
  }
  type_info = type_info.substr(0, type_info.size() - kIndex2);
  return {type_info + ")", guide_line};
}

std::string PythonArgParser::PrintParseError(const py::list &args, const py::dict &kwargs, const bool &is_method) {
  const auto arg_size = (args ? args.size() : 0) + (kwargs ? kwargs.size() : 0);
  std::vector<int> valid_signature_idx;
  for (const auto &signature : signatures_) {
    if (arg_size >= signature->min_args_ && arg_size <= signature->max_args_) {
      valid_signature_idx.emplace_back(signature->index_);
    }
  }
  if (valid_signature_idx.size() == 1) {
    ParserArgs parser_args(signatures_[valid_signature_idx[0]]);
    signatures_[valid_signature_idx[0]]->Parse(args, kwargs, parser_args, true);
  }
  std::vector<std::string> error_msg;
  std::unordered_set<std::string> signatures_str;
  for (auto idx : valid_signature_idx) {
    auto sig_str = is_method ? "Tensor." + function_name_ + signatures_[idx]->ToString()
                             : function_name_ + signatures_[idx]->ToString();
    if (signatures_str.find(sig_str) != signatures_str.end()) {
      continue;
    }
    signatures_str.insert(sig_str);
    error_msg.emplace_back("\"" + sig_str + "\"");
    std::vector<std::string> invalid_names;
    if (kwargs) {
      // unrecognized kwarg name
      invalid_names = GetInvalidKwargsName(kwargs, signatures_[idx]->params_);
      if (!invalid_names.empty()) {
        auto invalid_kw = std::accumulate(
          invalid_names.begin(), invalid_names.end(), std::string(),
          [](const std::string &a, const std::string &b) -> std::string { return a.empty() ? b : a + ", " + b; });
        error_msg.emplace_back("    match failed because incorrect keyword name: " + invalid_kw);
      }
    }
    if (invalid_names.empty()) {
      auto match_infos = ParamMatchInfo(args, kwargs, signatures_[idx]->params_);
      if (!match_infos.empty()) {
        error_msg.insert(error_msg.end(), match_infos.begin(), match_infos.end());
      } else {
        ParserArgs parser_args(signatures_[idx]);
        std::string error_info;
        signatures_[idx]->Parse(args, kwargs, parser_args, false, &error_info);
        error_msg.emplace_back("    " + error_info);
      }
    }
  }
  auto type_list = GetParseTypeListString(args, kwargs);
  if (error_msg.empty()) {
    return prim::BuildFunctionalErrorMsg(function_name_, type_list, is_method);
  } else {
    auto ss = prim::BuildApiInputInfo(function_name_, type_list);
    for (auto &error_info : error_msg) {
      ss << error_info << "\n";
    }
    return ss.str();
  }
}

ParserDefaultObjects &ParserDefaultObjects::GetInstance() {
  static ParserDefaultObjects default_objs_instance;
  return default_objs_instance;
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
