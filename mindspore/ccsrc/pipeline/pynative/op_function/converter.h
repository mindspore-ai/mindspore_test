/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_CONVERTER_H
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_CONVERTER_H
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <unordered_map>
#include "pipeline/pynative/base.h"
#include "pipeline/pynative/pynative_execute.h"
#include "include/common/pybind_api/api_register.h"
#include "include/common/utils/primfunc_utils.h"
#include "ops/op_def.h"

namespace mindspore {
namespace pynative {
static std::unordered_map<std::string, ops::OP_DTYPE> type_str_map = {
  {"int", ops::OP_DTYPE::DT_INT},
  {"float", ops::OP_DTYPE::DT_FLOAT},
  {"bool", ops::OP_DTYPE::DT_BOOL},
  {"number", ops::OP_DTYPE::DT_NUMBER},
  {"tuple[int]", ops::OP_DTYPE::DT_TUPLE_INT},
  {"tuple[float]", ops::OP_DTYPE::DT_TUPLE_FLOAT},
  {"tuple[bool]", ops::OP_DTYPE::DT_TUPLE_BOOL},
  {"tuple[tensor]", ops::OP_DTYPE::DT_TUPLE_TENSOR},
  {"tuple[number]", ops::OP_DTYPE::DT_TUPLE_NUMBER},
  {"tuple[str]", ops::OP_DTYPE::DT_STR},
  {"list[int]", ops::OP_DTYPE::DT_LIST_INT},
  {"list[float]", ops::OP_DTYPE::DT_LIST_FLOAT},
  {"list[bool]", ops::OP_DTYPE::DT_LIST_BOOL},
  {"list[tensor]", ops::OP_DTYPE::DT_LIST_TENSOR},
  {"list[number]", ops::OP_DTYPE::DT_LIST_NUMBER},
  {"list[str]", ops::OP_DTYPE::DT_LIST_STR},
  {"tensor", ops::OP_DTYPE::DT_TENSOR},
  {"str", ops::OP_DTYPE::DT_STR},
  {"type", ops::OP_DTYPE::DT_TYPE},
};
// information of single parameter
struct FunctionParameter {
  explicit FunctionParameter(const std::string &fmt);
  bool check(const py::object &obj);
  void set_default_str(const std::string &str);
  py::object get_default_value();

  ops::OP_DTYPE type_;  // type of parameter
  std::vector<ops::OP_DTYPE> cast_types_;
  bool optional_;  // if has default value
  bool allow_none_;
  int size_;          // size of vec type
  std::string name_;  // parameter name
  std::optional<std::vector<int64_t>> default_intlist;
  std::string default_string;
  bool default_bool;
  int64_t default_int;
  double default_double;
  double default_complex[2];
};

// single overload
struct FunctionSignature {
  explicit FunctionSignature(const std::string &fmt, int index);
  // bind with real args
  bool parse(const py::list &args, const py::dict &kwargs, py::list *python_args);

  std::string name_;
  std::vector<FunctionParameter> params_;
  size_t min_args_;
  size_t max_args_;
  int index_;
};

// parser util
struct PythonArgParser {
  explicit PythonArgParser(std::vector<std::string> fmts);
  const FunctionSignature &parse(const py::list &args, const py::dict &kwargs, py::list *arg_list);

 private:
  std::vector<FunctionSignature> signatures_;  // all overloads
  std::string function_name_;
  size_t max_args_;  // max num of args
};

class Converter {
 public:
  explicit Converter(ops::OpDef *op_def);
  void Parse(const py::list &python_args);
  ValuePtr ToTensor(const py::list &python_args, size_t i);
  std::optional<ValuePtr> ToTensorOptional(const py::list &python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToTensorList(const py::list &python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToTensorListOptional(const py::list &python_args, size_t i);
  Int64ImmPtr ToInt(const py::list &python_args, size_t i);
  std::optional<Int64ImmPtr> ToIntOptional(const py::list &python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToIntList(const py::list &python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToIntListOptional(const py::list &python_args, size_t i);
  BoolImmPtr ToBool(const py::list &python_args, size_t i);
  std::optional<BoolImmPtr> ToBoolOptional(const py::list &python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToBoolList(const py::list &python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToBoolListOptional(const py::list &python_args, size_t i);
  FP32ImmPtr ToFloat(const py::list &python_args, size_t i);
  std::optional<FP32ImmPtr> ToFloatOptional(const py::list &python_args, size_t i);
  template <typename T>
  ValueTuplePtr ToFloatList(const py::list &python_args, size_t i);
  template <typename T>
  std::optional<ValueTuplePtr> ToFloatListOptional(const py::list &python_args, size_t i);
  ScalarPtr ToScalar(const py::list &python_args, size_t i);
  std::optional<ScalarPtr> ToScalarOptional(const py::list &python_args, size_t i);
  StringImmPtr ToString(const py::list &python_args, size_t i);
  std::optional<StringImmPtr> ToStringOptional(const py::list &python_args, size_t i);
  Int64ImmPtr ToDtype(const py::list &python_args, size_t i);
  std::optional<Int64ImmPtr> ToDtypeOptional(const py::list &python_args, size_t i);
  ValuePtr ConvertByCastDtype(const py::object &input, const ops::OpInputArg &op_arg, size_t i);
  ValueTuplePtr ConvertValueTupleByCastDtype(const py::list &python_args, const ops::OpInputArg &op_arg, size_t index);
  const std::vector<ops::OP_DTYPE> &source_type() const { return source_type_; }

 private:
  ops::OpDefPtr op_def_;
  // If op not type cast, source_type is default type: DT_BEGIN, if op type cast, source_type is origin type.
  std::vector<ops::OP_DTYPE> source_type_;
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_CONVERTER_H
