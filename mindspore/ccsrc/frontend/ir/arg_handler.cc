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

#include <algorithm>
#include <vector>
#include <tuple>
#include <string>
#include "frontend/ir/arg_handler.h"
#include "ops/op_def.h"
#include "mindspore/ops/op_def/op_enum.h"
#include "pipeline/jit/ps/parse/data_converter.h"
#include "utils/anf_utils.h"

namespace mindspore {

namespace pynative {

namespace {
using OP_DTYPE = mindspore::ops::OP_DTYPE;
template <typename T, typename U>
std::shared_ptr<U> PyCast(const py::object &obj) {
  return std::make_shared<U>(py::cast<T>(obj));
}
}  // namespace

Int64ImmPtr ConvertInt(const py::object &obj) {
  // bool is also an instance of py::int_
  if (py::isinstance<py::bool_>(obj) || !py::isinstance<py::int_>(obj)) {
    return nullptr;
  }
  return PyCast<int64_t, Int64Imm>(obj);
}

Int64ImmPtr ToDtype(const py::object &obj) {
  auto convert = ConvertInt(obj);
  if (convert != nullptr) {
    return convert;
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    TypePtr type = py::cast<mindspore::TypePtr>(obj);
    return std::make_shared<Int64Imm>(static_cast<int>(type->type_id()));
  }
  return nullptr;
}

py::object DtypeToTypeId(const std::string &op_name, const std::string &arg_name, const py::object &obj) {
  if (py::isinstance<py::none>(obj)) {
    return obj;
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    auto dtype = ToDtype(obj);
    return dtype ? py::cast(dtype->value()) : py::none();
  }
  if (obj.equal(py::type::of(py::bool_()))) {
    return py::cast(static_cast<int>(kNumberTypeBool));
  }
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the input '" << arg_name
                    << "' should be one of ['mindspore dtype', 'bool'], but got " << obj << ".";
  return py::none();
}

py::object StrToEnum(const std::string &op_name, const std::string &arg_name, const py::object &obj) {
  if (py::isinstance<py::none>(obj)) {
    return obj;
  }
  if (!py::isinstance<py::str>(obj)) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the input '" << arg_name << "' should be a str, but got "
                      << py::str(obj.get_type()) << ".";
  }
  auto string_value = obj.cast<std::string>();
  auto enum_value = mindspore::ops::StringToEnumImpl(op_name, arg_name, string_value);
  return py::cast(enum_value);
}

py::object ToPair(const std::string &op_name, const std::string &arg_name, const py::object &arg_val) {
  if (py::isinstance<py::int_>(arg_val) || py::isinstance<py::float_>(arg_val)) {
    int value = arg_val.cast<int>();
    return py::cast(std::vector<int>({value, value}));
  }
  if (py::isinstance<py::list>(arg_val) || py::isinstance<py::tuple>(arg_val)) {
    std::vector<int> values;
    auto items = py::cast<std::vector<py::object>>(arg_val);
    std::transform(items.begin(), items.end(), std::back_inserter(values),
                   [](const py::object &item) { return item.cast<int>(); });
    return py::cast(values);
  }
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '"
                    << py::str(arg_val).cast<std::string>() << ".";
}

py::object To2dPaddings(const std::string &op_name, const std::string &arg_name, const py::object &pad) {
  if (py::isinstance<py::int_>(pad)) {
    int value = pad.cast<int>();
    return py::cast(std::vector<int>({value, value}));
  }
  if (py::isinstance<py::list>(pad) || py::isinstance<py::tuple>(pad)) {
    std::vector<int> values;
    auto items = py::cast<std::vector<py::object>>(pad);
    std::transform(items.begin(), items.end(), std::back_inserter(values),
                   [](const py::object &item) { return item.cast<int>(); });
    return py::cast(values);
  }
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '"
                    << py::str(pad).cast<std::string>() << ".";
}

py::object ToVector(const std::string &op_name, const std::string &arg_name, const py::object &arg) {
  if (py::isinstance<py::int_>(arg)) {
    int value = arg.cast<int>();
    return py::cast(std::vector<int>({value, value}));
  }
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    if (py::len(arg) == 4) {
      py::list arg_list = arg.cast<py::list>();
      return py::cast(std::vector<int>({arg_list[2].cast<int>(), arg_list[3].cast<int>()}));
    }
    std::vector<int> values;
    auto items = py::cast<std::vector<py::object>>(arg);
    std::transform(items.begin(), items.end(), std::back_inserter(values),
                   [](const py::object &item) { return item.cast<int>(); });
    return py::cast(values);
  }
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '"
                    << py::str(arg).cast<std::string>() << ".";
}

py::object ToKernelSize(const std::string &op_name, const std::string &arg_name, const py::object &kernel_size) {
  return ToVector(op_name, arg_name, kernel_size);
}

py::object ToStrides(const std::string &op_name, const std::string &arg_name, const py::object &stride) {
  return ToVector(op_name, arg_name, stride);
}

py::object ToDilations(const std::string &op_name, const std::string &arg_name, const py::object &dilation) {
  return ToVector(op_name, arg_name, dilation);
}

py::object ToOutputPadding(const std::string &op_name, const std::string &arg_name, const py::object &output_padding) {
  return ToVector(op_name, arg_name, output_padding);
}

py::object ToRates(const std::string &op_name, const std::string &arg_name, const py::object &rates) {
  return ToVector(op_name, arg_name, rates);
}

py::object NormalizeIntSequence(const std::string &op_name, const std::string &arg_name, const py::object &arg) {
  if (!py::isinstance<py::list>(arg) && !py::isinstance<py::tuple>(arg)) {
    if (py::isinstance<py::int_>(arg)) {
      return py::cast(std::vector<int>({arg.cast<int>()}));
    }
    MS_EXCEPTION(TypeError) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '"
                            << py::str(arg).cast<std::string>() << ". It should be a list or tuple.";
  }
  auto items = py::cast<std::vector<py::object>>(arg);
  py::tuple int_tuple(items.size());
  auto convert_type = parse::CombineTypesForTypeCast(ops::DT_TENSOR, ops::DT_INT);
  auto convert_func = parse::GetConverterByType(convert_type);
  MS_EXCEPTION_IF_NULL(convert_func);
  size_t i = 0;
  for (const auto &item : items) {
    if (py::isinstance<py::int_>(item)) {
      int_tuple[i] = item;
    } else {
      ValuePtr value = convert_func(item);
      if (!value) {
        MS_EXCEPTION(ValueError) << "For '" << op_name << "', '" << arg_name << "' contain non-integer element: '"
                                 << py::str(item).cast<std::string>() << "'.";
      }
      int_tuple[i] = py::cast(AnfUtils::GetIntValue(value));
    }
    i++;
  }
  return int_tuple;
}

py::object ScalarTensorToScalar(const std::string &op_name, const std::string &arg_name, const py::object &arg) {
  if (py::isinstance<py::int_>(arg)) {
    return py::cast(arg.cast<int>());
  } else if (py::isinstance<py::float_>(arg)) {
    return py::cast(arg.cast<double>());
  } else if (py::isinstance<py::bool_>(arg)) {
    return py::cast(arg.cast<bool>());
  }

  auto tensor = parse::ConvertTensorValue(arg);
  if (tensor) {
    if (tensor->DataDim() != 0) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', '" << arg_name << "' is not a scalar: '"
                               << py::str(arg).cast<std::string>() << "'.";
    }
    auto convert_type = parse::CombineTypesForTypeCast(ops::DT_TENSOR, ops::DT_NUMBER);
    auto convert_func = parse::GetConverterByType(convert_type);
    ValuePtr value = convert_func(arg);
    if (!value) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', '" << arg_name << "' is not an integral type: '"
                              << py::str(arg).cast<std::string>() << "'.";
    }
    if (value->isa<Int64Imm>()) {
      return py::cast(GetValue<int64_t>(value));
    } else if (value->isa<Int32Imm>()) {
      return py::cast(GetValue<int32_t>(value));
    } else if (value->isa<FP32Imm>()) {
      return py::cast(GetValue<float>(value));
    } else if (value->isa<FP64Imm>()) {
      return py::cast(GetValue<double>(value));
    } else if (value->isa<BoolImm>()) {
      return py::cast(GetValue<bool>(value));
    }
  }
  return arg;
}
py::object ScalarTensorToInt(const std::string &op_name, const std::string &arg_name, const py::object &arg) {
  return ScalarTensorToScalar(op_name, arg_name, arg);
}

py::object ScalarTensorToFloat(const std::string &op_name, const std::string &arg_name, const py::object &arg) {
  return ScalarTensorToScalar(op_name, arg_name, arg);
}
}  // namespace pynative
}  // namespace mindspore
