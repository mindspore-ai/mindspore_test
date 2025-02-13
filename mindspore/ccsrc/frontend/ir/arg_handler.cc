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
#include "pipeline/pynative/pynative_utils.h"
#include "mindspore/ops/op_def/op_enum.h"

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

std::optional<Int64ImmPtr> DtypeToTypeId(const std::string &op_name, const std::string &arg_name,
                                         const py::object &obj) {
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  if (!py::isinstance<mindspore::Type>(obj)) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the input '" << arg_name << "' should be mindspore dtype, but got "
                      << obj << ".";
  }
  return std::make_optional(ToDtype(obj));
}

std::optional<Int64ImmPtr> StrToEnum(const std::string &op_name, const std::string &arg_name, const py::object &obj) {
  if (py::isinstance<py::none>(obj)) {
    return std::nullopt;
  }
  if (!py::isinstance<py::str>(obj)) {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the input '" << arg_name << "' should be a str, but got "
                      << py::str(obj.get_type()) << ".";
  }
  auto string_value = obj.cast<std::string>();
  auto enum_value = mindspore::ops::StringToEnumImpl(op_name, arg_name, string_value);
  return std::make_optional(std::make_shared<Int64Imm>(enum_value));
}

std::vector<int> ToPair(const std::string &op_name, const std::string &arg_name, const py::object &arg_val) {
  if (py::isinstance<py::int_>(arg_val) || py::isinstance<py::float_>(arg_val)) {
    int value = arg_val.cast<int>();
    return {value, value};
  }
  if (py::isinstance<py::list>(arg_val) || py::isinstance<py::tuple>(arg_val)) {
    std::vector<int> values;
    auto items = py::cast<std::vector<py::object>>(arg_val);
    std::transform(items.begin(), items.end(), std::back_inserter(values),
                   [](const py::object &item) { return item.cast<int>(); });
    return values;
  }
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '"
                    << py::str(arg_val).cast<std::string>() << ".";
}

std::vector<int> To2dPaddings(const std::string &op_name, const std::string &arg_name, const py::object &pad) {
  if (py::isinstance<py::int_>(pad)) {
    int value = pad.cast<int>();
    return {value, value};
  }
  if (py::isinstance<py::list>(pad) || py::isinstance<py::tuple>(pad)) {
    std::vector<int> values;
    auto items = py::cast<std::vector<py::object>>(pad);
    std::transform(items.begin(), items.end(), std::back_inserter(values),
                   [](const py::object &item) { return item.cast<int>(); });
    return values;
  }
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '"
                    << py::str(pad).cast<std::string>() << ".";
}

std::vector<int> ToVector(const std::string &op_name, const std::string &arg_name, const py::object &arg) {
  if (py::isinstance<py::int_>(arg)) {
    int value = arg.cast<int>();
    return {value, value};
  }
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    if (py::len(arg) == 4) {
      py::list arg_list = arg.cast<py::list>();
      return {arg_list[2].cast<int>(), arg_list[3].cast<int>()};
    }
    std::vector<int> values;
    auto items = py::cast<std::vector<py::object>>(arg);
    std::transform(items.begin(), items.end(), std::back_inserter(values),
                   [](const py::object &item) { return item.cast<int>(); });
    return values;
  }
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '"
                    << py::str(arg).cast<std::string>() << ".";
}

std::vector<int> ToKernelSize(const std::string &op_name, const std::string &arg_name, const py::object &kernel_size) {
  return ToVector(op_name, arg_name, kernel_size);
}

std::vector<int> ToStrides(const std::string &op_name, const std::string &arg_name, const py::object &stride) {
  return ToVector(op_name, arg_name, stride);
}

std::vector<int> ToDilations(const std::string &op_name, const std::string &arg_name, const py::object &dilation) {
  return ToVector(op_name, arg_name, dilation);
}

std::vector<int> ToOutputPadding(const std::string &op_name, const std::string &arg_name,
                                 const py::object &output_padding) {
  return ToVector(op_name, arg_name, output_padding);
}

std::vector<int> ToRates(const std::string &op_name, const std::string &arg_name, const py::object &rates) {
  return ToVector(op_name, arg_name, rates);
}

}  // namespace pynative
}  // namespace mindspore
