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

#include "pybind_api/ir/arg_handler.h"
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

}  // namespace pynative
}  // namespace mindspore
