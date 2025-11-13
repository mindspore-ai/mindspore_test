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

#include <algorithm>
#include <vector>
#include <tuple>
#include <string>
#include "ops/op_def.h"
#include "mindspore/ops/op_def/op_enum.h"
#include "pynative/forward/pyboost/arg_handler_py.h"

namespace mindspore {

namespace pynative {

namespace {
using OP_DTYPE = mindspore::ops::OP_DTYPE;

template <typename T, typename U>
std::shared_ptr<U> PyCast(PyObject *obj);

template <>
std::shared_ptr<Int64Imm> PyCast<int64_t, Int64Imm>(PyObject *obj) {
  return std::make_shared<Int64Imm>(static_cast<int64_t>(PyLong_AsLongLong(obj)));
}

}  // namespace

Int64ImmPtr ConvertInt(PyObject *obj) {
  // bool is also an instance of py::int_
  if (PyBool_Check(obj) || !PyLong_Check(obj)) {
    return nullptr;
  }
  return PyCast<int64_t, Int64Imm>(obj);
}

Int64ImmPtr ToDtypePy(const py::object &obj) {
  auto convert = ConvertInt(obj.ptr());
  if (convert != nullptr) {
    return convert;
  }
  if (py::isinstance<mindspore::Type>(obj)) {
    TypePtr type = py::cast<mindspore::TypePtr>(obj);
    return std::make_shared<Int64Imm>(static_cast<int>(type->type_id()));
  }
  return nullptr;
}

std::optional<Int64ImmPtr> DtypeToTypeId(const std::string &op_name, const std::string &arg_name, PyObject *obj) {
  if (obj == Py_None) {
    return std::nullopt;
  }
  py::object py_obj = py::reinterpret_borrow<py::object>(obj);
  if (py::isinstance<mindspore::Type>(py_obj)) {
    return std::make_optional(ToDtypePy(py_obj));
  }
  if (obj == reinterpret_cast<PyObject *>(&PyBool_Type)) {
    auto ms_bool_type = mindspore::Bool();
    return std::make_optional(ToDtypePy(py::cast(ms_bool_type)));
  }
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the input '" << arg_name
                    << "' should be one of ['mindspore dtype', 'bool'], but got " << obj << ".";
  return std::nullopt;
}

std::optional<Int64ImmPtr> StrToEnum(const std::string &op_name, const std::string &arg_name, PyObject *obj) {
  if (obj == Py_None) {
    return std::nullopt;
  }
  if (!PyUnicode_Check(obj)) {
    PyObject *obj_type = PyObject_Str(PyObject_Type(obj));
    const char *type_str = PyUnicode_AsUTF8(obj_type);
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the input '" << arg_name << "' should be a str, but got " << type_str
                      << ".";
    Py_DECREF(obj_type);
  }
  auto string_value = PyUnicode_AsUTF8(obj);
  auto enum_value = mindspore::ops::StringToEnumImpl(op_name, arg_name, string_value);
  return std::make_optional(std::make_shared<Int64Imm>(enum_value));
}

std::vector<int> ToPair(const std::string &op_name, const std::string &arg_name, PyObject *arg_val) {
  if (PyLong_Check(arg_val) || PyFloat_Check(arg_val)) {
    int value = static_cast<int>(PyLong_AsLong(arg_val));
    return {value, value};
  }
  if (PyList_Check(arg_val)) {
    std::vector<int> values;
    Py_ssize_t arg_size = PyList_Size(arg_val);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyList_GetItem(arg_val, i);
      values.push_back(static_cast<int>(PyLong_AsLong(item)));
    }
    return values;
  } else if (PyTuple_Check(arg_val)) {
    std::vector<int> values;
    Py_ssize_t arg_size = PyTuple_Size(arg_val);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyTuple_GetItem(arg_val, i);
      values.push_back(static_cast<int>(PyLong_AsLong(item)));
    }
    return values;
  }
  PyObject *arg_str_py = PyObject_Str(arg_val);
  const char *arg_str = PyUnicode_AsUTF8(arg_str_py);
  Py_DECREF(arg_str_py);
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '" << arg_str << ".";
}

std::vector<int> To2dPaddings(const std::string &op_name, const std::string &arg_name, PyObject *pad) {
  if (PyLong_Check(pad)) {
    int value = static_cast<int>(PyLong_AsLong(pad));
    return {value, value};
  }
  if (PyList_Check(pad)) {
    std::vector<int> values;
    Py_ssize_t arg_size = PyList_Size(pad);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyList_GetItem(pad, i);
      values.push_back(static_cast<int>(PyLong_AsLong(item)));
    }
    return values;
  } else if (PyTuple_Check(pad)) {
    std::vector<int> values;
    Py_ssize_t arg_size = PyTuple_Size(pad);
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyTuple_GetItem(pad, i);
      values.push_back(static_cast<int>(PyLong_AsLong(item)));
    }
    return values;
  }
  PyObject *pad_str_py = PyObject_Str(pad);
  const char *pad_str = PyUnicode_AsUTF8(pad_str_py);
  Py_DECREF(pad_str_py);
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '" << pad_str << ".";
}

std::vector<int> ToVector(const std::string &op_name, const std::string &arg_name, PyObject *arg) {
  if (PyLong_Check(arg)) {
    int value = static_cast<int>(PyLong_AsLong(arg));
    return {value, value};
  }
  // compatible with input format like (N,C,H,W), only choose H and W
  int channels_num = 4;
  if (PyList_Check(arg)) {
    Py_ssize_t arg_size = PyList_Size(arg);
    if (arg_size == channels_num) {
      PyObject *item2 = PyList_GetItem(arg, 2);
      PyObject *item3 = PyList_GetItem(arg, 3);
      return {static_cast<int>(PyLong_AsLong(item2)), static_cast<int>(PyLong_AsLong(item3))};
    }
    std::vector<int> values;
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyList_GetItem(arg, i);
      values.push_back(static_cast<int>(PyLong_AsLong(item)));
    }
    return values;
  } else if (PyTuple_Check(arg)) {
    Py_ssize_t arg_size = PyTuple_Size(arg);
    if (arg_size == channels_num) {
      PyObject *item2 = PyTuple_GetItem(arg, 2);
      PyObject *item3 = PyTuple_GetItem(arg, 3);
      return {static_cast<int>(PyLong_AsLong(item2)), static_cast<int>(PyLong_AsLong(item3))};
    }
    std::vector<int> values;
    for (Py_ssize_t i = 0; i < arg_size; ++i) {
      PyObject *item = PyTuple_GetItem(arg, i);
      values.push_back(static_cast<int>(PyLong_AsLong(item)));
    }
    return values;
  }
  PyObject *arg_str_py = PyObject_Str(arg);
  const char *arg_str = PyUnicode_AsUTF8(arg_str_py);
  Py_DECREF(arg_str_py);
  MS_LOG(EXCEPTION) << "For '" << op_name << "', the value of '" << arg_name << "' is invalid: '" << arg_str << ".";
}

std::vector<int> ToKernelSize(const std::string &op_name, const std::string &arg_name, PyObject *kernel_size) {
  return ToVector(op_name, arg_name, kernel_size);
}

std::vector<int> ToStrides(const std::string &op_name, const std::string &arg_name, PyObject *stride) {
  return ToVector(op_name, arg_name, stride);
}

std::vector<int> ToDilations(const std::string &op_name, const std::string &arg_name, PyObject *dilation) {
  return ToVector(op_name, arg_name, dilation);
}

std::vector<int> ToOutputPadding(const std::string &op_name, const std::string &arg_name, PyObject *output_padding) {
  return ToVector(op_name, arg_name, output_padding);
}

std::vector<int> ToRates(const std::string &op_name, const std::string &arg_name, PyObject *rates) {
  return ToVector(op_name, arg_name, rates);
}

}  // namespace pynative
}  // namespace mindspore
