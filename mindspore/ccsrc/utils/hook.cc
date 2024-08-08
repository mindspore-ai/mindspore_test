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

#include "include/common/utils/hook.h"
#include <string>
#include "include/common/utils/convert_utils_py.h"
#include "pybind11/pytypes.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore {
namespace {
py::object GetPythonArg(const ValuePtr &grad) {
  // Get _c_expression tensor
  auto c_expression_tensor = ValueToPyData(grad);
  // Get python tensor
  return ConvertCTensorToPyTensor(c_expression_tensor);
}

ValuePtr GetCValue(const py::object &output) {
  // Convert pyobject output to c++ tensor.
  return ConvertPyObjectToCTensor(output);
}

py::object RunHook(std::map<uint64_t, py::function> *hook_map, const py::object &arg) {
  MS_EXCEPTION_IF_NULL(hook_map);
  py::object grad_out = arg;
  for (auto it = hook_map->begin(); it != hook_map->end();) {
    if (it->second.ptr() == nullptr) {
      MS_LOG(DEBUG) << "Hook id " << it->first << " have been delete by python";
      hook_map->erase(it++);
    } else {
      MS_LOG(DEBUG) << "Run hook id " << it->first << " and its value " << ConvertPyObjToString(it->second);
      auto res = (it->second)(grad_out);
      if (py::isinstance<py::none>(res)) {
        ++it;
        continue;
      }
      if (MS_UNLIKELY(py::isinstance<py::tuple>(res) || py::isinstance<py::list>(res))) {
        MS_LOG(EXCEPTION) << "Tensor hook should be return None or a single value";
      }
      ++it;
      grad_out = res;
    }
  }
  return grad_out;
}
}  // namespace

TensorBackwardHook::TensorBackwardHook(uint64_t tensor_id, const py::function &obj) {
  (void)hook_map_.emplace(tensor_id, obj);
}

TensorBackwardHook::~TensorBackwardHook() {
  py::gil_scoped_acquire acquire_gil;
  hook_map_.clear();
}

ValuePtr TensorBackwardHook::operator()(const ValuePtr &grad) {
  py::gil_scoped_acquire acquire_gil;
  auto py_args = GetPythonArg(grad);
  auto ret = RunHook(&hook_map_, py_args);
  return GetCValue(ret);
}
}  // namespace mindspore
