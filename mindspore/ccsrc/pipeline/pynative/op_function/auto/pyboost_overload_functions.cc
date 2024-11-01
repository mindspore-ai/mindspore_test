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


#include "ir/named.h"
#include "include/common/pybind_api/api_register.h"
#include "pybind_api/ir/tensor_func_reg.h"

namespace mindspore::pynative {
class ClampFunctional : public Functional {
 public:
  ClampFunctional() : Functional("clamp") {};
  ~ClampFunctional() = default;
  py::object Call(const py::args &args, const py::kwargs &kwargs) {
    return mindspore::tensor::TensorMethodClamp(py::none(), args, kwargs);
  }
};

void RegisterFunctional(py::module *m) {
  (void)py::class_<ClampFunctional, Functional, std::shared_ptr<ClampFunctional>>(
    *m, "ClampFunctional_")
    .def(py::init<>())
    .def("__call__", &ClampFunctional::Call, "Call Clamp functional.");
}
}  // namespace mindspore::pynative