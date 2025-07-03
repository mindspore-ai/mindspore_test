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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_FUNC_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_FUNC_UTILS_H_

#include <memory>
#include "utils/ms_context.h"
#include "pybind11/pybind11.h"
#include "include/common/visible.h"

namespace py = pybind11;
namespace mindspore {
namespace tensor {
enum TensorPyboostMethod : int {
  ${tensor_methods}
};

class PYNATIVE_EXPORT TensorPyboostMethodRegister {
 public:
  using PyBoostOp = std::function<py::object(const py::list &args)>;
  static void Register(const TensorPyboostMethod methodName, const PyBoostOp &op) { methods_[methodName] = op; }

  static const PyBoostOp &GetOp(TensorPyboostMethod methodName) { return methods_[methodName]; }

 private:
  inline static std::unordered_map<TensorPyboostMethod, PyBoostOp> methods_;
};
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_FUNC_UTILS_H_
