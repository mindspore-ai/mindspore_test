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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_FUNC_REG_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_FUNC_REG_H_

#include <memory>
#include "mindspore/core/include/ir/tensor.h"
#include "mindspore/core/include/ir/base_tensor.h"
#include "utils/ms_context.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;
namespace mindspore {
namespace tensor {
class StubTensorConverterImpl;

class StubTensorConverter {
 public:
  static StubTensorConverter &GetInstance();
  py::object ToPython(const py::object &object);
  void Clear();

 private:
  StubTensorConverter() = default;
  ~StubTensorConverter() = default;
  DISABLE_COPY_AND_ASSIGN(StubTensorConverter);

  std::unique_ptr<StubTensorConverterImpl> impl_;
};

${func_header_body}

${func_def_list}

void RegTensorFunc(py::class_<Tensor, BaseTensor, std::shared_ptr<Tensor>> *tensor_class);
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_FUNC_REG_H_
