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
#include "pybind_api/ir/tensor_func_reg.h"
#include "pipeline/pynative/op_function/converter.h"
#include <memory>

namespace mindspore {
namespace tensor {
class StubTensorConverterImpl {
 public:
  StubTensorConverterImpl() {
    auto stub_tensor_module = py::module::import("mindspore.common._stub_tensor");
    converter_ = stub_tensor_module.attr("_convert_stub");
  }

  ~StubTensorConverterImpl() {
    py::gil_scoped_acquire gil_acquire;
    converter_ = py::none();
  }

  py::object Convert(const py::object &object) { return converter_(object); }

 private:
  py::object converter_;
};

StubTensorConverter &StubTensorConverter::GetInstance() {
  static StubTensorConverter instance;
  return instance;
}

void StubTensorConverter::Clear() { impl_ = nullptr; }

py::object StubTensorConverter::ToPython(const py::object &object) {
  if (impl_ == nullptr) {
    impl_ = std::make_unique<StubTensorConverterImpl>();
  }
  return impl_->Convert(object);
}

inline py::object ToPython(const py::object &object) { return StubTensorConverter::GetInstance().ToPython(object); }

${func_call_body}

void RegTensorFunc(py::class_<Tensor, BaseTensor, std::shared_ptr<Tensor>> *tensor_class) {
    ${func_def_body}
}
}  // namespace tensor
}  // namespace mindspore