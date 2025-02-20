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

#include "mindspore/ccsrc/pynative/op_function/auto_generate/tensor_func_utils.h"

namespace mindspore {
namespace tensor {
StubTensorConverterImpl::StubTensorConverterImpl() {
  auto stub_tensor_module = py::module::import("mindspore.common._stub_tensor");
  converter_ = stub_tensor_module.attr("_convert_stub");
}

StubTensorConverterImpl::~StubTensorConverterImpl() {
  py::gil_scoped_acquire gil_acquire;
  converter_ = py::object();
}

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

}  // namespace tensor
}  // namespace mindspore
