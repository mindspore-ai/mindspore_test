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

#ifndef MINDSPORE_TESTS_UT_CPP_PYNATIVE_COMMON_H_
#define MINDSPORE_TESTS_UT_CPP_PYNATIVE_COMMON_H_

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "common/mockcpp.h"
#include "pybind11/embed.h"
#include "pybind11/pybind11.h"

#include "ir/tensor.h"
#include "include/common/utils/stub_tensor.h"
#include "include/common/utils/tensor_py.h"

namespace mindspore {
class PyCommon : public testing::Test {
 protected:
  virtual void SetUp() {}

  virtual void TearDown() { GlobalMockObject::verify(); }

  static void SetUpTestCase() {
    if (Py_IsInitialized() == 0) {
      guard_ = std::make_unique<pybind11::scoped_interpreter>();
    }
    m_ = pybind11::module::import("mindspore");
    stub_tensor_module_ = pybind11::module::import("mindspore.common._stub_tensor");
    tensor_module_ = pybind11::module::import("mindspore.common.tensor");
  }

  static void TearDownTestCase() {
    tensor_module_.release();
    stub_tensor_module_.release();
    m_.release();
    guard_ = nullptr;
  }

  pybind11::object NewPyTensor(const tensor::BaseTensorPtr &tensor) {
    py::object tensor_obj = tensor::PackTensorToPyObject(tensor);
    return tensor_module_.attr("Tensor")(tensor_obj);
  }

  pybind11::object NewPyStubTensor(const stub::StubNodePtr &stub_tensor) {
    return stub_tensor_module_.attr("_convert_stub")(stub_tensor);
  }

  pybind11::object NewPyStubTensor(const tensor::BaseTensorPtr &tensor) {
    auto node = stub::MakeTopNode(kTensorType);
    node.second->SetValue(tensor);
    return stub_tensor_module_.attr("_convert_stub")(node.first);
  }

 protected:
  inline static pybind11::module m_;
  inline static pybind11::module stub_tensor_module_;
  inline static pybind11::module tensor_module_;
  inline static std::unique_ptr<pybind11::scoped_interpreter> guard_;
};
}

#endif  // MINDSPORE_TESTS_UT_CPP_PYNATIVE_COMMON_H_
