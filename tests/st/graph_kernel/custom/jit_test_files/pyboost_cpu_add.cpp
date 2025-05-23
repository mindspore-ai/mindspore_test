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

#include "ms_extension/api.h"

class CustomAddUncontiguous : public ms::pynative::PyboostRunner {
 public:
  using PyboostRunner::PyboostRunner;
  void LaunchKernel() override {
    auto &x = inputs()[0];
    auto &y = inputs()[1];
    auto &z = outputs()[0];
    // Get data pointers
    const int32_t *x_base_ptr = static_cast<const int32_t *>(x.GetDataPtr());
    const int32_t *y_base_ptr = static_cast<const int32_t *>(y.GetDataPtr());
    int32_t *z_base_ptr = static_cast<int32_t *>(z.GetDataPtr());

    // Get shape and strides
    const std::vector<int64_t> &x_shape = x.shape();
    std::vector<int64_t> x_stride = x.stride();
    std::vector<int64_t> y_stride = y.stride();
    std::vector<int64_t> z_stride = z.stride();

    // Ensure all tensors have the same shape
    if (x_shape != y.shape() || y.shape() != z.shape()) {
      throw std::invalid_argument("Shape mismatch in element-wise addition.");
    }

    // Iterate through the tensor elements, assume the tensors are 3-dimensional.
    for (int64_t i = 0; i < x_shape[0]; ++i) {
      for (int64_t j = 0; j < x_shape[1]; ++j) {
        for (int64_t k = 0; k < x_shape[2]; ++k) {
          // Compute linear indices for x, y, and z
          int64_t x_index = x.storage_offset() + i * x_stride[0] + j * x_stride[1] + k * x_stride[2];
          int64_t y_index = y.storage_offset() + i * y_stride[0] + j * y_stride[1] + k * y_stride[2];
          int64_t z_index = z.storage_offset() + i * z_stride[0] + j * z_stride[1] + k * z_stride[2];

          // Perform element-wise addition
          z_base_ptr[z_index] = x_base_ptr[x_index] + y_base_ptr[y_index];
        }
      }
    }
  }
};

ms::Tensor add_uncontiguous(const ms::Tensor &x, const ms::Tensor &y) {
  // assume the shape of x and y is same
  auto out = ms::Tensor(x.data_type(), x.shape());
  if (x.is_contiguous() || y.is_contiguous()) {
    throw std::invalid_argument("For add_uncontiguous, the inputs should be uncontiguous tensor.");
  }
  auto runner = std::make_shared<CustomAddUncontiguous>("Add1");
  runner->Run({x, y}, {out});
  return out;
}

auto pyboost_add1(const ms::Tensor &x, const ms::Tensor &y) {
  x.SetNeedContiguous(false);
  y.SetNeedContiguous(false);
  return ms::pynative::PyboostRunner::Call<1>(add_uncontiguous, x, y);
}

class CustomAddContiguous : public ms::pynative::PyboostRunner {
 public:
  using PyboostRunner::PyboostRunner;
  void LaunchKernel() override {
    auto &x = inputs()[0];
    auto &y = inputs()[1];
    auto &z = outputs()[0];
    const int32_t *x_base_ptr = static_cast<const int32_t *>(x.GetDataPtr());
    const int32_t *y_base_ptr = static_cast<const int32_t *>(y.GetDataPtr());
    int32_t *z_base_ptr = static_cast<int32_t *>(z.GetDataPtr());
    for (size_t i = 0; i < x.numel(); i++) {
      z_base_ptr[i] = x_base_ptr[i] + y_base_ptr[i];
    }
  }
};

ms::Tensor add_contiguous(const ms::Tensor &x, const ms::Tensor &y) {
  // assume the shape of x and y is same
  auto out = ms::Tensor(x.data_type(), x.shape());
  if (!x.is_contiguous() || !y.is_contiguous()) {
    throw std::invalid_argument("For add_contiguous, the inputs should be contiguous tensor.");
  }
  auto runner = std::make_shared<CustomAddContiguous>("Add2");
  runner->Run({x, y}, {out});
  return out;
}

auto pyboost_add2(const ms::Tensor &x, const ms::Tensor &y) {
  return ms::pynative::PyboostRunner::Call<1>(add_contiguous, x, y);
}

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("add_uncontiguous", &pyboost_add1, "add, support uncontiguous", pybind11::arg("x"), pybind11::arg("y"));
  m.def("add_contiguous", &pyboost_add2, "add, only support contiguous", pybind11::arg("x"), pybind11::arg("y"));
}
