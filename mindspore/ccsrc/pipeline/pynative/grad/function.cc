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

#include "pipeline/pynative/grad/function.h"
#include <algorithm>
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/tensor_py.h"

namespace pybind11::detail {
bool type_caster<mindspore::tensor::BaseTensorPtr>::load(handle src, bool) {
  if (mindspore::tensor::IsTensorPy(src)) {
    value = mindspore::tensor::ConvertToTensor(src);
    return true;
  }
  if (mindspore::IsStubTensor(src)) {
    value = mindspore::ConvertStubTensor(src);
    return true;
  }
  return false;
}

handle type_caster<mindspore::tensor::BaseTensorPtr>::cast(const mindspore::tensor::BaseTensorPtr &src,
                                                           return_value_policy, handle) {
  auto tensor_py = std::make_shared<mindspore::tensor::TensorPy>(std::make_shared<mindspore::tensor::Tensor>(*src));
  auto obj = py::cast(tensor_py);
  const auto py_tensor_class = module::import("mindspore.common.tensor").attr("Tensor");
  return py_tensor_class(obj).release();
}
}  // namespace pybind11::detail
