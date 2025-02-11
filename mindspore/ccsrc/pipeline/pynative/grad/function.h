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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_H_
#ifndef _MSC_VER
#include <cxxabi.h>
#endif
#include <type_traits>
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include "include/common/visible.h"
#include "pybind11/pybind11.h"
#include "abstract/abstract_value.h"

namespace pybind11::detail {
template <>
struct ME_EXPORT type_caster<mindspore::tensor::BaseTensorPtr> {
  PYBIND11_TYPE_CASTER(mindspore::tensor::BaseTensorPtr, _("Tensor"));
  bool load(handle src, bool);
  static handle cast(const mindspore::tensor::BaseTensorPtr &src, return_value_policy, handle);
};
}  // namespace pybind11::detail
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_FUNCTION_H_
