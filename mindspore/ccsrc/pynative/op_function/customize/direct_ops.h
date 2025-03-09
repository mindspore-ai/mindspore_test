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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_CUSTOMIZE_DIRECT_OPS_H
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_CUSTOMIZE_DIRECT_OPS_H

#include <optional>
#include <vector>
#include "pybind11/pybind11.h"
#include "mindspore/core/include/ir/anf.h"
#include "include/common/visible.h"
#include "mindspore/core/include/ops/op_def.h"
#include "mindspore/core/include/ir/value.h"

namespace py = pybind11;
namespace mindspore::pynative {

PYNATIVE_EXPORT py::object Empty(const py::list &args);
PYNATIVE_EXPORT py::object EmptyLike(const py::list &args);
PYNATIVE_EXPORT py::object NewEmpty(const py::list &args);
PYNATIVE_EXPORT py::object Pyboost_Empty_Base(const PrimitivePtr &prim, const py::list &args);
PYNATIVE_EXPORT py::object Pyboost_Empty_OP(const PrimitivePtr &prim,
                                            const std::vector<mindspore::ops::OP_DTYPE> &source_type,
                                            const ValueTuplePtr &shape, const std::optional<Int64ImmPtr> &dtype,
                                            const std::optional<StringImmPtr> &device);
}  // namespace mindspore::pynative
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_OP_FUNCTION_CUSTOMIZE_DIRECT_OPS_H
