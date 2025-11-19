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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_FORWARD_PYBOOST_FALLBACK_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_FORWARD_PYBOOST_FALLBACK_H_
#include "pybind11/pybind11.h"
#include "include/common/visible.h"

namespace py = pybind11;
namespace mindspore {
namespace pynative {
PYNATIVE_EXPORT void RegFallback(py::module *m);
bool fallback_enabled();
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_FORWARD_PYBOOST_FALLBACK_H_
