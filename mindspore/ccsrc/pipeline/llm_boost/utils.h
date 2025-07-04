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

#ifndef MINDSPORE_CCSRC_PIPELINE_LLM_BOOST_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_LLM_BOOST_UTILS_H_

#include <string>
#include "ir/tensor.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/visible.h"

namespace mindspore {
namespace pipeline {
FRONTEND_EXPORT py::object SetFormat(const py::object &py_tensor, const std::string &format_name);
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_LLM_BOOST_UTILS_H_
