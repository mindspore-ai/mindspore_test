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

#ifndef MINDSPORE_CCSRC_PYBIND_API_IR_ARG_HANDLER_H
#define MINDSPORE_CCSRC_PYBIND_API_IR_ARG_HANDLER_H

#include <string>
#include <memory>
#include <vector>
#include <Python.h>
#include "ir/scalar.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {

namespace pynative {

FRONTEND_EXPORT std::optional<Int64ImmPtr> DtypeToTypeId(const std::string &op_name, const std::string &arg_name,
                                                         PyObject *obj);

FRONTEND_EXPORT std::optional<Int64ImmPtr> StrToEnum(const std::string &op_name, const std::string &arg_name,
                                                     PyObject *obj);

FRONTEND_EXPORT std::vector<int> ToPair(const std::string &op_name, const std::string &arg_name, PyObject *arg_val);

FRONTEND_EXPORT std::vector<int> To2dPaddings(const std::string &op_name, const std::string &arg_name, PyObject *pad);

FRONTEND_EXPORT std::vector<int> ToKernelSize(const std::string &op_name, const std::string &arg_name,
                                              PyObject *kernel_size);

FRONTEND_EXPORT std::vector<int> ToStrides(const std::string &op_name, const std::string &arg_name, PyObject *stride);

FRONTEND_EXPORT std::vector<int> ToDilations(const std::string &op_name, const std::string &arg_name,
                                             PyObject *dilation);

FRONTEND_EXPORT std::vector<int> ToOutputPadding(const std::string &op_name, const std::string &arg_name,
                                                 PyObject *output_padding);

FRONTEND_EXPORT std::vector<int> ToRates(const std::string &op_name, const std::string &arg_name, PyObject *rates);

}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_IR_ARG_HANDLER_H
