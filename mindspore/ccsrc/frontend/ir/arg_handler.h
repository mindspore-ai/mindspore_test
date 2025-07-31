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

#ifndef MINDSPORE_CCSRC_PYBIND_API_IR_ARG_HANDLER_H
#define MINDSPORE_CCSRC_PYBIND_API_IR_ARG_HANDLER_H

#include <string>
#include <memory>
#include <vector>
#include "ir/scalar.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {

namespace pynative {

FRONTEND_EXPORT py::object DtypeToTypeId(const std::string &op_name, const std::string &arg_name,
                                         const py::object &obj);

FRONTEND_EXPORT py::object StrToEnum(const std::string &op_name, const std::string &arg_name, const py::object &obj);

FRONTEND_EXPORT py::object ToPair(const std::string &op_name, const std::string &arg_name, const py::object &arg_val);

FRONTEND_EXPORT py::object To2dPaddings(const std::string &op_name, const std::string &arg_name, const py::object &pad);

FRONTEND_EXPORT py::object ToKernelSize(const std::string &op_name, const std::string &arg_name,
                                        const py::object &kernel_size);

FRONTEND_EXPORT py::object ToStrides(const std::string &op_name, const std::string &arg_name, const py::object &stride);

FRONTEND_EXPORT py::object ToDilations(const std::string &op_name, const std::string &arg_name,
                                       const py::object &dilation);

FRONTEND_EXPORT py::object ToOutputPadding(const std::string &op_name, const std::string &arg_name,
                                           const py::object &output_padding);

FRONTEND_EXPORT py::object ToRates(const std::string &op_name, const std::string &arg_name, const py::object &rates);

FRONTEND_EXPORT py::object NormalizeIntSequence(const std::string &op_name, const std::string &arg_name,
                                                const py::object &arg_val);
FRONTEND_EXPORT py::object ScalarTensorToScalar(const std::string &op_name, const std::string &arg_name,
                                                const py::object &arg_val);
FRONTEND_EXPORT py::object ScalarTensorToInt(const std::string &op_name, const std::string &arg_name,
                                             const py::object &arg_val);
FRONTEND_EXPORT py::object ScalarTensorToFloat(const std::string &op_name, const std::string &arg_name,
                                               const py::object &arg_val);

}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYBIND_API_IR_ARG_HANDLER_H
