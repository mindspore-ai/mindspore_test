/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PRIMFUNC_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PRIMFUNC_UTILS_H_

#include <string>
#include <vector>
#include "ops/op_def.h"
#include "abstract/abstract_value.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore::ops {
COMMON_EXPORT bool ValidateArgsType(const AbstractBasePtr &abs_arg, OP_DTYPE type_arg);
COMMON_EXPORT std::string EnumToString(OP_DTYPE dtype);
COMMON_EXPORT std::string BuildOpErrorMsg(const OpDefPtr &op_def, const std::vector<std::string> &op_type_list);
COMMON_EXPORT std::string BuildOpInputsErrorMsg(const OpDefPtr &op_def, const std::string &arg_name,
                                                const TypePtr &arg_type);
COMMON_EXPORT std::vector<OP_DTYPE> GetSourceDtypeByArgHandler(const AbstractBasePtr &abs_arg, OP_DTYPE type_arg);
}  // namespace mindspore::ops

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PRIMFUNC_UTILS_H_
