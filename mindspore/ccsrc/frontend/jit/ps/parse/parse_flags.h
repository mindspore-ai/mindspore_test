/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_JIT_PS_PARSE_PARSE_FLAGS_H_
#define MINDSPORE_CCSRC_FRONTEND_JIT_PS_PARSE_PARSE_FLAGS_H_

#include "include/common/visible.h"

namespace mindspore {
FRONTEND_EXPORT extern const char PYTHON_PRIMITIVE_FLAG[];
FRONTEND_EXPORT extern const char PYTHON_PRIMITIVE_FUNCTION_FLAG[];
FRONTEND_EXPORT extern const char PYTHON_CELL_AS_DICT[];
FRONTEND_EXPORT extern const char PYTHON_CELL_AS_LIST[];
FRONTEND_EXPORT extern const char PYTHON_MS_CLASS[];
FRONTEND_EXPORT extern const char PYTHON_JIT_FORBIDDEN[];
FRONTEND_EXPORT extern const char PYTHON_CLASS_MEMBER_NAMESPACE[];
FRONTEND_EXPORT extern const char PYTHON_FUNCTION_FORBID_REUSE[];
FRONTEND_EXPORT extern const char PYTHON_CELL_LIST_FROM_TOP[];
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_JIT_PS_PARSE_PARSE_FLAGS_H_
