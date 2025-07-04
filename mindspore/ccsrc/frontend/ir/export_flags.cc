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
#include "frontend/ir/export_flags.h"

namespace mindspore {
const char PYTHON_PRIMITIVE_FLAG[] = "__primitive_flag__";
const char PYTHON_PRIMITIVE_FUNCTION_FLAG[] = "__primitive_function_flag__";
const char PYTHON_CELL_AS_DICT[] = "__cell_as_dict__";
const char PYTHON_CELL_AS_LIST[] = "__cell_as_list__";
const char PYTHON_MS_CLASS[] = "__ms_class__";
const char PYTHON_JIT_FORBIDDEN[] = "__jit_forbidden__";
const char PYTHON_CLASS_MEMBER_NAMESPACE[] = "__class_member_namespace__";
const char PYTHON_FUNCTION_FORBID_REUSE[] = "__function_forbid_reuse__";
const char PYTHON_CELL_LIST_FROM_TOP[] = "__cell_list_from_top__";
}  // namespace mindspore
