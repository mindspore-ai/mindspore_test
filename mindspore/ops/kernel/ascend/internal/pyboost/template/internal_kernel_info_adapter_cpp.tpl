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
#include "kernel/ascend/internal/pyboost/auto_gen/internal_kernel_info_adapter.h"

#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

namespace mindspore {
namespace kernel {
${kernel_info_adapter_cpp_list}
${kernel_info_adapter_register}
}  // namespace kernel
}  // namespace mindspore
