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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYBIND_API_RESOURCE_MANAGER_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYBIND_API_RESOURCE_MANAGER_H_
#include "include/common/visible.h"
namespace mindspore {
void MemoryRecycle();
void ClearResAtexit();
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYBIND_API_RESOURCE_MANAGER_H_
