/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_FLOAT32_REDISTRIBUTION_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_FLOAT32_REDISTRIBUTION_H_

#include "ir/anf.h"

namespace mindspore {
namespace parallel {
// Automatically insert duplicated recomputed nodes.
void Float32Redistribution(const FuncGraphPtr &graph);
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_FLOAT32_REDISTRIBUTION_H_
