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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_INTERLEAVE_PARALLEL_INTERLEAVE_PARALLEL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_INTERLEAVE_PARALLEL_INTERLEAVE_PARALLEL_H_

#include <vector>
#include "ir/anf.h"
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
void SplitNotParallelCareOpsInterleaved(const FuncGraphPtr &root);
void EraseVirtualConverter(const FuncGraphPtr &root);
void ConvertInterleaveAllGatherToConcat(const FuncGraphPtr &func_graph, const CNodePtr &virtual_converter_end,
                                        const std::vector<std::vector<std::vector<int64_t>>> &ag_group_ranks_vectors);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_NODE_CHECK_H_
