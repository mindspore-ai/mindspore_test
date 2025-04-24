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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_INTERLEAVE_BRANCHES_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_INTERLEAVE_BRANCHES_UTILS_H_

#include <string>
#include <memory>
#include "ir/anf.h"

namespace mindspore {
namespace parallel {
auto const kSplitConcatDepend = "split_concat_depend";
auto const kInterleaveBranchId = "interleave_branch_id";
auto const kInterleaveScopeId = "interleave_scope_id";
auto const kInterleaveSharedBranchId = 0;
auto const kEnableOptimizeMatMulDwOrderFlag = 2;

struct InterLeaveScope {
  CNodePtr fork_node{nullptr};
  CNodePtr merge_node{nullptr};
  bool forward{false};
  size_t scope_id{0};
  FuncGraphPtr graph{nullptr};
  HashMap<CNodePtr, CNodePtr> *matmul_grad_dual_map{nullptr};
};

using InterLeaveScopePtr = std::shared_ptr<InterLeaveScope>;

void InterleaveParallelBranches(const InterLeaveScopePtr &interleave_scope, bool use_dp = false);

void EraseInterLeaveBranchAttr(const CNodePtr &node);

void UpdateMatMulGradDualMap(const CNodePtr &node, HashMap<std::string, CNodePtr> *matmul_unique_id_map_ptr,
                             HashMap<CNodePtr, CNodePtr> *matmul_grad_dual_map_ptr);
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_INTERLEAVE_BRANCHES_UTILS_H_
