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

#include "plugin/device/ascend/optimizer/ge/add_attr_to_dump.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>
#include "op_def/framework_ops.h"
#include "op_def/sequence_ops.h"
#include "include/common/env_vars.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
namespace {
constexpr size_t kIndexOne = 1;
constexpr size_t kInputSizeTwo = 2;

bool IsDumpNodes(const BaseRef &ref) {
  if (utils::isa<AnfNodePtr>(ref)) {
    auto node = utils::cast<AnfNodePtr>(ref);
    MS_EXCEPTION_IF_NULL(node);
    if (IsPrimitive(node, prim::kPrimPrint) || IsPrimitive(node, prim::kPrimTensorDump) ||
        IsPrimitive(node, prim::kPrimImageSummary) || IsPrimitive(node, prim::kPrimScalarSummary) ||
        IsPrimitive(node, prim::kPrimTensorSummary) || IsPrimitive(node, prim::kPrimHistogramSummary)) {
      return true;
    }
    return false;
  }
  return false;
}
}  // namespace

const BaseRef AddAttrToDump::DefinePattern() const {
  VarPtr V = std::make_shared<CondVar>(IsDumpNodes);
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

// set attributes slice_size and wait_time for print, tensordump, tensorsummary, etc.
const AnfNodePtr AddAttrToDump::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  if (common::GetDumpSliceSize() > 0) {
    constexpr int64_t kMegaBytes = 1LL << 20;
    int64_t slice_size_in_bytes = common::GetDumpSliceSize() * kMegaBytes;
    (void)primitive->AddAttr("slice_size", MakeValue<int64_t>(slice_size_in_bytes));
    (void)primitive->AddAttr("wait_time", MakeValue<int>(common::GetDumpWaitTime()));
    (void)primitive->AddAttr("slice_sync", MakeValue<bool>(true));
  }

  return cnode;
}
}  // namespace opt
}  // namespace mindspore
