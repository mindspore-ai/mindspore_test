/**
 * Copyright 2025Huawei Technologies Co., Ltd
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

#include <queue>
#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>
#include <memory>
#include "frontend/parallel/pipeline_transformer/zero_bubble_v.h"

namespace mindspore {
namespace parallel {
bool ZeroBubbleV::IsDetachedBackward(int64_t chunk, int64_t micro) {
  return std::any_of(need_detach_info_.begin(), need_detach_info_.end(),
                     [chunk, micro](const auto &info) { return info.chunk == chunk && info.micro == micro; });
}

void ZeroBubbleV::GetBackwardBorder(const CNodePtr &cnode) {
  auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
  auto micro = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
  Border border = {cnode, chunk, micro};
  Border border_cell = {nullptr, chunk, micro};
  auto node_users_map = manager_->node_users();

  if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
    auto bwd_cell = GetCellBySend(cnode);
    MS_EXCEPTION_IF_NULL(bwd_cell);
    if ((stage_ == stage_num_ - 1 && chunk == 0) || (stage_ == 0 && chunk == 1)) {
      Border bwd_begin = {bwd_cell, chunk, micro};
      bwd_begin_.emplace_back(bwd_begin);
      border_cell.border = bwd_cell;
      bwd_cell_.emplace_back(border_cell);
    }
    bwd_end_.emplace_back(border);
  }
  if (cnode->HasPrimalAttr(PIPELINE_END)) {
    auto bwd_cell = GetCellByReceive(cnode, manager_);
    MS_EXCEPTION_IF_NULL(bwd_cell);
    if ((stage_ == 0 && chunk == 0) || (stage_ == stage_num_ - 1 && chunk == 1)) {
      Border bwd_end = {bwd_cell, chunk, micro};
      bwd_end_.emplace_back(bwd_end);
    }
    border_cell.border = bwd_cell;
    bwd_cell_.emplace_back(border_cell);
    bwd_begin_.emplace_back(border);
  }
  auto dw_node = GetDwBorder(border_cell, node_users_map);
  if (dw_node != nullptr) {
    auto dw_cnode = dw_node->cast<CNodePtr>();
    if (dw_cnode->HasPrimalAttr(VISITED)) {
      return;
    }
    dw_cnode->AddPrimalAttr(VISITED, MakeValue(true));
    Border dw_border = {dw_cnode, chunk, micro};
    dw_border_.emplace_back(std::make_pair(dw_border, dw_border));
  }
  if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
    bwd_params_.emplace_back(border);
  }
}

AnfNodePtr ZeroBubbleV::GetDwBorder(const Border &bwd_cell, const NodeUsersMap &node_users_map) {
  auto cnode = bwd_cell.border;
  if (cnode == nullptr) {
    return nullptr;
  }
  auto tuple_get_item = cnode->input(0)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  auto closure_call = tuple_get_item->input(1);
  auto call_users = node_users_map.at(closure_call);
  for (const auto &call_user : call_users) {
    auto cuser = call_user.first->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cuser, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto index = GetTupleGetItemIndex(cuser);
    if (index != DW_INDEX) {
      continue;
    }
    auto dw_call = node_users_map.at(cuser).front();
    if (dw_call.second != 0) {
      MS_LOG(EXCEPTION) << "Can't find dw call node.";
    }
    return node_users_map.at(cuser).front().first;
  }
  return nullptr;
}

void ZeroBubbleV::GetBorderNode() {
  auto all_nodes = TopoSort(root_->get_return(), SuccDeeperSimple);
  GetChunkNumMicroSize(all_nodes);
  need_detach_info_ = InferNeedDetachInfo(stage_, micro_size_);
  if (chunk_num_ != 2) {
    MS_LOG(EXCEPTION) << "Zero Bubble V scheduler only support chunk_num is 2, but got:" << chunk_num_;
  }
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!cnode->HasPrimalAttr(CHUNK) || !cnode->HasPrimalAttr(MICRO)) {
      continue;
    }
    if (cnode->HasPrimalAttr(kPrimalAttrForwardNodeName)) {
      GetBackwardBorder(cnode);
      continue;
    }
    auto chunk = GetValue<int64_t>(cnode->GetPrimalAttr(CHUNK));
    auto micro = GetValue<int64_t>(cnode->GetPrimalAttr(MICRO));
    Border border = {cnode, chunk, micro};
    Border border_cell = {nullptr, chunk, micro};
    if (cnode->HasPrimalAttr(PIPELINE_BEGIN)) {
      auto fwd_cell = GetCellByReceive(cnode, manager_);
      MS_EXCEPTION_IF_NULL(fwd_cell);
      if ((chunk == 1 && stage_ == 0) || (chunk == 0 && stage_ == stage_num_ - 1)) {
        Border fwd_end = {fwd_cell, chunk, micro};
        fwd_end_.emplace_back(fwd_end);
      }
      border_cell.border = fwd_cell;
      fwd_cell_.emplace_back(border_cell);
      fwd_begin_.emplace_back(border);
      continue;
    }

    if (cnode->HasPrimalAttr(PIPELINE_END)) {
      auto fwd_cell = GetCellBySend(cnode);
      MS_EXCEPTION_IF_NULL(fwd_cell);
      if ((stage_ == 0 && chunk == 0) || (stage_ == stage_num_ - 1 && chunk == 1)) {
        Border fwd_begin = {fwd_cell, chunk, micro};
        fwd_begin_.emplace_back(fwd_begin);
        border_cell.border = fwd_cell;
        fwd_cell_.emplace_back(border_cell);
      }
      fwd_end_.emplace_back(border);
      continue;
    }
    if (cnode->HasPrimalAttr(PIPELINE_PARAM)) {
      fwd_params_.emplace_back(border);
      continue;
    }
  }
}

std::queue<BorderPair> ZeroBubbleV::GetTargetBorder(const std::vector<BorderPair> &ori_border, int64_t chunk) {
  std::queue<BorderPair> border_q;
  std::vector<BorderPair> border_v;
  std::copy_if(ori_border.begin(), ori_border.end(), std::back_inserter(border_v),
               [chunk](const auto &border) { return border.first.chunk == chunk; });
  std::sort(border_v.begin(), border_v.end(),
            [](BorderPair a, BorderPair b) -> bool { return a.first.micro < b.first.micro; });

  for (const auto &border : border_v) {
    border_q.push(border);
  }
  return border_q;
}

void ZeroBubbleV::ProcessStep1(const PipelineState &state, BorderVecPtr exec_order) {
  // step1: nF0
  const int64_t steps = (stage_num_ - stage_ - 1) * 2;
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, fwd_b_ph0_);
    state.SafeAdd(exec_order, fwd_c_ph0_);
    state.SafeAdd(exec_order, fwd_e_ph0_);
  }
}

void ZeroBubbleV::ProcessStep2(const PipelineState &state, BorderVecPtr exec_order) {
  // step2: nF0F1
  const int64_t steps = stage_ + 1;
  state.SafeAdd(exec_order, fwd_b_ph0_);
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, fwd_c_ph0_);
    state.CondAdd(exec_order, fwd_b_ph0_, !state.is_first_stage);
    state.SafeAdd(exec_order, fwd_b_ph1_);
    state.SafeAdd(exec_order, fwd_c_ph1_);
    state.CondAdd(exec_order, fwd_e_ph1_, !state.is_first_stage);
    state.CondAdd(exec_order, fwd_e_ph0_, !state.is_last_stage);
  }
}

void ZeroBubbleV::ProcessStep3(const PipelineState &state, BorderVecPtr exec_order) {
  // step3: nB1W1F1
  const int64_t steps = stage_num_ - stage_ - 1;
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, bwd_b_ph1_);
    state.SafeAdd(exec_order, bwd_c_ph1_);
    state.SafeAdd(exec_order, fwd_b_ph1_);
    state.SafeAdd(exec_order, bwd_e_ph1_);
    state.SafeAdd(exec_order, dw_ph1_);
    state.SafeAdd(exec_order, fwd_c_ph1_);
    state.SafeAdd(exec_order, fwd_e_ph1_);
  }
}

void ZeroBubbleV::ProcessStep4(const PipelineState &state, BorderVecPtr exec_order) {
  // step4: nB1F0B0F1
  const int64_t steps = micro_size_ - stage_num_ * 2 + stage_ + 1;
  for (int64_t i = 0; i < steps; ++i) {
    // B1F0
    if (i != 0) {
      state.CondAdd(exec_order, fwd_b_ph0_, !state.is_first_stage);
    } else {
      state.SafeAdd(exec_order, bwd_b_ph1_);
    }
    state.SafeAdd(exec_order, bwd_c_ph1_);
    state.CondAdd(exec_order, bwd_b_ph0_, !state.is_last_stage);
    state.CondAdd(exec_order, bwd_e_ph1_, !state.is_last_stage);
    state.SafeAdd(exec_order, fwd_c_ph0_);
    state.CondAdd(exec_order, fwd_b_ph1_, !state.is_last_stage);
    state.CondAdd(exec_order, fwd_e_ph0_, !state.is_last_stage);

    // B0F1
    state.SafeAdd(exec_order, bwd_c_ph0_);
    state.CondAdd(exec_order, bwd_e_ph0_, !state.is_first_stage);
    state.CondAdd(exec_order, bwd_b_ph1_, !state.is_first_stage);
    state.SafeAdd(exec_order, fwd_c_ph1_);
    state.SafeAdd(exec_order, fwd_e_ph1_);
  }
}

void ZeroBubbleV::ProcessStep5(const PipelineState &state, BorderVecPtr exec_order) {
  // step5: nB1B0F1
  const int64_t steps = stage_num_ - stage_ - 1;
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, bwd_c_ph1_);
    state.SafeAdd(exec_order, bwd_b_ph0_);
    state.SafeAdd(exec_order, bwd_e_ph1_);
    state.SafeAdd(exec_order, fwd_b_ph1_);
    state.SafeAdd(exec_order, bwd_c_ph0_);
    state.SafeAdd(exec_order, bwd_e_ph0_);
    state.CondAdd(exec_order, bwd_b_ph1_, !state.is_first_stage);
    state.SafeAdd(exec_order, fwd_c_ph1_);
    state.SafeAdd(exec_order, fwd_e_ph1_);
  }
}

void ZeroBubbleV::ProcessStep6(const PipelineState &state, BorderVecPtr exec_order) {
  // step6: nB1B0
  const int64_t steps = stage_ + 1;
  for (int64_t i = 0; i < steps; ++i) {
    state.SafeAdd(exec_order, bwd_c_ph1_);
    state.CondAdd(exec_order, bwd_b_ph0_, !state.is_last_stage);
    state.SafeAdd(exec_order, bwd_e_ph1_);
    state.SafeAdd(exec_order, bwd_c_ph0_);
    state.SafeAdd(exec_order, bwd_e_ph0_);
    if (i != steps - 1) {
      state.SafeAdd(exec_order, bwd_b_ph1_);
    }
  }
}

void ZeroBubbleV::ProcessStep7(const PipelineState &state, BorderVecPtr exec_order) {
  // step7: nWB0
  const int64_t steps = stage_num_ - stage_ - 1;
  for (int64_t i = 0; i < steps; ++i) {
    if (i >= state.offset_int) {
      state.SafeAdd(exec_order, dw_ph0_);
    }
    state.SafeAdd(exec_order, bwd_b_ph0_);
    state.SafeAdd(exec_order, bwd_c_ph0_);
    state.SafeAdd(exec_order, bwd_e_ph0_);
  }
}

void ZeroBubbleV::ProcessStep8(const PipelineState &state, BorderVecPtr exec_order) {
  // step8: nW
  while (!dw_ph0_->empty()) {
    state.SafeAdd(exec_order, dw_ph0_);
  }
  while (!dw_ph1_->empty()) {
    state.SafeAdd(exec_order, dw_ph1_);
  }
}

void ZeroBubbleV::Reorder() {
  auto fwd_begin = SortInsideMicro(fwd_begin_);
  auto fwd_end = SortInsideMicro(fwd_end_);
  auto fwd_cell = SortInsideMicro(fwd_cell_);
  auto bwd_begin = SortInsideMicro(bwd_begin_);
  auto bwd_end = SortInsideMicro(bwd_end_);
  auto bwd_cell = SortInsideMicro(bwd_cell_);

  fwd_b_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_begin, 0));
  fwd_c_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_cell, 0));
  fwd_e_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_end, 0));
  bwd_b_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_begin, 0));
  bwd_c_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_cell, 0));
  bwd_e_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_end, 0));
  dw_ph0_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(dw_border_, 0));

  fwd_b_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_begin, 1));
  fwd_c_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_cell, 1));
  fwd_e_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(fwd_end, 1));
  bwd_b_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_begin, 1));
  bwd_c_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_cell, 1));
  bwd_e_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(bwd_end, 1));
  dw_ph1_ = std::make_shared<std::queue<BorderPair>>(GetTargetBorder(dw_border_, 1));
  PipelineState state(this);
  BorderVecPtr exec_order = std::make_shared<std::vector<BorderPair>>();

  ProcessStep1(state, exec_order);
  ProcessStep2(state, exec_order);
  ProcessStep3(state, exec_order);
  ProcessStep4(state, exec_order);
  ProcessStep5(state, exec_order);
  ProcessStep6(state, exec_order);
  ProcessStep7(state, exec_order);
  ProcessStep8(state, exec_order);

  for (size_t i = 0; i < exec_order->size() - 1; ++i) {
    auto prior_border = exec_order->at(i).second;
    auto last_border = exec_order->at(i + 1).first;
    ControlOrder(prior_border, last_border);
  }
}

SchedulerRegisterAction PipelineSchedulerZeroBubbleV(parallel::kPipelineZeroBubbleV,
                                                     [](const FuncGraphManagerPtr &manager, const FuncGraphPtr &root,
                                                        int64_t stage, int64_t stage_num) {
                                                       return std::make_shared<parallel::ZeroBubbleV>(manager, root,
                                                                                                      stage, stage_num);
                                                     });
}  // namespace parallel
}  // namespace mindspore
