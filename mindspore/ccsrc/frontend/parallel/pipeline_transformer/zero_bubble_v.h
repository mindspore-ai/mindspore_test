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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_ZERO_BUBBLE_V_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_ZERO_BUBBLE_V_H_

#include <string>
#include <queue>
#include <vector>
#include <memory>
#include <utility>
#include "frontend/parallel/pipeline_transformer/detach_backward.h"
#include "frontend/parallel/pipeline_transformer/pipeline_scheduler.h"
#include "frontend/parallel/step_parallel_utils.h"

namespace mindspore {
namespace parallel {
using BorderQueuePtr = std::shared_ptr<std::queue<BorderPair>>;
using BorderVecPtr = std::shared_ptr<std::vector<BorderPair>>;
class ZeroBubbleV : public PipelineScheduler {
 public:
  ZeroBubbleV(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root, int64_t stage, int64_t stage_num)
      : PipelineScheduler(manager, root, stage, stage_num) {}
  virtual ~ZeroBubbleV() = default;
  void Reorder() override;
  void GetBorderNode() override;

 protected:
  void InsertCallControlOrder(const std::vector<BorderPair> &borders,
                              const std::string &tags = "zero_bubble_v_control");
  void InsertControlOrder(const std::vector<BorderPair> &borders, size_t start, size_t end,
                          const std::string &tags = "zero_bubble_v_control");
  void ReorderInnerOverlap(const std::vector<BorderPair> &borders,
                           const std::vector<std::pair<size_t, size_t>> &overlap_border,
                           const std::pair<size_t, size_t> &border_step4,
                           const std::pair<size_t, size_t> &border_step5);

 private:
  static int64_t CalculateOffset(ZeroBubbleV *self) {
    auto offset = common::GetEnv("ZBV_offset");
    int64_t offset_int = 0;
    if (offset.empty()) {
      offset = "0";
    }
    (void)StringToInt(&offset, &offset_int);
    return self->stage_ == 0 ? (self->stage_num_ - 1) : offset_int;
  }
  // State manager
  struct PipelineState {
    const void SafeAdd(BorderVecPtr exec_order, BorderQueuePtr q) const {
      if (!q->empty()) {
        exec_order->emplace_back(q->front());
        q->pop();
      } else {
        MS_LOG(EXCEPTION) << "Unexpected empty queue";
      }
    }

    const void CondAdd(BorderVecPtr exec_order, BorderQueuePtr q, bool condition) const {
      if (condition) {
        SafeAdd(exec_order, q);
      } else {
        q->pop();
      }
    }

    const bool is_first_stage;
    const bool is_last_stage;
    const int64_t offset_int;

    explicit PipelineState(ZeroBubbleV *self)
        : is_first_stage(self->stage_ == 0),
          is_last_stage(self->stage_ == self->stage_num_ - 1),
          offset_int(CalculateOffset(self)) {}
  };
  std::queue<BorderPair> GetTargetBorder(const std::vector<BorderPair> &ori_border, int64_t chunk);
  bool IsDetachedBackward(int64_t chunk, int64_t micro);
  AnfNodePtr GetDwBorder(const Border &bwd_cell, const NodeUsersMap &node_users_map);
  void GetBackwardBorder(const CNodePtr &cnode);
  void ProcessStep1(const PipelineState &state, BorderVecPtr exec_order);
  void ProcessStep2(const PipelineState &state, BorderVecPtr exec_order);
  void ProcessStep3(const PipelineState &state, BorderVecPtr exec_order);
  void ProcessStep4(const PipelineState &state, BorderVecPtr exec_order);
  void ProcessStep5(const PipelineState &state, BorderVecPtr exec_order);
  void ProcessStep6(const PipelineState &state, BorderVecPtr exec_order);
  void ProcessStep7(const PipelineState &state, BorderVecPtr exec_order);
  void ProcessStep8(const PipelineState &state, BorderVecPtr exec_order);
  void ReorderShardedParam(const BorderVecPtr &exec_order);
  void ReorderFor1b1fOverlap(const std::vector<BorderPair> borders, const std::pair<size_t, size_t> &border_step4,
                             const std::pair<size_t, size_t> &border_step5);
  std::vector<BorderPair> dw_border_;
  std::vector<PPInfo> need_detach_info_;
  BorderQueuePtr fwd_b_ph0_;
  BorderQueuePtr fwd_c_ph0_;
  BorderQueuePtr fwd_e_ph0_;
  BorderQueuePtr bwd_b_ph0_;
  BorderQueuePtr bwd_c_ph0_;
  BorderQueuePtr bwd_e_ph0_;
  BorderQueuePtr dw_ph0_;
  BorderQueuePtr fwd_b_ph1_;
  BorderQueuePtr fwd_c_ph1_;
  BorderQueuePtr fwd_e_ph1_;
  BorderQueuePtr bwd_b_ph1_;
  BorderQueuePtr bwd_c_ph1_;
  BorderQueuePtr bwd_e_ph1_;
  BorderQueuePtr dw_ph1_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_ZERO_BUBBLE_V_H_
