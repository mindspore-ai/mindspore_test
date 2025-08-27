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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_GPIPE_INTERLEAVE_SCHEDULER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_GPIPE_INTERLEAVE_SCHEDULER_H_

#include <set>
#include <utility>
#include <string>
#include <memory>
#include <vector>

#include "base/base.h"
#include "frontend/parallel/pipeline_transformer/pipeline_scheduler.h"

namespace mindspore {
namespace parallel {
class GpipeInterleavedScheduler : public PipelineScheduler {
 public:
  GpipeInterleavedScheduler(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root, int64_t stage,
                            int64_t stage_num)
      : PipelineScheduler(manager, root, stage, stage_num) {}
  virtual ~GpipeInterleavedScheduler() = default;
  void Reorder() override;

 private:
  std::vector<BorderPair> SortBetweenMicro(const std::vector<Border> &borders, bool is_backward);
  void GetBackwardBorderNode(const CNodePtr &cnode);
  void ForwardReorder(size_t bias, int64_t flag);
  void GetChunkNum(const std::vector<AnfNodePtr> &all_nodes);
  AbstractBasePtr GenerateTupleAbstract(const std::vector<AnfNodePtr> &nodes);
  void OptimizerShardCommReorder();
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_GPIPE_INTERLEAVE_SCHEDULER_H_
