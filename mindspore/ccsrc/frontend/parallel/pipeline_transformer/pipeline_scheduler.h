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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_SCHEDULER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_SCHEDULER_H_

#include <set>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

#include "base/base.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/graph_util/generate_graph.h"

namespace mindspore {
namespace parallel {
typedef struct BorderStruct {
  CNodePtr border;
  int64_t chunk;
  int64_t micro;
  int64_t seq_chunk = 0;
} Border;

using BorderPair = std::pair<Border, Border>;
class PipelineScheduler {
 public:
  explicit PipelineScheduler(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root, int64_t stage,
                             int64_t stage_num)
      : manager_(manager), root_(root), stage_(stage), stage_num_(stage_num) {}
  virtual ~PipelineScheduler() = default;
  virtual void GetBorderNode();
  virtual void Reorder() = 0;

 protected:
  void GetChunkNumMicroSize(const std::vector<AnfNodePtr> &all_nodes);
  void GetBackwardBorderNode(const CNodePtr &cnode);
  std::vector<BorderPair> SortInsideMicro(const std::vector<Border> &borders);
  std::pair<Border, Border> SpecifiedBorder(const std::vector<Border> &borders, int64_t chunk, int64_t micro);
  void ControlOrder(const Border &b_prior, const Border &b_last, const std::string &tags = "pipeline_control");
  int64_t micro_size_ = 1;
  int64_t chunk_num_ = 1;
  FuncGraphManagerPtr manager_;
  FuncGraphPtr root_;
  int64_t stage_;
  int64_t stage_num_;
  std::vector<Border> fwd_begin_;
  std::vector<Border> fwd_end_;
  std::vector<Border> bwd_begin_;
  std::vector<Border> bwd_end_;
  std::vector<Border> fwd_cell_;
  std::vector<Border> bwd_cell_;
  std::vector<Border> fwd_params_;
  std::vector<Border> bwd_params_;
};

using SchedulerFunc = std::function<std::shared_ptr<PipelineScheduler>(const FuncGraphManagerPtr &,
                                                                       const FuncGraphPtr &, int64_t, int64_t)>;

class SchedulerCreator {
 public:
  ~SchedulerCreator() = default;

  static SchedulerCreator &Instance() {
    static SchedulerCreator fac = SchedulerCreator();
    return fac;
  }
  void Register(std::string name, SchedulerFunc func) { (void)scheduler_generator_.insert(std::make_pair(name, func)); }
  std::shared_ptr<PipelineScheduler> Create(const std::string &name, const FuncGraphManagerPtr &manager,
                                            const FuncGraphPtr &root, int64_t stage, int64_t stage_num) {
    const auto iter = scheduler_generator_.find(name);
    if (iter == scheduler_generator_.end()) {
      MS_LOG(ERROR) << name << " is not register yet";
      return nullptr;
    }
    return iter->second(manager, root, stage, stage_num);
  }

 private:
  SchedulerCreator() = default;
  std::unordered_map<std::string, SchedulerFunc> scheduler_generator_;
};

class SchedulerRegisterAction {
 public:
  SchedulerRegisterAction(const std::string &name, SchedulerFunc creatfn) noexcept : name_(name) {
    SchedulerCreator::Instance().Register(name, creatfn);
  }
  ~SchedulerRegisterAction() = default;

 private:
  std::string name_;
};

class InterleavedScheduler : public PipelineScheduler {
 public:
  InterleavedScheduler(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root, int64_t stage, int64_t stage_num)
      : PipelineScheduler(manager, root, stage, stage_num) {}
  virtual ~InterleavedScheduler() = default;
  void Reorder() override;

 private:
  void MemoryOptimizedWarmUpPhaseReorder();
  void MemoryOptimizedStablePhaseReorder();
  void MemoryOptimizedReorder();
  void WarmUpPhaseReorder();
  void StablePhaseReorder();
  void LastForwardMicroReorder();
  void EndPhaseReorder();
  AbstractBasePtr GenerateTupleAbstract(const std::vector<AnfNodePtr> &nodes);
  void OptimizerShardCommReorder();
  void ParameterReorder(const std::vector<BorderPair> &sorted_fwd_begin, const std::vector<BorderPair> &sorted_bwd_end);
  std::vector<BorderPair> SortBetweenMicro(const std::vector<Border> &borders, bool is_backward);
  size_t bias_ = 0;
  size_t offset_ = 0;
  bool is_even_stage_ = true;
};
bool SortFuncInsideMicro(const Border &b_i, const Border &b_j);
CNodePtr GetCellByReceive(const AnfNodePtr &node, const FuncGraphManagerPtr &manager);
CNodePtr GetCellBySend(const AnfNodePtr &node);
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_PIPELINE_SCHEDULER_H_
