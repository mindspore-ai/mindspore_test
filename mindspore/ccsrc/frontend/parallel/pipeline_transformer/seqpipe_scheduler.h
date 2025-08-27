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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_SEQPIPE_SCHEDULER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_SEQPIPE_SCHEDULER_H_

#include <set>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

#include "base/base.h"
#include "frontend/parallel/pipeline_transformer/pipeline_scheduler.h"

namespace mindspore {
namespace parallel {
typedef struct TripletStruct {
  size_t seq_chunk;
  size_t micro;
  size_t chunk;
  bool is_bp;
  bool operator==(const TripletStruct &cmp) {
    return seq_chunk == cmp.seq_chunk && micro == cmp.micro && chunk == cmp.chunk && is_bp == cmp.is_bp;
  }
} Triplet;

typedef struct SchedulerNodeStruct {
  std::string type;
  int64_t seq_chunk;
  int64_t micro;
  int64_t chunk;
  bool is_bp;
  bool operator==(const SchedulerNodeStruct &cmp) const {
    return type == cmp.type && seq_chunk == cmp.seq_chunk && micro == cmp.micro && chunk == cmp.chunk &&
           is_bp == cmp.is_bp;
  }
} SchedulerNode;

struct SchedulerNodeHash {
  size_t operator()(const SchedulerNode &node) const {
    size_t hash = std::hash<std::string>()(node.type);
    hash ^= std::hash<int64_t>()(node.seq_chunk) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int64_t>()(node.micro) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int64_t>()(node.chunk) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<bool>()(node.is_bp) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return hash;
  }
};

class SeqpipeScheduler : public PipelineScheduler {
 public:
  SeqpipeScheduler(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root, int64_t stage, int64_t stage_num)
      : PipelineScheduler(manager, root, stage, stage_num) {}
  virtual ~SeqpipeScheduler() = default;
  void Reorder() override;
  void GetBorderNode() override;
  void SchedulerOrder();

 protected:
  void ExtractDataStruct();
  size_t GetOrderIndex(size_t seq_chunk, size_t micro, size_t chunk, bool is_bp, std::string type = "origin");
  int64_t CurrentChunkSize(size_t recv_node_index);
  std::vector<std::vector<std::vector<BorderPair>>> BorderMap(const std::vector<Border> &borders);
  std::vector<Triplet> ExecuteOrder(size_t warm_up_size);
  std::vector<Triplet> MakeExecuteOrder(const std::vector<size_t> &micro_index_list,
                                        const std::vector<size_t> &chunk_index_list, size_t warm_up_size);
  virtual void ComputeCycleList();
  std::vector<Triplet> FpBpExecuteOrder(bool is_bp);
  virtual BorderPair GetBorderNodeRecv(size_t index);
  BorderPair GetBorderNode(const std::string &border_type, size_t index);
  AbstractBasePtr GenerateTupleAbstract(const std::vector<AnfNodePtr> &nodes);
  void OptimizerShardCommReorder();
  void GetCleanAssigns();
  void SetCleanAssignsMicro();
  void ControlCleanAssigns();
  BorderPair BorderRecv(size_t index, size_t recv_node_index);
  BorderPair ControlAdvancedRecv(size_t index, size_t recv_node_index);
  void SpecialControl(const std::pair<BorderStruct, BorderStruct> &origin_recv,
                      const std::pair<BorderStruct, BorderStruct> &send,
                      const std::pair<BorderStruct, BorderStruct> &recv,
                      const std::pair<BorderStruct, BorderStruct> &prior_cell, size_t index);
  void SendRecvControl(const std::pair<BorderStruct, BorderStruct> &send,
                       const std::pair<BorderStruct, BorderStruct> &recv);
  virtual void ControlSendRecvOrder(const BorderPair &send, const BorderPair &post_recv, size_t index);
  virtual void ComputeBias();
  void ComputePrefetchInfo();
  void Reorder1f1bOverlap();
  void ReorderShardedParam();
  void Add1f1bAttr(const BorderPair &recv, const std::string &tag, size_t index_1f1b);
  std::pair<Border, Border> SeqpipeBorder(const std::vector<Border> &borders, int64_t seq_chunk, int64_t chunk,
                                          int64_t micro);
  std::unordered_map<std::string, std::vector<Border>> clean_mask_cache_assigns_;
  std::unordered_map<std::string, std::vector<Border>> clean_seq_chunk_assigns_;
  std::vector<std::vector<std::vector<BorderPair>>> sorted_fwd_begin_;
  std::vector<std::vector<std::vector<BorderPair>>> sorted_fwd_end_;
  std::vector<std::vector<std::vector<BorderPair>>> sorted_fwd_cell_;
  std::vector<std::vector<std::vector<BorderPair>>> sorted_bwd_begin_;
  std::vector<std::vector<std::vector<BorderPair>>> sorted_bwd_end_;
  std::vector<std::vector<std::vector<BorderPair>>> sorted_bwd_cell_;
  std::vector<Triplet> execute_order_;
  std::vector<Triplet> fp_execute_order_;
  std::vector<Triplet> bp_execute_order_;
  std::vector<size_t> cycle_list_;
  size_t warm_up_size_ = 0;
  size_t calm_down_index_ = 0;
  size_t last_stage_pre_fetch_index_ = 0;
  size_t last_stage_pre_fetch_bp_index_ = 0;
  size_t last_stage_pre_fetched_bp_index_ = 0;
  size_t fp_block_size_ = 0;
  std::vector<size_t> advanced_recv_indexs_;
  int64_t seq_chunk_size_ = 1;
  int64_t bias_ = 1;
  bool small_micro_handle_stage_ = false;
  bool before_small_micro_handle_stage_ = false;
  std::vector<std::unordered_map<SchedulerNode, size_t, SchedulerNodeHash>> scheduler_node_order_;
  void InsertSchedulerNode(SchedulerNode prior_node, SchedulerNode next_node, size_t index);
  bool bp_fp_inline_ = false;
  std::unordered_map<size_t, BorderPair> recv_nodes_map_;
};

class SeqvppScheduler : public SeqpipeScheduler {
 public:
  SeqvppScheduler(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root, int64_t stage, int64_t stage_num)
      : SeqpipeScheduler(manager, root, stage, stage_num) {}
  virtual ~SeqvppScheduler() = default;

 protected:
  BorderPair GetBorderNodeRecv(size_t index) override;
  void ComputeCycleList() override;
};

class SeqsmartvppScheduler : public SeqvppScheduler {
 public:
  SeqsmartvppScheduler(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root, int64_t stage, int64_t stage_num)
      : SeqvppScheduler(manager, root, stage, stage_num) {}
  virtual ~SeqsmartvppScheduler() = default;

 protected:
  BorderPair GetBorderNodeRecv(size_t index) override;
  void ControlSendRecvOrder(const BorderPair &send, const BorderPair &post_recv, size_t index) override;
  void ComputeBias() override;
  std::vector<size_t> special_recv_indexs_;
  void ControlSpecialPreRecv(size_t index);
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_SEQPIPE_SCHEDULER_H_
