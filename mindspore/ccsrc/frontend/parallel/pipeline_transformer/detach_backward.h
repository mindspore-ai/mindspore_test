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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_DETACH_BACKWARD_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_DETACH_BACKWARD_H_

#include <set>
#include <vector>
#include <utility>
#include "ir/manager.h"

namespace mindspore {
namespace parallel {
typedef struct PPInfoStruct {
  int64_t chunk;
  int64_t micro;
} PPInfo;

class DetachBackward {
 public:
  DetachBackward(const FuncGraphManagerPtr &manager, const FuncGraphPtr &root, int64_t stage)
      : manager_(manager), root_(root), stage_(stage) {}
  virtual ~DetachBackward() = default;
  void Init();
  void Run();

 private:
  bool IsNeedDetach(int64_t chunk, int64_t micro);
  void GetChunkNumMicroSize();
  void AdapteDwOverlap(const FuncGraphPtr &fg);
  size_t HandleMonadNode(const FuncGraphPtr &dx_fg, const FuncGraphPtr &dw_fg, size_t partial_dw_size,
                         std::vector<size_t> *dw_index);
  std::vector<size_t> DetachDxAndDwGraph(const FuncGraphPtr &fg, bool is_dw_fg, const CNodePtr &partial_cnode,
                                         std::vector<AnfNodePtr> *new_partial_inputs);
  void HandleClosureGraph(const FuncGraphPtr &fg);
  CNodePtr CreateDwCallNode(const NodeUsersMap &node_users_map, const CNodePtr &dx_call_node,
                            const CNodePtr &closure_call_node, std::vector<size_t> dw_index);
  void HandleDataDependency(const std::vector<size_t> &dw_index, const FuncGraphPtr &fg, size_t num_diff);
  std::vector<size_t> HandleBwdGraphOutputs(
    const std::pair<std::vector<AnfNodePtr>, std::vector<AnfNodePtr> > &out_inputs, bool is_dw_fg,
    const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &parameters, size_t num_diff);
  AnfNodePtr CreateTupleGetItem(const FuncGraphPtr &fg, const AnfNodePtr &node, int64_t index);
  AnfNodePtr CreateMakeTuple(const FuncGraphPtr &fg, const std::vector<AnfNodePtr> &inputs);
  FuncGraphManagerPtr manager_;
  FuncGraphPtr root_;
  int64_t micro_size_ = 1;
  int64_t chunk_num_ = 1;
  int64_t stage_;
  std::set<FuncGraphPtr> closure_graphs_;
  std::vector<PPInfo> need_detach_info_;
};

std::vector<PPInfo> InferNeedDetachInfo(int64_t stage, int64_t micro_size);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PIPELINE_TRANSFORMER_DETACH_BACKWARD_H_
