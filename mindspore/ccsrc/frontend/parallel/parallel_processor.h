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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_PROCESSOR_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_PROCESSOR_H_

#include <vector>
#include <memory>
#include <utility>

#include "ir/anf.h"
#include "frontend/parallel/parallel_processor_context.h"
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
class ParallelProcessor {
 public:
  explicit ParallelProcessor(const ParallelProcessorContextPtr &context) : processor_context_(context) {
    MS_EXCEPTION_IF_NULL(processor_context_);
  }

  void Process();

  static void ForwardCommunication(const OperatorVector &forward_op, const ForwardOpList &forward_op_list,
                                   const CNodePtr &node);
  static void InsertForwardOps(const OperatorInfoPtr &distribute_operator, const CNodePtr &cnode);
  static TensorLayout GetTensorInLayout(const AnfNodePtr &pre_node, std::vector<int> get_item_index);
  static void Redistribution(const std::pair<AnfNodePtr, PosPair> &node_pair, const AnfNodePtr &pre_node,
                             const std::vector<int> &get_item_index, std::pair<AnfNodePtr, int> *new_next_node);
  static void StepRedistribution(const CNodePtr &cnode, const NodeUsersMap &node_users_map);

 private:
  const ParallelProcessorContextPtr &processor_context_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARALLEL_PROCESSOR_H_
