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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_MOVE_TO_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_MOVE_TO_UTILS_H_
#include "include/backend/kernel_graph.h"

namespace mindspore {
namespace opt {
struct MoveToInfo {
  const char *to_;
  AnfNodePtr data_previous_node_;
  CNodePtr data_following_node_;
  size_t input_index_;
  CNodePtr control_previous_node_;
  CNodePtr control_following_node_;
};

struct MoveAssignInfo {
  const char *to_;
  ParameterPtr parameter_;
  AnfNodePtr value_;
  CNodePtr control_previous_node_;
  CNodePtr control_following_node_;
};

class MoveToUtils {
 public:
  static CNodePtr InsertMoveTo(const KernelGraphPtr &kernel_graph, const MoveToInfo &info);
  static CNodePtr InsertMoveAssign(const KernelGraphPtr &kernel_graph, const MoveAssignInfo &info);
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_MOVE_TO_UTILS_H_
