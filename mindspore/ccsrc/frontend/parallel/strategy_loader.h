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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_LOADER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_LOADER_H_

#include <vector>
#include "base/base.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/ops_info/ops_utils.h"

namespace mindspore {
namespace parallel {
class StrategyLoader {
 public:
  static Status LoadStrategyFromFile(const std::vector<AnfNodePtr> &all_nodes);
  static void SaveStrategyToFile(const std::vector<AnfNodePtr> &all_nodes);
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_LOADER_H_
