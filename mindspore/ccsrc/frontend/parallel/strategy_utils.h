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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_UTILS_H_

#include <string>

#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/ops_info/ops_utils.h"

namespace mindspore {
namespace parallel {
class StrategyUtils {
 public:
  static bool CheckExtractInformation(const CNodePtr &cnode);
  static void SetVirtualDatasetStrategy(const CNodePtr &node);
  static void SetDatasetLayout(const CNodePtr &node, const std::string &attrName);
  static void SetGetNextLayout(const CNodePtr &node);
  static void ExtractStrategyAndInit(const CNodePtr &cnode, const PrimitivePtr &prim, const OperatorInfoPtr &op_info);
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STRATEGY_UTILS_H_
