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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CHECK_INVALID_VIEW_INPLACE_DOUT_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CHECK_INVALID_VIEW_INPLACE_DOUT_H_
#include <vector>
#include "include/common/utils/utils.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
namespace irpass {

constexpr auto kCheckDoutLevelSceneOne = "1";
constexpr auto kCheckDoutLevelSceneTwo = "2";
constexpr auto kInvalidInplaceDout = "invalid_inplace_dout";
constexpr auto kFlagNeedCheckViewInplaceDoutBprop = "need_check_view_inplace_dout_bprop";

class CheckInvalidViewInplaceDout {
 public:
  CheckInvalidViewInplaceDout() = default;
  virtual ~CheckInvalidViewInplaceDout() = default;
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer);
};

void CheckBpropGraphHasInvalidDoutHelper(const FuncGraphPtr &func_graph, const std::vector<bool> &need_grads);
void MarkInvalidInplaceOpDout(const FuncGraphPtr &fprop_graph);
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_CHECK_INVALID_VIEW_INPLACE_DOUT_H_
