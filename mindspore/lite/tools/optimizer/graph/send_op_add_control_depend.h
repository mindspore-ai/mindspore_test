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

#ifndef MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SEND_OP_ADD_CONTROL_DEPEND_H_
#define MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SEND_OP_ADD_CONTROL_DEPEND_H_

#include <memory>
#include <string>
#include <unordered_map>
#include "include/common/utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/common/multiple_pattern_process_pass.h"
#include "tools/optimizer/common/pattern_process_pass_extends.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"

namespace mindspore {
namespace opt {
class SendOpAddControlDepend : public opt::LitePatternProcessPass {
 public:
  explicit SendOpAddControlDepend(bool multigraph = true)
      : opt::LitePatternProcessPass("send_op_add_control_depend", multigraph) {}
  ~SendOpAddControlDepend() override = default;

  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const override;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_LITE_TOOLS_OPTIMIZER_GRAPH_SEND_OP_ADD_CONTROL_DEPEND_H_
