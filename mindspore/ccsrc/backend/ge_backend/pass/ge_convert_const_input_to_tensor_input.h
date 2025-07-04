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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_GE_CONVERT_CONST_INPUT_TO_TENSOR_INPUT_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_GE_CONVERT_CONST_INPUT_TO_TENSOR_INPUT_H_
#include <string>

#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BACKEND_EXPORT GEConvertConstInputToTensorInput : public PatternProcessPass {
 public:
  explicit GEConvertConstInputToTensorInput(bool multigraph = true)
      : PatternProcessPass("convert_const_input_to_tensor_input", multigraph) {}
  ~GEConvertConstInputToTensorInput() override = default;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
};

}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_PASS_GE_CONVERT_CONST_INPUT_TO_TENSOR_INPUT_H_
