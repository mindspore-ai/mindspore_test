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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_PASS_GRAPH_VIEW_REPALCE_PASS_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_PASS_GRAPH_VIEW_REPALCE_PASS_H_

#include <string>
#include <vector>
#include "include/backend/optimizer/pass.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT GraphViewReplacePass : public Pass {
 public:
  explicit GraphViewReplacePass(const std::string &name = "graph_view_replace") : Pass(name) {}
  ~GraphViewReplacePass() override = default;
  bool Run(const FuncGraphPtr &func_graph) override;

 private:
  std::vector<std::string> out_kernels_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_PASS_GRAPH_VIEW_REPALCE_PASS_H_
