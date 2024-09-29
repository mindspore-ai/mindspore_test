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

#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_PASS_SILENT_CHECK_V2_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_PASS_SILENT_CHECK_V2_H_

#include <vector>
#include <string>
#include <map>
#include <set>
#include "base/base.h"
#include "include/backend/optimizer/pass.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "utils/log_adapter.h"
#include "pipeline/jit/ps/resource.h"

namespace mindspore {
namespace pipeline {
const char kSilentCheckV2[] = "silent_check_v2";
bool IsNpuAsdEnable();

class SilentCheckV2 {
 public:
  explicit SilentCheckV2(const FuncGraphPtr &root) : root_(root) { GetLossScale(); }
  ~SilentCheckV2() = default;

  bool HasFloat16Input();
  void GetLastGradNode();
  bool Run(const FuncGraphPtr &func_graph);
  void UpdateNodes();

 private:
  void GetLossScale();
  AnfNodePtr FindGetNextNode();
  AnfNodePtrList &GetRootGraphTopoNodes();
  CNodePtr GetLastGradNode(const FuncGraphPtr &func_graph, const AnfNodePtr &start_node);
  AnfNodePtr CreateSlientCheckNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node);

  // root graph
  FuncGraphPtr root_ = nullptr;
  // buffering topo nodes of root graph
  AnfNodePtrList root_graph_nodes_;
  // pointer to loss_scale of the whole network if exists
  ParameterPtr loss_scale_ = nullptr;
  // last node in backward grad which needs inserting SilentCheck operator
  CNodePtr last_grad_node_ = nullptr;

  // pointer to parameter `scale_sense` of graph being processed, create it if not exist
  ParameterPtr scale_sense_ = nullptr;
  // map for recoreding user cnodes of each graph
  std::map<FuncGraphPtr, std::set<CNodePtr>> graph_users_;
  // recording graphs which were added parameter `scale_sense`
  std::set<FuncGraphPtr> add_param_graphs_;
};

bool SilentCheckPass(const ResourcePtr &resource);
}  // namespace pipeline
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_PASS_SILENT_CHECK_V2_H_
