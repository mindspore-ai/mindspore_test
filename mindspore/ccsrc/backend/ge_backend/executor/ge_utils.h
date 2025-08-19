/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_UTILS_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_UTILS_H_

#include <map>
#include <string>
#include <vector>
#include <unordered_set>
#include "backend/ge_backend/graph_ir/types.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "acl/acl_rt.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
using mindspore::backend::ge_backend::OptionMap;

std::string GetGraphName(const FuncGraphPtr &graph);
// session options
void GetGeSessionOptions(backend::ge_backend::SessionOptions *options);
// ge options from user setting
void SetPassthroughGeOptions(std::string option_level, OptionMap *options);
bool AddDFGraph(const FuncGraphPtr &anf_graph, const backend::ge_backend::TensorOrderMap &init_inputs_map,
                bool export_air);
bool AddFakeGraph(const FuncGraphPtr &anf_graph);
bool IsGeTrain();
void SavePrevStepWeight(const std::vector<AnfNodePtr> &weights, aclrtStream stream);
class InferNeedUpdateParaNames {
 public:
  std::unordered_set<std::string> &GetInferParameterNames() { return infer_need_update_para_names; }

 private:
  std::unordered_set<std::string> infer_need_update_para_names;
};
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_UTILS_H_
