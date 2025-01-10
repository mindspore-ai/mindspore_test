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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_GE_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_GE_UTILS_H_

#include <map>
#include <string>
#include <vector>
#include <unordered_set>
#include "include/transform/graph_ir/types.h"

namespace mindspore {
namespace device {
namespace ascend {
using mindspore::transform::OptionMap;

std::string GetGraphName(const FuncGraphPtr &graph);
// session options
void GetGeSessionOptions(transform::SessionOptions *options);
// global options, for GeInitialize
void GetGeOptions(std::map<std::string, std::string> *ge_options);
// options from set_context
void SetPassthroughGeOptions(bool is_global, OptionMap *options);
bool AddDFGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map, bool export_air);
bool AddFakeGraph(const FuncGraphPtr &anf_graph);
bool IsGeTrain();

class InferNeedUpdateParaNames {
 public:
  std::unordered_set<std::string> &GetInferParameterNames() { return infer_need_update_para_names; }

 private:
  std::unordered_set<std::string> infer_need_update_para_names;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_GE_UTILS_H_
