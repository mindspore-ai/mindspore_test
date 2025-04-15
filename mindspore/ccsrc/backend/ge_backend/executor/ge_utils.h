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
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include "backend/ge_backend/graph_ir/types.h"
#include "backend/ge_backend/executor/ge_device_res_manager.h"
#include "plugin/res_manager/ascend/op_adapter/op_adapter_base.h"
#include "acl/acl_rt.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
using mindspore::backend::ge_backend::OptionMap;

std::string GetGraphName(const FuncGraphPtr &graph);
// session options
void GetGeSessionOptions(backend::ge_backend::SessionOptions *options);
// global options, for GeInitialize
void GetGeGlobalOptions(std::map<std::string, std::string> *ge_options);
// ge options from user setting
void SetPassthroughGeOptions(std::string option_level, OptionMap *options);
bool AddDFGraph(const FuncGraphPtr &anf_graph, const backend::ge_backend::TensorOrderMap &init_inputs_map,
                bool export_air);
bool AddFakeGraph(const FuncGraphPtr &anf_graph);
bool IsGeTrain();
void SavePrevStepWeight(const std::vector<AnfNodePtr> &weights, aclrtStream stream);
void SaveCopyWeight(const std::vector<tensor::TensorPtr> &copy_weights, const std::vector<AnfNodePtr> &weights,
                    aclrtStream stream);
void StorageWeights(std::vector<tensor::TensorPtr> *copy_weights, const std::vector<AnfNodePtr> &weights,
                    const std::shared_ptr<GeDeviceResManager> &ge_res_manager, bool *first_save);
size_t GetFreeMemoryInfo();
void SplitWeightsByFreeMemory(const std::vector<AnfNodePtr> &root_weights, std::vector<AnfNodePtr> *prev_part,
                              std::vector<AnfNodePtr> *copy_part, const size_t &free_mem_size_for_save);
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
