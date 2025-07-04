/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARAMETER_MANAGER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARAMETER_MANAGER_H_

#include <set>
#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <unordered_map>
#include "base/base.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/came_parallel_handler.h"
#include "pipeline/jit/ps/resource.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace parallel {
constexpr size_t THIRD_BORDER_INFO_INDEX = 3;
constexpr size_t SECOND_BORDER_INFO_INDEX = 2;
constexpr char OBJ[] = "obj";
constexpr char CLONED_OBJ[] = "cloned_obj";
constexpr char SLICE_PARAMETER_FN_PATH[] = "mindspore.parallel._utils";
constexpr char SLICE_PARAMETER_FN_NAME[] = "_slice_parameter";
constexpr char INIT_OPTIMIZER_STATE_FN[] = "_init_optimizer_state";
constexpr char SLICE_TENSOR_FN_NAME[] = "_slice_tensor";

using RefKeyPair = std::pair<AnfNodePtr, std::vector<AnfNodePtr>>;
using ParameterUsersInfo = std::pair<std::string, std::pair<AnfNodePtr, AnfNodeIndexSet>>;

ParameterUsersInfo FindParameterUsers(const AnfNodePtr &node, bool (*IsCareNode)(const CNodePtr &),
                                      const std::vector<AnfNodePtr> &all_nodes);
AnfNodePtr RefParameterToActualParameter(const AnfNodePtr &node);
void CheckParameterSplit(const std::vector<AnfNodePtr> &all_nodes);
void HandleSymbolicKeyInstance(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes);
void HandleNoUsedParameter(const FuncGraphPtr &root);
void HandleFullySplitParameters(const FuncGraphPtr &root);
void SetClonedTensorShapeForOptimizer(const FuncGraphPtr &root);
void HandleCameAndAdaFactorOpt(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                               const FuncGraphManagerPtr &manager);
void AutoParallelPostProcess(const FuncGraphPtr &root);
void SliceTensorObj(const ParameterPtr &parameter, const TensorLayoutPtr &tensor_layout, size_t rank_id = 0);
// Init the parameters for graph which not specified by shard under PyNative mode.
void InitPynativeNoShardParams(const FuncGraphPtr &root);
void InitCompileCacheParams(const pipeline::ResourcePtr &resource);
std::pair<AnfNodePtr, bool> FindParameter(const AnfNodePtr &node, const FuncGraphPtr &func_graph);
std::pair<AnfNodePtr, bool> FindParameterWithAllgather(const AnfNodePtr &node, const FuncGraphPtr &func_graph,
                                                       const std::string &name);
std::unordered_map<std::string, std::shared_ptr<TensorLayout>> AdaSumParamTensorLayout(const FuncGraphPtr &root);
bool HandleAdaSum(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                  std::unordered_map<std::string, std::shared_ptr<TensorLayout>> *adasum_param_tensor_layout_map);
void HandleMirrorInAdaSum(
  const FuncGraphPtr &root,
  std::unordered_map<std::string, std::shared_ptr<TensorLayout>> *adasum_param_tensor_layout_map);
void GetSubRootParams(const AnfNodePtrList &root_params, AnfNodePtrList *sub_root_params);
bool ParameterIsCloned(const AnfNodePtr &parameter_node);
bool IsStrategySaved(const AnfNodePtr &parameter_node);
py::object GetPyParameterObj(const ParamInfoPtr &param_info, const std::string &obj);
bool IsFullySplitParameter(const ParameterPtr &param_ptr, size_t allow_repeat_num = 1);
std::shared_ptr<TensorLayout> CreateParameterLayout(const AnfNodePtr &node);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PARAMETER_MANAGER_H_
