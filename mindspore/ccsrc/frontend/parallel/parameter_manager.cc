/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/parameter_manager.h"

#include <cinttypes>
#include <algorithm>

#include <map>
#include <memory>
#include <unordered_set>
#include <string>
#include <utility>
#include <deque>
#include <functional>

#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "utils/hash_map.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/get_parallel_info.h"
#include "frontend/parallel/graph_util/pipeline_split_utils.h"
#include "frontend/parallel/graph_util/parallel_tensordump.h"
#include "frontend/parallel/node_check.h"
#include "ir/tensor.h"
#include "include/common/utils/comm_manager.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "pipeline/jit/ps/executor/graph_executor_py.h"
#include "pipeline/jit/ps/pipeline.h"
#include "frontend/parallel/parallel_node_check.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "include/common/utils/tensor_py.h"
#include "include/common/utils/tensor_py_wrapper.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace parallel {
using TensorLayoutPtr = std::shared_ptr<TensorLayout>;
static ParameterUsersInfo FindRefKeyNodeUsers(const RefKeyPair &ref_key_pair, bool (*IsCareNode)(const CNodePtr &)) {
  // Dealing with the RefKey case
  ParameterUsersInfo parameter_user_info;
  auto refkeys = ref_key_pair.second;
  auto cnode = ref_key_pair.first;

  auto cnode_ptr = cnode->cast<CNodePtr>();
  if ((cnode_ptr == nullptr) || !IsValueNode<Primitive>(cnode_ptr->input(0)) || !IsCareNode(cnode_ptr)) {
    return parameter_user_info;
  }

  if (refkeys.size() > 1) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "CNode: " << cnode->fullname_with_scope()
                                       << "'s inputs have more than 1 RefKeys";
  }
  MS_EXCEPTION_IF_NULL(cnode->func_graph());
  auto cnode_func_graph = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(cnode->func_graph()->manager());

  // Find the RefKey being used
  auto candidate_set_by_refkey = cnode_func_graph->manager()->node_users()[refkeys[0]];
  for (auto &candidate : candidate_set_by_refkey) {
    auto candidate_node = candidate.first;
    auto c = candidate_node->cast<CNodePtr>();
    if ((c == nullptr) || !IsValueNode<Primitive>(c->input(0)) || !IsCareNode(c)) {
      continue;
    }
    parameter_user_info.second.second.insert(candidate);
  }

  // Find the corresponding Parameter being used
  std::vector<AnfNodePtr> parameters = FindParameterByRefKeyNode(refkeys[0], cnode_func_graph);
  if (parameters.size() != 1) {
    MS_LOG(EXCEPTION) << "Find parameter by ref key node failed";
  }
  MS_EXCEPTION_IF_NULL(parameters[0]);
  auto para_ptr = parameters[0]->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(para_ptr);
  parameter_user_info.first = para_ptr->name();
  parameter_user_info.second.first = parameters[0];
  MS_EXCEPTION_IF_NULL(cnode_func_graph);
  auto candidate_set_by_para = cnode_func_graph->manager()->node_users()[parameters[0]];
  for (auto &candidate : candidate_set_by_para) {
    auto candidate_node = candidate.first;
    auto c = candidate_node->cast<CNodePtr>();
    if ((c == nullptr) || !IsValueNode<Primitive>(c->input(0)) || !IsCareNode(c)) {
      continue;
    }
    parameter_user_info.second.second.insert(candidate);
  }
  return parameter_user_info;
}
// In this case, node is a Parameter
static ParameterUsersInfo FindParameterNodeUsers(const AnfNodePtr &node, const std::vector<AnfNodePtr> &all_nodes) {
  ParameterUsersInfo parameter_user_info;
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->func_graph());
  MS_EXCEPTION_IF_NULL(node->func_graph()->manager());
  auto candidate_set = node->func_graph()->manager()->node_users()[node];
  for (auto &candidate : candidate_set) {
    auto candidate_node = candidate.first;
    if (!IsPrimitiveCNode(candidate_node, prim::kPrimLoad)) {
      auto c = candidate_node->cast<CNodePtr>();
      if (c == nullptr || !c->has_user_data<OperatorInfo>() || IsSomePrimitive(c, RECEIVE)) {
        continue;
      }
      parameter_user_info.second.second.insert(candidate);
      continue;
    }
    if (candidate.second != 1) {
      continue;
    }
    auto &node_user_map = node->func_graph()->manager()->node_users();
    auto load_node_users = node_user_map[candidate_node];
    for (auto &node_user : load_node_users) {
      auto cnode = node_user.first->cast<CNodePtr>();
      if (cnode == nullptr) {
        continue;
      }
      std::pair<AnfNodePtr, int> child_parallel_care_node;
      if (IsSomePrimitive(cnode, UPDATESTATE) || !cnode->in_forward_flag()) {
        continue;
      }
      if (!IsSomePrimitive(cnode, MAKE_TUPLE) && (IsParallelCareNode(cnode) || IsAutoParallelCareNode(cnode))) {
        child_parallel_care_node = node_user;
      } else {
        child_parallel_care_node = BFSParallelCareNode(cnode, node_user_map, node_user.second, all_nodes);
      }
      if (child_parallel_care_node.first) {
        cnode = child_parallel_care_node.first->cast<CNodePtr>();
      } else {
        continue;
      }
      if (cnode == nullptr || !cnode->has_user_data<OperatorInfo>() || IsSomePrimitive(cnode, RECEIVE)) {
        continue;
      }
      parameter_user_info.second.second.insert(child_parallel_care_node);
    }
  }
  MS_EXCEPTION_IF_NULL(node);
  auto node_pra_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(node_pra_ptr);
  parameter_user_info.first = node_pra_ptr->name();
  parameter_user_info.second.first = node;
  return parameter_user_info;
}

static RefKeyPair CNodeWithRefKeys(const AnfNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  std::vector<AnfNodePtr> refkeys;
  if (cnode->isa<CNode>()) {
    auto cnode_ptr = cnode->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode_ptr);
    auto inputs = cnode_ptr->inputs();
    for (auto &one_input : inputs) {
      if (IsValueNode<RefKey>(one_input)) {
        refkeys.push_back(one_input);
      }
    }
    if (refkeys.size() >= 1) {
      return std::make_pair(cnode, refkeys);
    }
  }
  return {nullptr, refkeys};
}

ParameterUsersInfo FindParameterUsers(const AnfNodePtr &node, bool (*IsCareNode)(const CNodePtr &),
                                      const std::vector<AnfNodePtr> &all_nodes) {
  ParameterUsersInfo parameter_users_info;

  auto cnode_with_refkeys = CNodeWithRefKeys(node);
  if (cnode_with_refkeys.first != nullptr) {
    // the node is a ref key node
    return FindRefKeyNodeUsers(cnode_with_refkeys, IsCareNode);
  } else if (node->isa<Parameter>()) {
    auto param_ptr = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    // the node is a parameter node
    if (param_ptr->has_default()) {
      return FindParameterNodeUsers(node, all_nodes);
    }
  }

  return parameter_users_info;
}

static bool IsUsedParameter(const FuncGraphPtr &graph, const AnfNodePtr &parameter, size_t max_depth) {
  if (max_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(EXCEPTION) << "Recursive call is larger than 100000.";
  }
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(parameter);
  auto manager = graph->manager();
  auto node_users = manager->node_users()[parameter];
  if (node_users.empty()) {
    return false;
  }
  for (auto node_user : node_users) {
    auto use_node = node_user.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(use_node);
    if (IsValueNode<FuncGraph>(use_node->input(0))) {
      auto graph_sub = GetValueNode<FuncGraphPtr>(use_node->input(0));
      MS_EXCEPTION_IF_NULL(graph_sub);
      auto parameters = graph_sub->parameters();
      auto parameter_sub = parameters[IntToSize(node_user.second - 1)];
      return IsUsedParameter(graph_sub, parameter_sub, max_depth + 1);
    }
    if (use_node->input(0)->isa<CNode>()) {
      auto cnode = use_node->input(0)->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (!IsSomePrimitive(cnode, J) || !IsValueNode<FuncGraph>(cnode->input(1))) {
        return true;
      }
      auto graph_sub = GetValueNode<FuncGraphPtr>(cnode->input(1));
      MS_EXCEPTION_IF_NULL(graph_sub);
      auto parameters = graph_sub->parameters();
      auto parameter_sub = parameters[IntToSize(node_user.second - 1)];
      return IsUsedParameter(graph_sub, parameter_sub, max_depth + 1);
    }
    return true;
  }
  return true;
}

void CheckParameterSplit(const std::vector<AnfNodePtr> &all_nodes) {
  for (auto &node : all_nodes) {
    ParameterUsersInfo parameter_users_info = FindParameterUsers(node, IsParallelCareNode, all_nodes);
    auto &users_set = parameter_users_info.second.second;
    if (users_set.size() <= 1) {
      continue;
    }

    auto parameter_name = parameter_users_info.first;
    MS_LOG(INFO) << "The parameter: " << parameter_name << " has " << users_set.size() << " users";
    auto &first_user = users_set.front();
    auto parameter_tensor_info = GetInputsTensorInfo(first_user);

    for (auto iter = users_set.begin() + 1; iter != users_set.end(); ++iter) {
      auto &user = *iter;
      auto user_tensor_info = GetInputsTensorInfo(user);
      if (IsSameTensorInfo(parameter_tensor_info, user_tensor_info)) {
        continue;
      } else {
        MS_LOG(EXCEPTION) << "The parameter: " << parameter_name
                          << " has multiple users, but the TensorInfo are different";
      }
    }
  }
}

namespace {
void RevertSymbolicKeyInstance(const FuncGraphPtr &root, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(node);
  auto symbolic_key = GetValueNode<SymbolicKeyInstancePtr>(node);
  MS_EXCEPTION_IF_NULL(symbolic_key);
  auto all_upstream_node = root->manager()->node_users()[node];
  for (auto &upstream_node : all_upstream_node) {
    FuncGraphPtr fg = upstream_node.first->func_graph();
    if (symbolic_key->node()->isa<Parameter>()) {
      for (auto &param : root->parameters()) {
        if (*param == *symbolic_key->node()) {
          AnfNodePtr reverted_node = root->NewCNode({NewValueNode(prim::kPrimEmbed), param});
          MS_EXCEPTION_IF_NULL(reverted_node);
          MS_LOG(DEBUG) << "before replace " << node->ToString() << " to node " << reverted_node->DebugString();
          (void)fg->manager()->Replace(node, reverted_node);
          MS_LOG(DEBUG) << "revert node " << node->ToString() << " to node " << reverted_node->DebugString();
        }
      }
    }
  }
}
}  // namespace

void HandleSymbolicKeyInstance(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes) {
  MS_EXCEPTION_IF_NULL(root);
  for (auto &node : all_nodes) {
    // revert back SymbolicKeyInstance to embed() primitive
    if (IsValueNode<SymbolicKeyInstance>(node)) {
      RevertSymbolicKeyInstance(root, node);
      continue;
    }
  }
}

bool IsStrategySaved(const AnfNodePtr &parameter_node) {
  MS_EXCEPTION_IF_NULL(parameter_node);
  auto cloned_parameter = parameter_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(cloned_parameter);

  // find the clone parameter
  if (!cloned_parameter->has_default()) {
    return false;
  }
  auto param_value = cloned_parameter->param_info();
  if (param_value == nullptr) {
    return false;
  }
  return param_value->strategy_ckpt_saved();
}

bool ParameterIsCloned(const AnfNodePtr &parameter_node) {
  MS_EXCEPTION_IF_NULL(parameter_node);
  auto cloned_parameter = parameter_node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(cloned_parameter);

  // find the clone parameter
  if (!cloned_parameter->has_default()) {
    return false;
  }
  auto param_value = cloned_parameter->param_info();
  if (param_value == nullptr) {
    return false;
  }
  bool cloned = param_value->cloned();
  if (!cloned) {
    return false;
  }

  MS_LOG(INFO) << "The parameter: " << cloned_parameter->name() << " is cloned";
  return true;
}

void HandleNoUsedParameter(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  bool full_batch = ParallelContext::GetInstance()->full_batch();
  if (full_batch) {
    return;
  }

  auto dev_num = g_device_manager->stage_device_num();
  auto parameters = root->parameters();
  if (parameters.empty()) {
    MS_LOG(INFO) << "Parameters is not in graph, thus no need to set parallel shape";
  } else {
    for (auto &parameter : parameters) {
      if (IsUsedParameter(root, parameter, 0)) {
        continue;
      }
      auto parameter_shape = GetNodeShape(parameter);
      if (parameter_shape.empty()) {
        continue;
      }
      Shape slice_shape = parameter_shape[0];
      if (slice_shape.empty() || slice_shape[0] < dev_num) {
        continue;
      }
      slice_shape[0] = slice_shape[0] / dev_num;
      auto slice_shape_ptr = std::make_shared<abstract::Shape>(slice_shape);
      auto abstract = parameter->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      auto abstract_cloned = abstract->Clone();
      MS_EXCEPTION_IF_NULL(abstract_cloned);
      abstract_cloned->set_shape(slice_shape_ptr);
      parameter->set_abstract(abstract_cloned);
    }
  }
}

bool IsFullySplitParameter(const ParameterPtr &param_ptr, size_t allow_repeat_num) {
  auto tensor_layout = param_ptr->user_data<parallel::TensorLayout>();
  if (tensor_layout == nullptr) {
    return false;
  }

  auto dev_mat_shape = tensor_layout->device_arrangement().array();
  auto tensor_map = tensor_layout->tensor_map().array();
  int64_t rank = g_device_manager->global_rank();
  RankList rank_list = g_device_manager->GetDeviceListInThisStage();
  DeviceMatrix dev_matrix(rank, rank_list, dev_mat_shape);
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(tensor_map, &group_devices) != SUCCESS) {
    MS_LOG(WARNING) << "Get devices by tensor map failed, invalid tensor layout";
    return false;
  }

  if (group_devices.size() <= allow_repeat_num) {
    MS_LOG(INFO) << "The parameter: " << param_ptr->name() << " is fully split";
    return true;
  }
  return false;
}

py::object GetPyParameterObj(const ParamInfoPtr &param_info, const std::string &obj) {
  py::object py_obj = py::cast(param_info);
  if (py::isinstance<py::none>(py_obj)) {
    return py::none();
  }
  return python_adapter::GetPyObjAttr(py_obj, obj);
}

static bool IsAccuGradObj(const py::object &py_obj) {
  auto name = python_adapter::GetPyObjAttr(py_obj, PARAM_NAME);
  if (py::isinstance<py::none>(name)) {
    return false;
  }
  if (py::cast<std::string>(name).find(ACCU_GRADS) == 0) {
    return true;
  }
  return false;
}

void SliceParameterObj(const ParameterPtr &parameter, const TensorLayoutPtr &tensor_layout) {
  auto param_info = parameter->param_info();
  if (param_info == nullptr) {
    MS_LOG(WARNING) << "parameter: " << parameter->DebugString() << " doesn't have param_info.";
    return;
  }
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  auto phase = graph_executor->phase();
  auto py_obj = GetPyParameterObj(param_info, OBJ);
  if (py::isinstance<py::none>(py_obj)) {
    MS_LOG(WARNING) << "Parameter: " << parameter->DebugString() << " can't find python obj.";
    return;
  }
  if (tensor_layout == nullptr) {
    (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, py_obj, py::str(phase),
                                   py::none());
    return;
  }
  // create python layout obj
  const auto &device_arrangement = tensor_layout->device_arrangement().array();
  const auto &tensor_map = tensor_layout->tensor_map().array();
  auto slice_shape = tensor_layout->base_slice_shape().array();
  int64_t field_size = tensor_layout->get_field_size();
  bool uniform_split = tensor_layout->uniform_split();
  std::string opt_shard_group = tensor_layout->opt_shard_group();
  if (!opt_shard_group.empty()) {
    slice_shape = tensor_layout->opt_shard_slice_shape();
  }
  auto full_shape = tensor_layout->tensor_shape().array();
  py::tuple layout =
    py::make_tuple(device_arrangement, tensor_map, slice_shape, field_size, uniform_split, opt_shard_group, full_shape);

  // Call Python _slice_parameter Fn to slice python parameter obj
  (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, py_obj, py::str(phase), layout);

  // handle cloned parameter, like accu_grad and optimizer param
  auto grad_accumulation_shard =
    ParallelContext::GetInstance()->grad_accumulation_shard() || ParallelContext::GetInstance()->zero3();
  auto cloned_py_obj = GetPyParameterObj(param_info, CLONED_OBJ);
  if (!py::isinstance<py::none>(cloned_py_obj)) {
    if (!py::isinstance<py::list>(cloned_py_obj)) {
      MS_LOG_WITH_NODE(EXCEPTION, parameter)
        << "parameter: " << parameter->DebugString() << " doesn't have correct cloned obj";
    }
    auto obj_list = py::cast<py::list>(cloned_py_obj);
    for (size_t i = 0; i < obj_list.size(); ++i) {
      py::object each_cloned_obj = obj_list[i];
      auto cloned_param_slice_shape = tensor_layout->base_slice_shape().array();
      if (!opt_shard_group.empty()) {
        if (!IsAccuGradObj(each_cloned_obj) || grad_accumulation_shard) {
          cloned_param_slice_shape = tensor_layout->opt_shard_slice_shape();
        }
        if (IsAccuGradObj(each_cloned_obj) && !grad_accumulation_shard) {
          // clear to avoid further accu grad parameter slicing for opt shard if grad_accumulation_shard is false
          opt_shard_group.clear();
        }
      }
      py::tuple cloned_param_layout = py::make_tuple(device_arrangement, tensor_map, cloned_param_slice_shape,
                                                     field_size, uniform_split, opt_shard_group, full_shape);
      (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, each_cloned_obj, py::str(phase),
                                     cloned_param_layout);
    }
  }
}

void SliceTensorObj(const ParameterPtr &parameter, const TensorLayoutPtr &tensor_layout, size_t rank_id) {
  auto param = parameter->default_param();
  MS_EXCEPTION_IF_NULL(param);
  auto p_tensor = param->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(p_tensor);
  if (p_tensor->DataSize() == 1) {
    MS_LOG(INFO) << "The parameter's data size is 1, no need to layout.";
    return;
  }
  if (tensor_layout == nullptr) {
    MS_LOG(INFO) << "No need to layout parameter";
    return;
  }
  // start get layout info
  const auto &device_arrangement = tensor_layout->device_arrangement().array();
  for (auto i : device_arrangement) std::cout << i << ' ';
  const auto &tensor_map = tensor_layout->tensor_map().array();
  auto slice_shape = tensor_layout->slice_shape().array();
  int64_t field_size = tensor_layout->get_field_size();
  bool uniform_split = tensor_layout->uniform_split();
  if (uniform_split == 0) {
    MS_LOG(ERROR) << "The load tensor only support uniform split now.";
  }
  std::string opt_shard_group = tensor_layout->opt_shard_group();
  if (!opt_shard_group.empty()) {
    slice_shape = tensor_layout->opt_shard_slice_shape();
  }
  py::tuple layout =
    py::make_tuple(device_arrangement, tensor_map, slice_shape, field_size, uniform_split, opt_shard_group);

  MS_LOG(INFO) << "origin p_tensor:" << p_tensor->name() << p_tensor->Size() << p_tensor->shape();
  auto tensor_py = python_adapter::CastToPyObj(p_tensor);
  // Call Python _slice_tensor Fn to slice python tensor obj
  auto new_tensor_py =
    python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_TENSOR_FN_NAME, tensor_py, layout, rank_id);
  MS_LOG(INFO) << "Success Call Python _slice_parameter Fn to slice python parameter obj";
  auto new_tensor = tensor::ConvertToTensor(new_tensor_py);
  MS_EXCEPTION_IF_NULL(new_tensor);
  MS_LOG(INFO) << "new p_tensor:" << new_tensor->name() << new_tensor->Size() << new_tensor->shape();
  parameter->set_default_param(tensor::ConvertToTensorPyWrapper(new_tensor_py));
}

static void SliceCacheParameterObj(const ParameterPtr &parameter, const py::dict &layout_dict) {
  auto param_info = parameter->param_info();
  constexpr int64_t SLICE_SHAPE_INDEX = 12;
  constexpr size_t MIN_LENGTH = 13;
  if (param_info == nullptr) {
    MS_LOG(WARNING) << "parameter: " << parameter->DebugString() << " doesn't have param_info.";
    return;
  }
  auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
  MS_EXCEPTION_IF_NULL(graph_executor);
  auto phase = graph_executor->phase();
  auto py_obj = GetPyParameterObj(param_info, OBJ);
  if (py::isinstance<py::none>(py_obj)) {
    MS_LOG(WARNING) << "Parameter: " << parameter->DebugString() << " can't find python obj.";
    return;
  }
  auto name = parameter->name();
  if (!layout_dict.contains(name)) {
    (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, INIT_OPTIMIZER_STATE_FN, py_obj, py::str(phase));
    return;
  }
  auto layout = layout_dict[py::str(name)];
  if (py::len(layout) < MIN_LENGTH) {
    MS_LOG(WARNING) << "The length of layout must be larger than 13, but got " << py::len(layout)
                    << ". Parameter:" << parameter->DebugString();
    return;
  }
  const auto &device_arrangement = layout[py::cast<size_t>(0)];
  const auto &tensor_map = layout[py::cast<size_t>(1)];
  auto slice_shape = layout[py::cast<size_t>(2)];
  auto field_size = layout[py::cast<size_t>(3)];
  auto uniform_split = layout[py::cast<size_t>(4)];
  auto opt_shard_group = layout[py::cast<size_t>(5)];
  auto full_shape = layout[py::cast<size_t>(6)];
  auto grad_accumulation_shard = ParallelContext::GetInstance()->grad_accumulation_shard();
  if (!opt_shard_group.cast<std::string>().empty()) {
    slice_shape = layout[py::cast(SLICE_SHAPE_INDEX)];
  }
  py::tuple param_layout =
    py::make_tuple(device_arrangement, tensor_map, slice_shape, field_size, uniform_split, opt_shard_group, full_shape);

  // param init in parallel mode
  param_info->set_is_param_init(true);

  // Call Python _slice_parameter Fn to slice python parameter obj
  (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, py_obj, py::str(phase),
                                 param_layout);

  // handle cloned parameter, like accu_grad and optimizer param
  auto cloned_py_obj = GetPyParameterObj(param_info, CLONED_OBJ);
  if (!py::isinstance<py::none>(cloned_py_obj)) {
    if (!py::isinstance<py::list>(cloned_py_obj)) {
      MS_LOG_WITH_NODE(EXCEPTION, parameter)
        << "parameter: " << parameter->DebugString() << " doesn't have correct cloned obj";
    }
    auto obj_list = py::cast<py::list>(cloned_py_obj);
    for (size_t i = 0; i < obj_list.size(); ++i) {
      py::object each_cloned_obj = obj_list[i];
      auto cloned_param_slice_shape = layout[py::cast<size_t>(2)];
      if (!opt_shard_group.cast<std::string>().empty()) {
        if (!IsAccuGradObj(each_cloned_obj) || grad_accumulation_shard) {
          cloned_param_slice_shape = layout[py::cast(SLICE_SHAPE_INDEX)];
        }
      }
      py::tuple cloned_param_layout = py::make_tuple(device_arrangement, tensor_map, cloned_param_slice_shape,
                                                     field_size, uniform_split, opt_shard_group, full_shape);
      (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, SLICE_PARAMETER_FN_NAME, each_cloned_obj, py::str(phase),
                                     cloned_param_layout);
    }
  }
}

void InitCompileCacheParams(const pipeline::ResourcePtr &resource) {
  auto layout_dict = GetParameterLayoutFromResource(resource);
  auto graph = resource->func_graph();
  auto params = graph->parameters();
  for (auto &param : params) {
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (!param_ptr->has_default()) {
      continue;
    }
    SliceCacheParameterObj(param_ptr, layout_dict);
  }
}

void InitPynativeNoShardParams(const FuncGraphPtr &root) {
  auto parameters = root->parameters();
  for (auto &parameter : parameters) {
    auto param_ptr = parameter->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    auto param_info = param_ptr->param_info();
    if (!param_info) {
      MS_LOG(DEBUG) << "Parameter:" << parameter->DebugString() << " doesn't have param_info.";
      continue;
    }
    auto graph_executor = pipeline::GraphExecutorPy::GetInstance();
    MS_EXCEPTION_IF_NULL(graph_executor);
    auto phase = graph_executor->phase();
    auto py_obj = GetPyParameterObj(param_info, OBJ);
    if (py::isinstance<py::none>(py_obj)) {
      MS_LOG(WARNING) << "Parameter: " << parameter->DebugString() << " can't find python obj.";
      continue;
    }

    // param init in parallel mode
    param_info->set_is_param_init(true);

    (void)python_adapter::CallPyFn(SLICE_PARAMETER_FN_PATH, INIT_OPTIMIZER_STATE_FN, py_obj, py::str(phase));
  }
}

void AutoParallelPostProcess(const FuncGraphPtr &root) {
  auto parameters = root->parameters();
  auto enable_param_init_prof = common::GetEnv("MS_DEV_PARAM_INIT_PROF_COLLECT");
  for (auto &param : parameters) {
    if (ParameterIsCloned(param)) {
      continue;
    }
    auto layout = param->user_data<TensorLayout>();
    auto param_ptr = param->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    if (!param_ptr->has_default()) {
      continue;
    }
    if (layout != nullptr && !enable_param_init_prof.empty()) {
      auto slice_shape = layout->base_slice_shape().array();
      std::string opt_shard_group = layout->opt_shard_group();
      if (!opt_shard_group.empty()) {
        slice_shape = layout->opt_shard_slice_shape();
      }
      auto full_shape = layout->tensor_shape().array();
      MS_LOG(WARNING) << "Slice start: " << param_ptr->name() << ", full_shape: " << full_shape
                      << ", slice_shape: " << slice_shape;
    }
    SliceParameterObj(param_ptr, layout);
    if (layout != nullptr && !enable_param_init_prof.empty()) {
      MS_LOG(WARNING) << "Slice finish: " << param_ptr->name();
    }
    auto param_info = param_ptr->param_info();
    MS_EXCEPTION_IF_NULL(param_info);
    param_info->set_is_param_init(true);
  }
}

void GetSubRootParams(const AnfNodePtrList &root_params, AnfNodePtrList *sub_root_params) {
  for (auto &be_cloned_parameter_node : root_params) {
    if (ParallelContext::GetInstance()->get_redundancy_node().count(be_cloned_parameter_node)) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(be_cloned_parameter_node);
    auto be_cloned_parameter = be_cloned_parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(be_cloned_parameter);
    if (!be_cloned_parameter->has_default()) {
      continue;
    }

    auto param_value_in = be_cloned_parameter->param_info();
    if (param_value_in == nullptr) {
      continue;
    }
    if (!param_value_in->be_cloned()) {
      continue;
    }
    (*sub_root_params).emplace_back(be_cloned_parameter_node);
  }
}

void SetClonedTensorShapeForOptimizer(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  auto grad_accumulation_shard =
    ParallelContext::GetInstance()->grad_accumulation_shard() || ParallelContext::GetInstance()->zero3();
  auto root_params = root->parameters();
  AnfNodePtrList sub_root_params;
  GetSubRootParams(root_params, &sub_root_params);

  for (auto &cloned_parameter_node : root_params) {
    if (ParallelContext::GetInstance()->get_redundancy_node().count(cloned_parameter_node)) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(cloned_parameter_node);
    auto cloned_parameter = cloned_parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(cloned_parameter);

    if (!ParameterIsCloned(cloned_parameter_node)) {
      continue;
    }
    auto param_value = cloned_parameter->param_info();
    if (param_value == nullptr) {
      continue;
    }
    // get the cloned index
    int64_t cloned_index = param_value->cloned_index();

    // find the be cloned parameter
    bool found_be_cloned_parameter = false;
    ParameterPtr cloned_from_parameter = nullptr;
    AnfNodePtr cloned_from_node = nullptr;
    for (auto &be_cloned_parameter_node : sub_root_params) {
      MS_EXCEPTION_IF_NULL(be_cloned_parameter_node);
      auto be_cloned_parameter = be_cloned_parameter_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(be_cloned_parameter);
      auto param_value_in = be_cloned_parameter->param_info();
      // get the be cloned index
      MS_EXCEPTION_IF_NULL(param_value_in);
      auto &be_cloned_index = param_value_in->be_cloned_index();
      if (std::find(be_cloned_index.begin(), be_cloned_index.end(), cloned_index) != be_cloned_index.end()) {
        found_be_cloned_parameter = true;
        cloned_from_parameter = be_cloned_parameter;
        cloned_from_node = be_cloned_parameter_node;
      }
    }

    if (found_be_cloned_parameter) {
      // set the shape and tensor layout for cloned parameter
      MS_EXCEPTION_IF_NULL(cloned_parameter_node);
      std::string param_name = cloned_parameter_node->cast<ParameterPtr>()->name();
      if (cloned_from_parameter->user_data<TensorLayout>() == nullptr) {
        MS_LOG(WARNING) << "The parameter " << param_name << " has not tensor layout, skip it";
        continue;
      }
      auto tensor_layout = cloned_from_parameter->user_data<TensorLayout>();
      MS_EXCEPTION_IF_NULL(cloned_parameter_node->abstract());
      MS_EXCEPTION_IF_NULL(cloned_from_node->abstract());
      auto cloned_abstract = cloned_parameter_node->abstract()->Clone();
      MS_EXCEPTION_IF_NULL(cloned_abstract);
      // from pipeline or grad accumulation
      if (param_name.find(ACCU_GRADS) != std::string::npos) {
        auto slice_shape = cloned_from_parameter->user_data<TensorLayout>()->base_slice_shape().array();
        auto opt_shard_group = tensor_layout->opt_shard_group();
        auto opt_shard_shape = cloned_from_parameter->user_data<TensorLayout>()->opt_shard_slice_shape();
        std::shared_ptr<abstract::BaseShape> parallel_shape = nullptr;
        // set opt shard shape if the pipeline sharding is set
        if (grad_accumulation_shard && !opt_shard_group.empty()) {
          parallel_shape = std::make_shared<abstract::Shape>(opt_shard_shape);
        } else {
          parallel_shape = std::make_shared<abstract::Shape>(slice_shape);
        }
        MS_EXCEPTION_IF_NULL(parallel_shape);
        cloned_abstract->set_shape(parallel_shape);
        // in opt shard, accu_grad's shape is different from the original param's shape
        // if the grad_accumulation_shard is enabled, the accu_grads will be a opt-sharded shape
        if (!grad_accumulation_shard && ParallelContext::GetInstance()->enable_parallel_optimizer()) {
          TensorLayout new_layout = *tensor_layout;
          new_layout.set_opt_shard_group("");
          tensor_layout = std::make_shared<TensorLayout>(new_layout);
        }
      } else {
        cloned_abstract->set_shape(cloned_from_node->abstract()->GetShapeTrack());
      }
      cloned_parameter->set_user_data<TensorLayout>(tensor_layout);
      cloned_parameter_node->set_abstract(cloned_abstract);
      // copy the fusion tag
      auto cloned_param_info = cloned_parameter->param_info();
      MS_EXCEPTION_IF_NULL(cloned_param_info);
      auto cloned_from_param_info = cloned_from_parameter->param_info();
      MS_EXCEPTION_IF_NULL(cloned_from_param_info);
      cloned_param_info->set_comm_fusion(cloned_from_param_info->comm_fusion());

      MS_LOG(INFO) << "The parameter: " << cloned_parameter->name()
                   << " is cloned, the be cloned parameter is: " << cloned_from_parameter->name()
                   << ", clone index is:  " << cloned_index;
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, cloned_parameter)
        << "The parameter: " << cloned_parameter->name() << " is cloned, cloned index is  " << cloned_index
        << ", but not found the be cloned parameter";
    }
  }
}

// For adafactor optimizer, the relationship between parameter and state's shape as follows:
// 1) parameter: [A, B, C, D] (shape_size > 2), exp_avg_sq_row: [A, B, C], exp_avg_sq_col: [A, B, D], exp_avg_sq: [1]
//    If the parameter is opt shard, the exp_avg_sq_row and exp_avg_sq_col need to be shard accordingly.
// 2) parameter: [A, B] (shape_size = 2), exp_avg_sq_row: [A], exp_avg_sq_col: [B], exp_avg_sq: [1]
//    If the parameter is opt shard, the exp_avg_sq_row needs to be shard accordingly.
// 3) parameter: [A] (shape_size = 1), exp_avg_sq_row: [1], exp_avg_sq_col: [1], exp_avg_sq: [A]
//    If the parameter is opt shard, the exp_avg_sq needs to be shard accordingly.
static bool AdafactorStateIsOptShard(const std::string &opt_shard_group, size_t shape_size,
                                     const std::string &param_name, const std::string &state_name) {
  if (opt_shard_group.empty()) {
    return false;
  }

  std::string exp_row_name = EXP_AVG_SQ_ROW + param_name;
  std::string exp_col_name = EXP_AVG_SQ_COL + param_name;
  std::string exp_avg_name = EXP_AVG_SQ + param_name;
  std::string exp_insta_row_name = EXP_AVG_INSTA_ROW + param_name;
  std::string exp_insta_col_name = EXP_AVG_INSTA_COL + param_name;

  if (shape_size > 2 && state_name == exp_avg_name) {
    return false;
  }

  if (shape_size == 2 &&
      (state_name == exp_col_name || state_name == exp_avg_name || state_name == exp_insta_col_name)) {
    return false;
  }

  if (shape_size == 1 &&
      (state_name == exp_row_name || state_name == exp_col_name || state_name == exp_insta_row_name)) {
    return false;
  }

  MS_LOG(INFO) << "The parameter " << param_name << " is opt shard";
  return true;
}

static bool IsOriginWeight(const ParameterPtr &param) {
  std::string param_name = param->name();
  if (param_name.find(EXP_AVG) != std::string::npos) {
    return false;
  }

  auto tensor_layout = param->user_data<TensorLayout>();
  if (tensor_layout == nullptr) {
    return false;
  }

  return true;
}

static std::pair<AnfNodePtr, bool> FindParameterByValueNode(const AnfNodePtr &node, const FuncGraphPtr &func_graph,
                                                            const std::string &name = ALL_REDUCE) {
  if (IsValueNode<RefKey>(node)) {
    std::vector<AnfNodePtr> param_v = FindParameterByRefKeyNode(node, func_graph);
    if (param_v.size() != 1) {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "FindParameterByRefKeyNode failed, return vector size must be 1, real is  "
                                        << param_v.size();
    }
    auto param_ptr = param_v[0]->user_data<parallel::TensorLayout>();
    if (param_ptr && !param_ptr->opt_shard_group().empty() && param_ptr->opt_shard_mirror_group().empty() &&
        name == ALL_REDUCE) {
      return std::make_pair(nullptr, true);
    }
    return std::make_pair(node, true);
  }
  return std::make_pair(nullptr, false);
}

AnfNodePtr RefParameterToActualParameter(const AnfNodePtr &node) {
  if (!node->isa<Parameter>()) {
    return nullptr;
  }
  auto node_param_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(node_param_ptr);
  if (node_param_ptr->has_default()) {
    return node;
  }
  auto sub_func_graph = node_param_ptr->func_graph();
  auto call_cnodes_map = sub_func_graph->func_graph_cnodes_index();
  auto sub_graph_parameters = sub_func_graph->parameters();
  auto curr_param_iter = std::find(sub_graph_parameters.begin(), sub_graph_parameters.end(), node);
  if (curr_param_iter == sub_graph_parameters.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, node_param_ptr)
      << "Cannot find param " << node_param_ptr->DebugString() << " in current sub_graph";
  }
  size_t curr_param_index = static_cast<size_t>(curr_param_iter - sub_graph_parameters.begin());
  for (const auto &node_pair : call_cnodes_map) {
    if (!node_pair.first->first->isa<CNode>() || node_pair.first->second > 0) {
      continue;
    }
    auto cnode = node_pair.first->first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto cnode_input = cnode->input(curr_param_index + 1);
    auto new_cnode = GetInputNodeWithFilter(cnode_input, [&](const CNodePtr &cnode) {
      bool filter = IsPrimitiveCNode(cnode, prim::kPrimMicroStepAllGather) ||
                    IsPrimitiveCNode(cnode, prim::kPrimLoad) || IsPrimitiveCNode(cnode, prim::kPrimDepend) ||
                    IsPrimitiveCNode(cnode, prim::kPrimCast) ||
                    (IsPrimitiveCNode(cnode, prim::kPrimAllGather) &&
                     GetCNodePrimitive(cnode)->instance_name().find(PARALLEL_OPTIMIZER) != std::string::npos);
      return std::make_pair(filter, 1);
    });
    return RefParameterToActualParameter(new_cnode);
  }
  return nullptr;
}

static std::pair<AnfNodePtr, bool> FindParameterByParameter(const AnfNodePtr &node,
                                                            const std::string &name = ALL_REDUCE) {
  if (!node->isa<Parameter>()) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "The node is not a parameter, node:" << node->DebugString();
  }
  auto node_param_ptr = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(node_param_ptr);
  if (node_param_ptr->has_default()) {
    auto param_ptr = node->user_data<parallel::TensorLayout>();
    if (param_ptr) {
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (!param_ptr->opt_shard_group().empty() && param_ptr->opt_shard_mirror_group().empty() && name == ALL_REDUCE) {
        return std::make_pair(nullptr, false);
      }
    }
    return std::make_pair(node, false);
  }
  AnfNodePtr ref_param = RefParameterToActualParameter(node);
  if (!ref_param) {
    return std::make_pair(nullptr, false);
  }
  auto ref_param_layout = ref_param->user_data<parallel::TensorLayout>();
  if (ref_param_layout && ParallelContext::GetInstance()->zero3()) {
    return std::make_pair(ref_param, false);
  }
  if (ref_param_layout && !ref_param_layout->opt_shard_group().empty() &&
      ref_param_layout->opt_shard_mirror_group().empty() && name == ALL_REDUCE) {
    return std::make_pair(nullptr, false);
  }
  return std::make_pair(ref_param, false);
}

static std::pair<AnfNodePtr, bool> FindParameterByFuncGraph(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto fg = GetValueNode<FuncGraphPtr>(cnode->input(0));
  MS_EXCEPTION_IF_NULL(fg);
  auto pre_node = GetRealKernelNode(fg->output(), -1, nullptr).first;
  if (pre_node) {
    return FindParameter(pre_node, pre_node->func_graph());
  }
  return std::make_pair(nullptr, false);
}

bool IsSkipNodes(const PrimitivePtr &prim) {
  return prim->name() == DEPEND || prim->name() == LOAD || prim->name() == INSERTGRADIENTOF || prim->name() == CAST;
}

// Only used for InsertMirrorOps
std::pair<AnfNodePtr, bool> FindParameter(const AnfNodePtr &node, const FuncGraphPtr &func_graph) {
  if (!node->isa<Parameter>() && !node->isa<CNode>() && !node->isa<ValueNode>()) {
    return std::make_pair(nullptr, false);
  }

  if (node->isa<Parameter>()) {
    return FindParameterByParameter(node);
  }

  if (node->isa<ValueNode>()) {
    return FindParameterByValueNode(node, func_graph);
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (IsValueNode<FuncGraph>(cnode->input(0))) {
    return FindParameterByFuncGraph(node);
  }
  if (!IsValueNode<Primitive>(cnode->input(0))) {
    for (size_t index = 0; index < cnode->size(); ++index) {
      auto res = FindParameter(cnode->input(index), func_graph);
      if (!res.first) {
        continue;
      }
      return res;
    }
  }

  // When not fully use opt shard, allgather and mirror would be both inserted.
  // Skip allgather here and find parameter recursively.
  if (IsParallelCareNode(cnode) && !IsInAllGatherNodeList(cnode)) {
    return std::make_pair(nullptr, false);
  }
  ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(prim_anf_node);
  for (size_t index = 0; index < cnode->size(); ++index) {
    PrimitivePtr prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(prim);
    if (prim->name() == DUMPGRADIENT && index != kDumpGradientSkipIndex) {
      continue;
    }
    if ((IsSkipNodes(prim) || IsInAllGatherNodeList(cnode)) && index != 1) {
      continue;
    }
    auto res = FindParameter(cnode->input(index), func_graph);
    if (!res.first) {
      continue;
    }
    return res;
  }
  return std::make_pair(nullptr, false);
}

// Used for allgather and reducescatter
std::pair<AnfNodePtr, bool> FindParameterWithAllgather(const AnfNodePtr &node, const FuncGraphPtr &func_graph,
                                                       const std::string &name) {
  if (!node->isa<Parameter>() && !node->isa<CNode>() && !node->isa<ValueNode>()) {
    return std::make_pair(nullptr, false);
  }

  if (node->isa<Parameter>()) {
    return FindParameterByParameter(node, name);
  }

  if (node->isa<ValueNode>()) {
    return FindParameterByValueNode(node, func_graph, name);
  }

  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  for (size_t index = 0; index < cnode->size(); ++index) {
    if (index != 1) {
      continue;
    }
    auto res = FindParameterWithAllgather(cnode->input(index), func_graph, name);
    if (!res.first) {
      continue;
    }
    return res;
  }
  return std::make_pair(nullptr, false);
}

std::unordered_map<std::string, std::shared_ptr<TensorLayout>> AdaSumParamTensorLayout(const FuncGraphPtr &root) {
  MS_EXCEPTION_IF_NULL(root);
  std::unordered_map<std::string, std::shared_ptr<TensorLayout>> adasum_param_map;
  for (auto &parameter_node : root->parameters()) {
    MS_EXCEPTION_IF_NULL(parameter_node);
    auto cloned_parameter = parameter_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(cloned_parameter);

    if (!ParameterIsCloned(parameter_node)) {
      auto parameter_tensor_layout = cloned_parameter->user_data<TensorLayout>();
      adasum_param_map["adasum_delta_weight." + cloned_parameter->name()] = parameter_tensor_layout;
    }
  }
  return adasum_param_map;
}

Shape ValueSequeueScaleToShape(const ValuePtr &value_seq, const Shape &scale, size_t expand_ratio = 1) {
  MS_EXCEPTION_IF_NULL(value_seq);
  if (!value_seq->isa<ValueSequeue>()) {
    MS_LOG(EXCEPTION) << "The input is not a value_sequeue";
  }
  std::vector<int64_t> origin_value_vector;
  if (TransValueSequeueToVector(value_seq, &origin_value_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Transform value_seq to vector failed";
  }
  if (origin_value_vector.size() > scale.size()) {
    MS_LOG(EXCEPTION) << "Cannot scale, the size of value_seq is: " << origin_value_vector.size()
                      << ", which should be less_equal than scale's size which is: " << scale.size();
  }
  for (size_t i = 0; i < origin_value_vector.size(); ++i) {
    origin_value_vector[i] = origin_value_vector[i] / scale[i];
    if (i == 0) {
      origin_value_vector[i] = origin_value_vector[i] * SizeToLong(expand_ratio);
    }
  }
  return origin_value_vector;
}

ValuePtr ValueSequeueScale(const ValuePtr &value_seq, const Shape &scale, size_t expand_ratio = 1) {
  Shape origin_value_vector = ValueSequeueScaleToShape(value_seq, scale, expand_ratio);
  if (value_seq->isa<ValueTuple>()) {
    return TransVectorToValueSequeue<ValueTuple>(origin_value_vector);
  }
  return TransVectorToValueSequeue<ValueList>(origin_value_vector);
}

void ReplaceAdaSumStridedSliceValue(const CNodePtr &stridedslice_cnode1,
                                    const std::shared_ptr<TensorLayout> &target_param_layout,
                                    size_t slice_expand_ratio) {
  auto target_param_info = std::make_shared<TensorInfo>(target_param_layout->SqueezeShape());
  MS_EXCEPTION_IF_NULL(target_param_info);
  Dimensions param_strategy = target_param_info->InferStrategy();
  MS_EXCEPTION_IF_NULL(stridedslice_cnode1);
  auto new_begin1_value =
    ValueSequeueScale(GetValueNode(stridedslice_cnode1->input(2)), param_strategy, slice_expand_ratio);
  auto new_end1_value =
    ValueSequeueScale(GetValueNode(stridedslice_cnode1->input(3)), param_strategy, slice_expand_ratio);
  ValueNodePtr new_begin_value_node = std::make_shared<ValueNode>(new_begin1_value);
  ValueNodePtr new_end_value_node = std::make_shared<ValueNode>(new_end1_value);
  stridedslice_cnode1->set_input(2, new_begin_value_node);
  stridedslice_cnode1->set_input(3, new_end_value_node);
}

RankList GetRankListByLayout(const std::shared_ptr<TensorLayout> &target_param_layout) {
  int64_t rank = g_device_manager->global_rank();
  auto dev_shape = target_param_layout->device_arrangement().array();
  auto stage_device_list = g_device_manager->GetDeviceListInThisStage();
  DeviceMatrix dev_matrix(rank, stage_device_list, dev_shape);
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(target_param_layout->tensor_map().array(), &group_devices) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Get adasum parameter origin mirror group by tensor layout failed.";
  }
  return group_devices;
}

std::vector<bool> IsBorderAdaSumSendReceive(const AnfNodePtr &node, const RankList &group_devices) {
  bool is_send = IsPrimitiveCNode(node, prim::kPrimSend);
  PrimitivePtr send_rec_prim = GetCNodePrimitive(node);
  int64_t origin_dest_rank = GetValue<int64_t>(send_rec_prim->GetAttr(OPPOSITE_RANK));
  int64_t rank = g_device_manager->global_rank();
  if (group_devices.size() - 1 == 0) {
    MS_LOG(EXCEPTION) << "May division by zero.";
  }
  int64_t adasum_rank_distance = (group_devices.back() - group_devices.front()) / SizeToLong(group_devices.size() - 1);
  if (adasum_rank_distance < ADASUM_MIN_DIS) {
    adasum_rank_distance = ADASUM_MIN_DIS;
  }
  size_t border_step = size_t(log2(adasum_rank_distance / ADASUM_MIN_DIS));
  int64_t fusion_id = GetValue<int64_t>(send_rec_prim->GetAttr("origin_fusion"));
  // when cutting nodes, the fusion id should change.
  int64_t new_fusion_id = fusion_id + SizeToLong(g_device_manager->DeviceNum() * (border_step + IntToSize(1)));
  send_rec_prim->set_attr(FUSION, MakeValue(new_fusion_id));
  std::vector<int64_t> group_list;
  int64_t new_dest_src_rank;
  if (rank > origin_dest_rank) {
    group_list = {origin_dest_rank, rank};
    new_dest_src_rank = 0;
  } else {
    group_list = {rank, origin_dest_rank};
    new_dest_src_rank = 1;
  }
  Group adasum_send_rec_group;
  if (g_device_manager->CreateGroup(group_list, &adasum_send_rec_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create send/receive group in adasum failed, the group is:" << group_list;
  }
  send_rec_prim->set_attr(GROUP, MakeValue(adasum_send_rec_group.name()));
  if (is_send) {
    send_rec_prim->set_attr(DEST_RANK, MakeValue(new_dest_src_rank));
  } else {
    send_rec_prim->set_attr(SRC_RANK, MakeValue(new_dest_src_rank));
  }
  int64_t rank_dis = abs(origin_dest_rank - rank);
  if (adasum_rank_distance == ADASUM_MIN_DIS) {
    return {false, false, false, false};
  }
  bool is_origin_first_node_if_forward = false;
  bool is_new_first_node_if_forward = false;
  bool is_origin_last_node_if_rollback = false;
  bool is_new_last_node_if_rollback = false;
  if (rank_dis == ADASUM_MIN_DIS) {
    is_origin_first_node_if_forward = true;
    is_origin_last_node_if_rollback = true;
  }
  if (rank_dis == adasum_rank_distance) {
    is_new_first_node_if_forward = true;
  }
  if (rank_dis == adasum_rank_distance / 2) {
    is_new_last_node_if_rollback = true;
  }
  return {is_origin_first_node_if_forward, is_new_first_node_if_forward, is_origin_last_node_if_rollback,
          is_new_last_node_if_rollback};
}

void HandleAdaSumReshape(const CNodePtr &reshape_cnode, const std::shared_ptr<TensorLayout> &target_param_layout) {
  auto slice_shape = target_param_layout->slice_shape().array();
  auto slice_shape_value = TransVectorToValueSequeue<ValueTuple>(slice_shape);
  ValueNodePtr new_slice_shape_value_node = std::make_shared<ValueNode>(slice_shape_value);
  reshape_cnode->set_input(2, new_slice_shape_value_node);
}

void RemoveAdasumRedundantNodes(const FuncGraphManagerPtr &manager,
                                std::unordered_map<std::string, CNodePtr> *forward_origin_first_node_map,
                                std::unordered_map<std::string, CNodePtr> *forward_new_first_node_map,
                                std::unordered_map<std::string, CNodePtr> *rollback_origin_last_node_map,
                                std::unordered_map<std::string, CNodePtr> *rollback_new_last_node_map) {
  // connect forward last node and rollback first node
  if (forward_origin_first_node_map->size() != forward_new_first_node_map->size() ||
      rollback_origin_last_node_map->size() != rollback_new_last_node_map->size()) {
    MS_LOG(EXCEPTION) << "The over border node is not equal in adasum forward process and rollback process.";
  }
  for (auto node : *forward_origin_first_node_map) {
    std::string target_param = node.first;
    CNodePtr forward_origin_first_node = node.second;
    CNodePtr forward_new_first_node = (*forward_new_first_node_map)[target_param];
    manager->SetEdge(forward_new_first_node, 1, forward_origin_first_node->input(1));
  }
  for (auto node : *rollback_origin_last_node_map) {
    std::string target_param = node.first;
    CNodePtr rollback_origin_last_node = node.second;
    CNodePtr rollback_new_last_node = (*rollback_new_last_node_map)[target_param];
    (void)manager->Replace(rollback_origin_last_node, rollback_new_last_node);
  }
}

void HandleAdasumAllReduce(const PrimitivePtr &prim, const RankList &group_devices) {
  size_t step = size_t(GetValue<int64_t>(prim->GetAttr("step")));
  std::vector<int64_t> neighbor_ids;
  int64_t adasum_rank_distance =
    (group_devices.back() - group_devices.front()) / SizeToLong((group_devices.size() - 1));
  if (adasum_rank_distance < ADASUM_MIN_DIS) {
    adasum_rank_distance = ADASUM_MIN_DIS;
  }
  size_t border_step = size_t(log2(adasum_rank_distance / ADASUM_MIN_DIS));
  MS_LOG(INFO) << "current border step is: " << border_step;
  if (step < border_step) {
    return;
  }
  int64_t rank = g_device_manager->global_rank();
  size_t double_d = size_t(IntToSize(2) << step);
  for (size_t index = 0; index < double_d; ++index) {
    int64_t node_rank = rank / ADASUM_MIN_DIS;
    int64_t neighbor_id =
      (node_rank / SizeToLong(double_d) * SizeToLong(double_d) + SizeToLong(index)) * ADASUM_MIN_DIS +
      rank % ADASUM_MIN_DIS;
    neighbor_ids.push_back(neighbor_id);
  }
  Group adasum_allreduce_group;
  if (g_device_manager->CreateGroup(neighbor_ids, &adasum_allreduce_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create group allreduce group in adasum failed, the group is " << neighbor_ids;
  }
  auto new_group_name = MakeValue(adasum_allreduce_group.name());
  int64_t fusion_id = GetValue<int64_t>(prim->GetAttr("origin_fusion"));
  int64_t new_fusion_id = fusion_id + SizeToLong(g_device_manager->DeviceNum() * (border_step + IntToSize(1)));
  prim->set_attr(GROUP, new_group_name);
  prim->set_attr(FUSION, MakeValue(new_fusion_id));
}

void HandleAdasumSlice(const AnfNodePtr &stridedslice_node1, const std::shared_ptr<TensorLayout> &target_param_layout,
                       size_t slice_expand_ratio) {
  auto stridedslice_cnode1 = stridedslice_node1->cast<CNodePtr>();
  ReplaceAdaSumStridedSliceValue(stridedslice_cnode1, target_param_layout, slice_expand_ratio);
  auto squeeze_node = RealInputNode(stridedslice_cnode1, 1);
  if (!IsPrimitiveCNode(squeeze_node, prim::kPrimSqueeze)) {
    MS_LOG_WITH_NODE(EXCEPTION, squeeze_node) << "The stridedslice input node should be squeeze in adasum";
  }
  auto squeeze_cnode = squeeze_node->cast<CNodePtr>();
  FuncGraphManagerPtr manager = squeeze_node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[squeeze_cnode];
  for (auto &node_pair : node_set) {
    if (IsPrimitiveCNode(node_pair.first, prim::kPrimStridedSlice) && node_pair.first != stridedslice_node1) {
      CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
      ReplaceAdaSumStridedSliceValue(use_apply, target_param_layout, slice_expand_ratio);
    }
  }
}

void HandleAdaSumConcat(const AnfNodePtr &concat_node, const std::vector<bool> &border_info,
                        const std::string &target_param,
                        std::unordered_map<std::string, CNodePtr> *rollback_new_last_node_map,
                        std::unordered_map<std::string, CNodePtr> *rollback_origin_last_node_map) {
  if (border_info[THIRD_BORDER_INFO_INDEX]) {
    (*rollback_new_last_node_map)[target_param] = concat_node->cast<CNodePtr>();
  }
  if (border_info[SECOND_BORDER_INFO_INDEX]) {
    auto manager = concat_node->func_graph()->manager();
    AnfNodeIndexSet concat_node_user_set = manager->node_users()[concat_node];
    for (auto &node_pair : concat_node_user_set) {
      if (IsPrimitiveCNode(node_pair.first, prim::kPrimMakeTuple)) {
        AnfNodeIndexSet make_tuple_node_user_set = manager->node_users()[node_pair.first];
        for (auto &tuple_user : make_tuple_node_user_set) {
          if (IsPrimitiveCNode(tuple_user.first, prim::kPrimConcat)) {
            (*rollback_origin_last_node_map)[target_param] = tuple_user.first->cast<CNodePtr>();
            return;
          }
        }
        return;
      }
    }
  }
}

void HandleAdaSumSqueeze(const AnfNodePtr &stridedslice_node1, const std::vector<bool> &border_info,
                         const std::string &target_param,
                         std::unordered_map<std::string, CNodePtr> *forward_origin_first_node_map,
                         std::unordered_map<std::string, CNodePtr> *forward_new_first_node_map) {
  auto squeeze_node = RealInputNode(stridedslice_node1->cast<CNodePtr>(), 1);
  if (border_info[0]) {
    (*forward_origin_first_node_map)[target_param] = squeeze_node->cast<CNodePtr>();
  }
  if (border_info[1]) {
    (*forward_new_first_node_map)[target_param] = squeeze_node->cast<CNodePtr>();
  }
}

void HandleAdaSumPureModelParallel(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimSend) && !IsPrimitiveCNode(node, prim::kPrimReceive)) {
    return;
  }
  PrimitivePtr send_rec_prim = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(send_rec_prim);
  int64_t origin_dest_rank = GetValue<int64_t>(send_rec_prim->GetAttr(OPPOSITE_RANK));
  int64_t rank = g_device_manager->global_rank();
  CNodePtr cnode = node->cast<CNodePtr>();
  auto pre_cnode = RealInputNode(cnode, 1);
  int64_t rank_dis = abs(origin_dest_rank - rank);
  if (rank_dis == ADASUM_MIN_DIS && IsPrimitiveCNode(pre_cnode, prim::kPrimStridedSlice)) {
    MS_EXCEPTION_IF_NULL(pre_cnode);
    auto pre_cnnode_ptr = pre_cnode->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(pre_cnnode_ptr);
    auto squeeze_node = pre_cnnode_ptr->input(1);
    if (!IsPrimitiveCNode(squeeze_node, prim::kPrimSqueeze)) {
      return;
    }
    MS_EXCEPTION_IF_NULL(squeeze_node);
    auto squeeze_node_ptr = squeeze_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(squeeze_node_ptr);
    auto squeeze_input = squeeze_node_ptr->input(1);
    auto manager = squeeze_node->func_graph()->manager();
    AnfNodeIndexSet squeeze_input_node_user_set = manager->node_users()[squeeze_input];
    for (auto &squeeze_input_user : squeeze_input_node_user_set) {
      if (IsPrimitiveCNode(squeeze_input_user.first, prim::kPrimSqueeze) ||
          IsPrimitiveCNode(squeeze_input_user.first, prim::kPrimUpdateState) ||
          IsPrimitiveCNode(squeeze_input_user.first, prim::kPrimMakeTuple)) {
        continue;
      }
      (void)manager->Replace(squeeze_input_user.first, squeeze_input);
    }
  }
}

bool HandleAdaSum(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                  std::unordered_map<std::string, std::shared_ptr<TensorLayout>> *adasum_param_tensor_layout_map) {
  std::unordered_map<std::string, CNodePtr> forward_origin_first_node_map;
  std::unordered_map<std::string, CNodePtr> forward_new_first_node_map;
  std::unordered_map<std::string, CNodePtr> rollback_origin_last_node_map;
  std::unordered_map<std::string, CNodePtr> rollback_new_last_node_map;
  bool is_adasum = false;
  for (auto &node : all_nodes) {
    bool is_allreduce = IsPrimitiveCNode(node, prim::kPrimAllReduce);
    bool is_reshape = IsPrimitiveCNode(node, prim::kPrimReshape);
    bool is_send = IsPrimitiveCNode(node, prim::kPrimSend);
    bool is_receive = IsPrimitiveCNode(node, prim::kPrimReceive);
    if (!is_allreduce && !is_reshape && !is_send && !is_receive) {
      continue;
    }
    std::string target_param;
    MS_EXCEPTION_IF_NULL(node);
    CNodePtr cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(cnode->input(0)->cast<ValueNodePtr>());
    MS_EXCEPTION_IF_NULL(prim);
    if (!prim->HasAttr(TARGET_PARAM)) {
      continue;
    }
    target_param = GetValue<std::string>(prim->GetAttr(TARGET_PARAM));
    auto target_param_layout = (*adasum_param_tensor_layout_map)[target_param];
    RankList group_devices = GetRankListByLayout(target_param_layout);
    // only model parallel
    if (group_devices.size() == 1) {
      HandleAdaSumPureModelParallel(node);
      continue;
    }

    int64_t adasum_rank_distance =
      (group_devices.back() - group_devices.front()) / SizeToLong((group_devices.size() - 1));
    // when the repeat dim is right, the parameter do not enable adasum.
    if (adasum_rank_distance == 1 && group_devices.size() < size_t(g_device_manager->stage_device_num())) {
      continue;
    }
    MS_LOG(INFO) << "Apply adasum in auto parallel, current dealing node is: " << node->fullname_with_scope();
    is_adasum = true;
    size_t slice_expand_ratio =
      LongToSize(adasum_rank_distance / ADASUM_MIN_DIS) > 0 ? LongToSize(adasum_rank_distance / ADASUM_MIN_DIS) : 1;
    if (is_reshape) {
      HandleAdaSumReshape(cnode, (*adasum_param_tensor_layout_map)[target_param]);
    }
    if (is_allreduce && prim->HasAttr("step")) {
      HandleAdasumAllReduce(prim, group_devices);
    }
    if (is_send || is_receive) {
      std::vector<bool> border_info = IsBorderAdaSumSendReceive(node, group_devices);
      if (is_receive) {
        auto target_param_info = std::make_shared<TensorInfo>(*target_param_layout);
        Dimensions param_strategy = target_param_info->InferStrategy();
        Shape new_rec_shape = ValueSequeueScaleToShape(prim->GetAttr(SHAPE), param_strategy, slice_expand_ratio);
        auto new_rec_shape_value = TransVectorToValueSequeue<ValueList>(new_rec_shape);
        prim->set_attr(SHAPE, new_rec_shape_value);
        continue;
      }
      auto stridedslice_node1 = RealInputNode(cnode, 1);
      if (IsPrimitiveCNode(stridedslice_node1, prim::kPrimConcat)) {
        HandleAdaSumConcat(stridedslice_node1, border_info, target_param, &rollback_new_last_node_map,
                           &rollback_origin_last_node_map);
        continue;
      }
      if (!IsPrimitiveCNode(stridedslice_node1, prim::kPrimStridedSlice)) {
        continue;
      }
      HandleAdasumSlice(stridedslice_node1, target_param_layout, slice_expand_ratio);
      HandleAdaSumSqueeze(stridedslice_node1, border_info, target_param, &forward_origin_first_node_map,
                          &forward_new_first_node_map);
    }
  }
  RemoveAdasumRedundantNodes(root->manager(), &forward_origin_first_node_map, &forward_new_first_node_map,
                             &rollback_origin_last_node_map, &rollback_new_last_node_map);
  return is_adasum;
}

void ResetMirrorAttr(const PrimitivePtr &prim, const RankList &new_group) {
  MS_EXCEPTION_IF_NULL(prim);
  if (new_group.size() == 1) {
    MS_EXCEPTION_IF_NULL(prim);
    prim->set_attr(DEV_NUM, MakeValue<int64_t>(SizeToLong(new_group.size())));
    prim->set_attr(GROUP, MakeValue("one_rank_group"));
    prim->set_attr(GROUP_RANKS, MakeValue(std::to_string(new_group[0])));
    return;
  }
  Group adasum_mirror_group;
  if (g_device_manager->CreateGroup(new_group, &adasum_mirror_group) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create new mirror group failed in adasum, new group is: " << new_group;
  }
  auto new_group_name = MakeValue(adasum_mirror_group.name());
  prim->set_attr(GROUP, new_group_name);
  prim->set_attr(DEV_NUM, MakeValue<int64_t>(SizeToLong(new_group.size())));
  std::string rank_list_name = g_device_manager->FindRankListNameByHashName(adasum_mirror_group.name());
  prim->set_attr(GROUP_RANKS, MakeValue(rank_list_name));
}

void HandleMirrorInAdaSum(
  const FuncGraphPtr &root,
  std::unordered_map<std::string, std::shared_ptr<TensorLayout>> *adasum_param_tensor_layout_map) {
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(root->get_return());
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimMirror)) {
      continue;
    }
    CNodePtr mirror_cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(mirror_cnode);
    auto param_node_pair = FindParameter(mirror_cnode->input(1), node->func_graph());
    if (!param_node_pair.first) {
      MS_LOG_WITH_NODE(EXCEPTION, mirror_cnode) << "Mirror input is not a param";
    }
    auto param_ptr = param_node_pair.first->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param_ptr);
    std::string param_name = param_ptr->name();
    MS_LOG(INFO) << "Mirror param name is: " << param_name;
    std::string target_param = "adasum_delta_weight." + param_name;
    auto target_param_layout = (*adasum_param_tensor_layout_map)[target_param];

    // Change mirror group
    RankList group_devices = GetRankListByLayout(target_param_layout);
    int64_t rank = g_device_manager->global_rank();
    size_t group_dis = LongToSize(group_devices.back() - group_devices.front()) / (group_devices.size() - 1);
    auto prim = GetCNodePrimitive(node);
    if (group_dis < ADASUM_MIN_DIS && group_dis > 0) {
      size_t new_group_size = size_t(ADASUM_MIN_DIS) / group_dis;
      // compute new group range
      size_t group_begin = 0;
      for (size_t group_end = new_group_size; group_end < group_devices.size() + new_group_size;
           group_end += new_group_size) {
        int64_t max_group_value =
          group_end >= group_devices.size() ? (group_devices.back() + 1) : group_devices[group_end];
        if (group_devices[group_begin] <= rank && rank < max_group_value) {
          std::vector<int64_t> new_group(group_devices.begin() + SizeToLong(group_begin),
                                         group_devices.begin() + SizeToLong(group_end));
          MS_LOG(INFO) << "Find new mirror group in adasum: " << new_group << " target_param:" << target_param;
          ResetMirrorAttr(prim, new_group);
          break;
        }
        group_begin = group_end;
      }
      continue;
    }
    ResetMirrorAttr(prim, {rank});
  }
}

void SetParamInfoSaveStrategy(ParameterPtr row_col_param) {
  if (!row_col_param) {
    return;
  }
  auto param_info = row_col_param->param_info();
  if (param_info) {
    param_info->set_strategy_ckpt_saved(true);
  }
}

void FindParamNodes(std::unordered_map<string, AnfNodePtr> *root_params_map_1,
                    std::unordered_map<AnfNodePtr, AnfNodePtrList> *root_params_map,
                    const AnfNodePtrList &root_params) {
  AnfNodePtrList exp_root_params;
  std::string exp_row_name = EXP_AVG_SQ_ROW;
  std::string exp_col_name = EXP_AVG_SQ_COL;
  std::string exp_insta_row_name = EXP_AVG_INSTA_ROW;
  std::string exp_insta_col_name = EXP_AVG_INSTA_COL;
  std::string exp_avg_name = EXP_AVG_SQ;
  for (auto &param_node : root_params) {
    if (ParallelContext::GetInstance()->get_redundancy_node().find(param_node) !=
        ParallelContext::GetInstance()->get_redundancy_node().end()) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(param_node);
    auto param = param_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    std::string param_name = param->name();
    if (IsOriginWeight(param)) {
      (*root_params_map_1)[param_name] = param_node;
    } else {
      exp_root_params.push_back(param_node);
    }
  }
  for (auto &param_node : exp_root_params) {
    if (ParallelContext::GetInstance()->get_redundancy_node().find(param_node) !=
        ParallelContext::GetInstance()->get_redundancy_node().end()) {
      continue;
    }
    auto param = param_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    std::string param_name = param->name();
    if ((param_name.substr(0, kPreLenFifteen) == exp_row_name ||
         param_name.substr(0, kPreLenFifteen) == exp_col_name) &&
        ((*root_params_map_1).count(param_name.substr(kPreLenFifteen)) != 0)) {
      (*root_params_map)[(*root_params_map_1)[param_name.substr(kPreLenFifteen)]].emplace_back(param_node);
    } else if ((param_name.substr(0, kPreLenEighteen) == exp_insta_row_name ||
                param_name.substr(0, kPreLenEighteen) == exp_insta_col_name) &&
               ((*root_params_map_1).count(param_name.substr(kPreLenEighteen)) != 0)) {
      (*root_params_map)[(*root_params_map_1)[param_name.substr(kPreLenEighteen)]].emplace_back(param_node);
    } else if (param_name.substr(0, kPreLenEleven) == exp_avg_name &&
               (*root_params_map_1).count(param_name.substr(kPreLenEleven)) != 0) {
      (*root_params_map)[(*root_params_map_1)[param_name.substr(kPreLenEleven)]].emplace_back(param_node);
    }
  }
}

void HandleCameAndAdaFactorOpt(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes,
                               const FuncGraphManagerPtr &manager) {
  MS_LOG(INFO) << "Adafactor or Came optimizer process start";
  MS_EXCEPTION_IF_NULL(root);
  std::set<AnfNodePtr> origin_params;
  auto root_params = root->parameters();
  std::string exp_row_name = EXP_AVG_SQ_ROW;
  std::string exp_col_name = EXP_AVG_SQ_COL;
  std::string exp_insta_row_name = EXP_AVG_INSTA_ROW;
  std::string exp_insta_col_name = EXP_AVG_INSTA_COL;
  std::string exp_avg_name = EXP_AVG_SQ;

  std::unordered_map<string, AnfNodePtr> root_params_map_1;
  std::unordered_map<AnfNodePtr, AnfNodePtrList> root_params_map;

  FindParamNodes(&root_params_map_1, &root_params_map, root_params);

  for (auto &param_pair : root_params_map) {
    auto param_node = param_pair.first;
    if (ParallelContext::GetInstance()->get_redundancy_node().find(param_node) !=
        ParallelContext::GetInstance()->get_redundancy_node().end()) {
      continue;
    }
    for (auto &row_col_node : param_pair.second) {
      MS_EXCEPTION_IF_NULL(param_node);
      auto param = param_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(param);
      std::string param_name = param->name();
      MS_EXCEPTION_IF_NULL(row_col_node);
      auto row_col_param = row_col_node->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(row_col_param);
      std::string row_col_param_name = row_col_param->name();
      origin_params.insert(param_node);
      auto tensor_layout = param->user_data<TensorLayout>();
      MS_EXCEPTION_IF_NULL(tensor_layout);
      auto slice_shape = tensor_layout->slice_shape().array();
      Shape opt_shard_slice_shape = slice_shape;
      if (!tensor_layout->opt_shard_group().empty()) {
        opt_shard_slice_shape = tensor_layout->opt_shard_slice_shape();
      }

      auto shape_size = slice_shape.size();
      bool is_row_or_col_param = row_col_param_name != (exp_avg_name + param_name);
      if (is_row_or_col_param && shape_size <= 1) {
        continue;
      }
      if (row_col_param_name == (exp_avg_name + param_name) && shape_size != 1) {
        continue;
      }

      auto origin_shape = tensor_layout->tensor_shape().array();
      auto dev_mat = tensor_layout->device_arrangement().array();
      auto tensor_map = tensor_layout->tensor_map().array();

      if (row_col_param_name == (exp_row_name + param_name) ||
          row_col_param_name == (exp_insta_row_name + param_name)) {
        opt_shard_slice_shape.pop_back();
        origin_shape.pop_back();
        tensor_map.pop_back();
      } else if (row_col_param_name == (exp_col_name + param_name) ||
                 row_col_param_name == (exp_insta_col_name + param_name)) {
        (void)opt_shard_slice_shape.erase(opt_shard_slice_shape.cbegin() +
                                          static_cast<different_type>(SECOND_FROM_END(shape_size)));
        (void)origin_shape.erase(origin_shape.cbegin() + static_cast<different_type>(SECOND_FROM_END(shape_size)));
        (void)tensor_map.erase(tensor_map.cbegin() + static_cast<different_type>(SECOND_FROM_END(shape_size)));
      }

      TensorLayout new_tensor_layout;
      if (new_tensor_layout.InitFromVector(dev_mat, tensor_map, origin_shape) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Init tensor layout failed";
      }

      if (AdafactorStateIsOptShard(tensor_layout->opt_shard_group(), shape_size, param_name, row_col_param_name)) {
        new_tensor_layout.set_opt_shard_group(tensor_layout->opt_shard_group());
        new_tensor_layout.set_opt_shard_slice_shape(opt_shard_slice_shape);
      }
      SetParamInfoSaveStrategy(row_col_param);
      auto cloned_abstract = row_col_node->abstract()->Clone();
      MS_EXCEPTION_IF_NULL(cloned_abstract);
      std::shared_ptr<abstract::BaseShape> parallel_shape = std::make_shared<abstract::Shape>(opt_shard_slice_shape);
      MS_EXCEPTION_IF_NULL(parallel_shape);
      cloned_abstract->set_shape(parallel_shape);
      row_col_param->set_user_data<TensorLayout>(std::make_shared<TensorLayout>(new_tensor_layout));
      row_col_node->set_abstract(cloned_abstract);
    }
  }

  const auto &node_users_map = manager->node_users();
  for (const auto &origin_param_node : origin_params) {
    if (ParallelContext::GetInstance()->get_redundancy_node().find(origin_param_node) !=
        ParallelContext::GetInstance()->get_redundancy_node().end()) {
      continue;
    }
    auto inserter = CameCommHandler(origin_param_node->cast<ParameterPtr>(), root_params, node_users_map);
    inserter.Process();
  }
}

static std::shared_ptr<TensorLayout> GenerateTensorLayoutForParamReshapeWithStra(const AnfNodePtr &node,
                                                                                 const Dimensions input_stra) {
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  MS_EXCEPTION_IF_ZERO("dev_num", dev_num);

  Shapes inputs_shape = GetNodeShape(node);
  Shape param_shape = inputs_shape[0];

  Shape param_dev_matrix_shape(input_stra.size() + 1, 0);
  for (size_t i = param_dev_matrix_shape.size() - 1; i > 0; i--) {
    param_dev_matrix_shape[i] = input_stra[i - 1];
  }
  param_dev_matrix_shape[0] =
    dev_num / std::accumulate(input_stra.begin(), input_stra.end(), 1, std::multiplies<int64_t>());

  TensorMap param_tensor_map;
  for (size_t i = 0; i < param_shape.size(); ++i) {
    param_tensor_map.push_back(static_cast<int64_t>(param_shape.size() - i - 1));
  }

  TensorLayout param_layout;

  if (param_layout.InitFromVector(param_dev_matrix_shape, param_tensor_map, param_shape) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Infer param-Reshape with strategy tensor layout failed.";
  }

  return std::make_shared<TensorLayout>(param_layout);
}

static AnfNodePtr FindParameterByCallNode(const CNodePtr &call, int64_t index) {
  MS_EXCEPTION_IF_NULL(call);
  AnfNodePtr graph_value_node = call->input(0);
  if (!IsValueNode<FuncGraph>(graph_value_node)) {
    return nullptr;
  }
  auto graph_sub = GetValueNode<FuncGraphPtr>(graph_value_node);
  MS_EXCEPTION_IF_NULL(graph_sub);
  auto parameters = graph_sub->parameters();
  if (LongToSize(index - 1) >= parameters.size()) {
    MS_LOG(EXCEPTION) << "The index is out of range, index is: " << (index - 1) << ", vector size is "
                      << parameters.size();
  }
  return parameters[LongToSize(index - 1)];
}

static std::shared_ptr<TensorLayout> FindParameterNextLayout(const AnfNodePtr &node, size_t curr_depth) {
  if (curr_depth > MAX_RECURSIVE_DEPTH) {
    MS_LOG(WARNING) << "When finding the next tensor layout for the parameter, exceeded the maximum recursion depth: "
                    << MAX_RECURSIVE_DEPTH;
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphManagerPtr manager = node->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodeIndexSet node_set = manager->node_users()[node];
  for (auto &node_pair : node_set) {
    if (IsPrimitiveCNode(node_pair.first, prim::kPrimLoad)) {
      auto layout_param = FindParameterNextLayout(node_pair.first, ++curr_depth);
      if (!layout_param) {
        continue;
      }
      return layout_param;
    }
    CNodePtr use_apply = node_pair.first->cast<CNodePtr>();
    if (use_apply == nullptr) {
      continue;
    }
    auto op = use_apply->input(0);
    MS_EXCEPTION_IF_NULL(op);
    if (IsValueNode<FuncGraph>(op)) {
      auto fg = GetValueNode<FuncGraphPtr>(op);
      auto para = FindParameterByCallNode(use_apply, node_pair.second);
      auto layout_param = FindParameterNextLayout(para, ++curr_depth);
      if (!layout_param) {
        continue;
      }
      return layout_param;
    }
    if (!IsValueNode<Primitive>(use_apply->input(0))) {
      continue;
    }
    ValueNodePtr prim_anf_node = use_apply->input(0)->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(prim_anf_node);
    PrimitivePtr node_prim = prim_anf_node->value()->cast<PrimitivePtr>();
    MS_EXCEPTION_IF_NULL(node_prim);
    if (node_prim->name() == DEPEND && node_pair.second != 1) {
      continue;
    }
    if (node_prim->name() == RESHAPE) {
      auto attrs_temp = node_prim->attrs();
      if (!StrategyFound(attrs_temp)) {
        continue;
      }
      StrategyPtr strategy = ExtractStrategy(attrs_temp[IN_STRATEGY]);
      MS_EXCEPTION_IF_NULL(strategy);
      Strategies stra = strategy->GetInputDim();
      Dimensions input_strategy = stra.at(0);

      auto param_layout = GenerateTensorLayoutForParamReshapeWithStra(node, input_strategy);

      return param_layout;
    }
    if (IsParallelCareNode(use_apply) && use_apply->has_user_data<OperatorInfo>()) {
      auto layout = GetInputLayoutFromCNode(node_pair, -1);
      return std::make_shared<TensorLayout>(layout);
    }
  }
  return nullptr;
}

std::shared_ptr<TensorLayout> CreateParameterLayout(const AnfNodePtr &node) {
  // Create DataParallel tensor layout for parameter(support WideDeep).
  auto next_layout = FindParameterNextLayout(node, 0);
  if (next_layout != nullptr) {
    return next_layout;
  }
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  MS_EXCEPTION_IF_ZERO("dev_num", dev_num);
  TensorLayout input_tensor_layout;
  // create input_shape
  Shapes inputs_shape = GetNodeShape(node);
  Shape input_shape_array = inputs_shape[0];

  // create dev_matrix
  Shape dev_matrix_array = {dev_num};

  // create tensor_map
  size_t shape_size = input_shape_array.size();
  TensorMap input_tensor_map_array(shape_size, MAP_NONE);
  if ((shape_size > 0) && (input_shape_array[0] % dev_num == 0)) {
    input_tensor_map_array[0] = 0;  // shard parameter's first dimension when parameter->Reshape->Op
  }

  if (input_tensor_layout.InitFromVector(dev_matrix_array, input_tensor_map_array, input_shape_array) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Create tensor layout for parameter failed.";
  }
  return std::make_shared<TensorLayout>(input_tensor_layout);
}
}  // namespace parallel
}  // namespace mindspore
