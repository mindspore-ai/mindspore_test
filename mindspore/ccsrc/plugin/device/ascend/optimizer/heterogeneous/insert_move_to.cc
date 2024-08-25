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

#include "plugin/device/ascend/optimizer/heterogeneous/insert_move_to.h"

#include "plugin/device/ascend/optimizer/heterogeneous/move_to_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/offload_context.h"
#include "pybind_api/ir/tensor_py.h"

namespace mindspore {
namespace opt {
constexpr auto kMoveToNpuStr = "NPU";
constexpr auto kMoveToCpuStr = "CPU";
constexpr auto kMoveToDiskStr = "Disk";
constexpr auto kParamterDiskUserDataName = "parameter_device";

bool InsertMoveTo::Run(const FuncGraphPtr &graph) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    return false;
  }
  Init(graph);
  // 1. Insert MoveTo and MoveAssign for offloaded parameter.
  bool changed = HandleParameter();

  // 2. Execution order by default
  kernel_graph_->SetExecOrderByDefault();
  return changed;
}

void InsertMoveTo::Init(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  func_graph_ = graph;
  kernel_graph_ = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph_);
}

void InsertMoveTo::CollectOffloadedParameter() {
  const auto &execution_order = kernel_graph_->execution_order();
  for (size_t execution_idx = 0; execution_idx < execution_order.size(); ++execution_idx) {
    auto cnode = execution_order[execution_idx];
    MS_EXCEPTION_IF_NULL(cnode);
    const size_t input_size = common::AnfAlgo::GetInputTensorNum(cnode);
    for (size_t idx = 1; idx <= input_size; ++idx) {
      auto kernel_with_idx = common::AnfAlgo::VisitKernelWithReturnType(cnode->input(idx), 0, true);
      auto input = kernel_with_idx.first;
      if (input == nullptr || !input->isa<Parameter>()) {
        continue;
      }
      auto parameter = input->cast<ParameterPtr>();
      const auto value = parameter->default_param();
      if (value == nullptr) {
        continue;
      }
      const auto meta_tensor = value->cast_ptr<tensor::MetaTensor>();
      if (meta_tensor == nullptr) {
        continue;
      }
      const auto &user_data = meta_tensor->user_data<tensor::TensorPy::TensorPyUserData>(kParamterDiskUserDataName);
      if (user_data == nullptr) {
        continue;
      }
      if (!py::isinstance<py::str>(user_data->obj)) {
        continue;
      }
      std::string device_str = py::cast<std::string>(user_data->obj);
      if (device_str.empty() || device_str == "Npu") {
        continue;
      }
      const auto is_side_effect = common::AnfAlgo::HasNodeAttr(GRAPH_FLAG_SIDE_EFFECT_MEM, cnode) &&
                                  common::AnfAlgo::GetNodeAttr<bool>(cnode, GRAPH_FLAG_SIDE_EFFECT_MEM);
      OffloadParamInfo info{cnode, idx, execution_idx, is_side_effect, device_str};
      MS_LOG(INFO) << "Offloaded parameter is used by " << cnode->fullname_with_scope() << ", input index: " << idx
                   << ", kernel execution order: " << execution_idx << ", side effect: " << is_side_effect;
      offloaded_parameters_[parameter].emplace_back(info);
    }
  }
}

CNodePtr InsertMoveTo::InsertParamMoveTo(const ParameterPtr &parameter, const OffloadParamInfo &info) const {
  MS_EXCEPTION_IF_NULL(parameter);

  // Get control previous and following node.
  const auto pre_load_execution_order_l =
    info.execution_order_ > load_lead_dh_ ? info.execution_order_ - load_lead_dh_ : 0;
  auto pre_node = kernel_graph_->execution_order()[pre_load_execution_order_l];
  if (pre_node == info.user_node_) {
    pre_node = nullptr;
  }
  const auto pre_load_execution_order_r = pre_load_execution_order_l + 1;
  const auto following_node = kernel_graph_->execution_order()[pre_load_execution_order_r];
  MS_EXCEPTION_IF_NULL(following_node);

  const MoveToInfo to_d_info{kMoveToNpuStr, parameter, info.user_node_, info.input_index_, pre_node, following_node};

  auto move_to_d_node = MoveToUtils::InsertMoveTo(kernel_graph_, to_d_info);
  MS_LOG(INFO) << "Add MoveTo node[" << move_to_d_node->DebugString() << "] for " << info.input_index_ << "th input of "
               << info.user_node_->fullname_with_scope() << ".";

  if (info.offload_device_ == kMoveToDiskStr) {
    const auto load_lead = load_lead_dh_ + load_lead_hf_;
    const auto l = info.execution_order_ > load_lead ? info.execution_order_ - load_lead : 0;
    const auto l_node = kernel_graph_->execution_order()[l];
    MS_EXCEPTION_IF_NULL(l_node);
    const auto r = l + 1;
    const auto r_node = kernel_graph_->execution_order()[r];
    MS_EXCEPTION_IF_NULL(r_node);

    const MoveToInfo to_h_info{kMoveToCpuStr, parameter, move_to_d_node, 1, l_node, r_node};

    const auto move_to_h_node = MoveToUtils::InsertMoveTo(kernel_graph_, to_h_info);
    MS_LOG(INFO) << "Add MoveTo node[" << move_to_h_node->DebugString() << "] for " << info.input_index_
                 << "th input of " << info.user_node_->fullname_with_scope() << ".";
  }
  return move_to_d_node;
}

void InsertMoveTo::InsertParamMoveAssign(const ParameterPtr &parameter, const OffloadParamInfo &info,
                                         const CNodePtr &move_to) const {
  MS_EXCEPTION_IF_NULL(parameter);

  auto next_node = kernel_graph_->get_return();
  const auto &execution_order = kernel_graph_->execution_order();
  if (info.execution_order_ + 1 < execution_order.size()) {
    next_node = execution_order[info.execution_order_ + 1];
  }
  MS_EXCEPTION_IF_NULL(next_node);

  const MoveAssignInfo move_assign_info{info.offload_device_.c_str(), parameter, move_to, info.user_node_, next_node};
  const auto &move_assign_node = MoveToUtils::InsertMoveAssign(kernel_graph_, move_assign_info);
  MS_EXCEPTION_IF_NULL(move_assign_node);

  MS_LOG(INFO) << "Add MoveAssign node[" << move_assign_node->DebugString() << "] for " << info.input_index_
               << "th input of " << info.user_node_->fullname_with_scope() << ".";
}

bool InsertMoveTo::HandleParameter() {
  CollectOffloadedParameter();
  if (offloaded_parameters_.empty()) {
    return false;
  }
  if (OffloadContext::GetInstance()->auto_offload() || !OffloadContext::GetInstance()->offload_param().empty()) {
    MS_LOG(EXCEPTION) << "Setting \"CPU\" device for parameter is not supported when \"offload_param\" is not empty"
                      << " string or \"auto_offload\" is True in offload_context";
  }
  OffloadContext::GetInstance()->set_specific_param_offload(true);

  bool changed = false;
  struct MoveToInfo {
    OffloadParamInfo user_;
    CNodePtr move_to_;
    ParameterPtr parameter_;
  };
  std::vector<MoveToInfo> move_assign_to_insert;
  for (const auto &iter : offloaded_parameters_) {
    auto parameter = iter.first;
    MS_EXCEPTION_IF_NULL(parameter);
    auto parameter_abstract = parameter->abstract();
    for (const auto &user : iter.second) {
      auto move_to = InsertParamMoveTo(parameter, user);
      if (user.side_effect_) {
        kernel_graph_->ReplaceRefPair({parameter, 0}, {move_to, 0});
        MoveToInfo move_to_info{user, move_to, parameter};
        move_assign_to_insert.emplace_back(move_to_info);
      }
      changed = true;
    }
  }
  for (const auto &item : move_assign_to_insert) {
    InsertParamMoveAssign(item.parameter_, item.user_, item.move_to_);
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
