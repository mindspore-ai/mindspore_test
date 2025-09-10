/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_OFFLOAD_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_OFFLOAD_H_

#include "ir/graph_utils.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"

namespace mindspore {
namespace opt {
namespace irpass {
constexpr size_t kReturnIndex = 1;

class OffLoadFuncGraph {
 public:
  OffLoadFuncGraph(const FuncGraphPtr &offload_graph) : offload_graph_(offload_graph) {}
  ~OffLoadFuncGraph() = default;

  void GetPackFunc();
  void GetUnPackFunc();
  void GetPrefetch();

  FuncGraphPtr GetBpropFuncGraph();
  void InsertNodesPreprocess();
  void InsertOffloadNodes();

  CNodePtr NewPackCaller(const CNodePtr &cnode);
  CNodePtr NewUnPackCaller(const CNodePtr &pack_caller);
  void AddAttrForUnpackFuncNodes();

  void ClearOffloadFlags();
  void AddNotCheckFlags();

  FuncGraphManagerPtr manager_;
  FuncGraphPtr offload_graph_;
  FuncGraphPtr pack_func_;
  FuncGraphPtr unpack_func_;
  int64_t prefetch_;

  std::set<CNodePtr> need_insert_forward_nodes_;
  std::map<CNodePtr, std::set<std::pair<AnfNodePtr, int>>> need_replace_backward_nodes_;
};

bool CheckFuncGraphOffLoadFlag(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (func_graph->has_attr(FUNC_GRAPH_FLAG_PACK_FN_GRAD)) {
    MS_LOG(DEBUG) << "The func_graph has pack_fn flag: " << func_graph->ToString();
    if (func_graph->has_attr(FUNC_GRAPH_FLAG_UNPACK_FN_GRAD)) {
      MS_LOG(DEBUG) << "The func_graph has unpack_fn flag: " << func_graph->ToString();
      if (func_graph->has_attr(FUNC_GRAPH_FLAG_PREFETCH_GRAD)) {
        MS_LOG(DEBUG) << "The func_graph has count flag: " << func_graph->ToString();
        return true;
      } else {
        MS_LOG(WARNING) << "The func_graph has not count flag: " << func_graph->ToString();
      }
    } else {
      MS_LOG(WARNING) << "The func_graph has not unpack_fn flag: " << func_graph->ToString();
    }
  }
  return false;
}

void OffLoadFuncGraph::GetPackFunc() {
  MS_EXCEPTION_IF_NULL(offload_graph_);
  auto pack_fn_value = offload_graph_->get_attr(FUNC_GRAPH_FLAG_PACK_FN_GRAD);
  MS_EXCEPTION_IF_NULL(pack_fn_value);
  MS_LOG(DEBUG) << "pack_fn_value: " << pack_fn_value->ToString();
  if (pack_fn_value->isa<parse::InterpretedObject>()) {
    const auto &interpreted_value = dyn_cast<parse::InterpretedObject>(pack_fn_value);
    ValuePtr converted_value = nullptr;
    if (!parse::ConvertData(interpreted_value->obj(), &converted_value)) {
      MS_LOG(EXCEPTION) << "Convert data failed";
    }
    MS_EXCEPTION_IF_NULL(converted_value);
    MS_LOG(DEBUG) << "converted_value: " << converted_value->ToString();
    pack_func_ = converted_value->cast<FuncGraphPtr>();
    MS_EXCEPTION_IF_NULL(pack_func_);
    MS_LOG(DEBUG) << "pack_func_: " << pack_func_->ToString();
  }
}

void OffLoadFuncGraph::GetUnPackFunc() {
  MS_EXCEPTION_IF_NULL(offload_graph_);
  auto unpack_fn_value = offload_graph_->get_attr(FUNC_GRAPH_FLAG_UNPACK_FN_GRAD);
  MS_EXCEPTION_IF_NULL(unpack_fn_value);
  MS_LOG(DEBUG) << "unpack_fn_value: " << unpack_fn_value->ToString();
  if (unpack_fn_value->isa<parse::InterpretedObject>()) {
    const auto &interpreted_value = dyn_cast<parse::InterpretedObject>(unpack_fn_value);
    ValuePtr converted_value = nullptr;
    if (!parse::ConvertData(interpreted_value->obj(), &converted_value)) {
      MS_LOG(EXCEPTION) << "Convert data failed";
    }
    MS_EXCEPTION_IF_NULL(converted_value);
    MS_LOG(DEBUG) << "converted_value: " << converted_value->ToString();
    unpack_func_ = converted_value->cast<FuncGraphPtr>();
    MS_EXCEPTION_IF_NULL(unpack_func_);
    MS_LOG(DEBUG) << "unpack_func_: " << unpack_func_->ToString();
  }
}

void OffLoadFuncGraph::GetPrefetch() {
  MS_EXCEPTION_IF_NULL(offload_graph_);
  auto count_value = offload_graph_->get_attr(FUNC_GRAPH_FLAG_PREFETCH_GRAD);
  MS_EXCEPTION_IF_NULL(count_value);
  MS_LOG(DEBUG) << "count_value: " << count_value->ToString();
  if (count_value->isa<parse::InterpretedObject>()) {
    const auto &interpreted_value = dyn_cast<parse::InterpretedObject>(count_value);
    prefetch_ = py::cast<int64_t>(interpreted_value->obj());
    MS_LOG(DEBUG) << "prefetch_: " << prefetch_;
  }
}

FuncGraphPtr OffLoadFuncGraph::GetBpropFuncGraph() {
  const auto &return_node = offload_graph_->get_return();
  if (return_node == nullptr) {
    return nullptr;
  }
  const auto &return_input = return_node->input(kReturnIndex);
  if (return_input == nullptr || !return_input->isa<CNode>()) {
    return nullptr;
  }
  auto make_tuple = return_input->cast<CNodePtr>();
  if (!IsPrimitiveCNode(return_input, prim::kPrimMakeTuple)) {
    return nullptr;
  }
  constexpr size_t bprop_index = 2;
  auto bprop_func_node = make_tuple->input(bprop_index);
  if (!IsValueNode<FuncGraph>(bprop_func_node)) {
    return nullptr;
  }
  FuncGraphPtr bprop_func = GetValueNode<FuncGraphPtr>(bprop_func_node);
  return bprop_func;
}

void OffLoadFuncGraph::InsertNodesPreprocess() {
  auto bprop_func = GetBpropFuncGraph();
  if (bprop_func == nullptr) {
    return;
  }
  const auto &all_nodes = TopoSort(offload_graph_->return_node());
  for (auto &node : all_nodes) {
    MS_LOG(DEBUG) << "node: " << node->DebugString();
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_LOG(DEBUG) << "cnode: " << node->DebugString();
    auto &node_users = manager_->node_users();
    auto iter = node_users.find(cnode);
    if (iter == node_users.end()) {
      continue;
    }
    for (auto &user : iter->second) {
      auto &user_node = user.first;
      std::set<std::pair<AnfNodePtr, int>> users_in_bprop;
      if (user_node->func_graph() == bprop_func) {
        MS_LOG(DEBUG) << "cnode: " << node->DebugString();
        MS_LOG(DEBUG) << "cnode user: " << user_node->DebugString() << " user.second:" << user.second;
        need_insert_forward_nodes_.insert(cnode);
        users_in_bprop.insert(std::make_pair(user_node, user.second));
      }
      MS_LOG(DEBUG) << "users_in_bprop size: " << users_in_bprop.size();
      if (!users_in_bprop.empty()) {
        need_replace_backward_nodes_[cnode] = users_in_bprop;
      }
    }
  }
  MS_LOG(DEBUG) << "need_insert_forward_nodes_ size: " << need_insert_forward_nodes_.size();
  MS_LOG(DEBUG) << "need_replace_backward_nodes_ size: " << need_replace_backward_nodes_.size();
}

CNodePtr OffLoadFuncGraph::NewPackCaller(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(pack_func_);
  std::vector<AnfNodePtr> caller_inputs{NewValueNode(pack_func_), cnode};
  auto cur_dunc = cnode->func_graph();
  MS_EXCEPTION_IF_NULL(cur_dunc);
  auto caller = cur_dunc->NewCNode(caller_inputs);
  MS_LOG(DEBUG) << "pack caller: " << caller->DebugString();
  return caller;
}

CNodePtr OffLoadFuncGraph::NewUnPackCaller(const CNodePtr &pack_caller) {
  MS_EXCEPTION_IF_NULL(unpack_func_);
  std::vector<AnfNodePtr> caller_inputs{NewValueNode(unpack_func_), pack_caller};
  auto cur_dunc = pack_caller->func_graph();
  MS_EXCEPTION_IF_NULL(cur_dunc);
  auto caller = cur_dunc->NewCNode(caller_inputs);
  MS_LOG(DEBUG) << "unpack caller: " << caller->DebugString();
  return caller;
}

void OffLoadFuncGraph::InsertOffloadNodes() {
  for (auto cnode : need_insert_forward_nodes_) {
    MS_LOG(DEBUG) << "cnode: " << cnode->DebugString();
    auto pack_caller = NewPackCaller(cnode);
    auto unpack_caller = NewUnPackCaller(pack_caller);
    auto node_user_in_bprop_info = need_replace_backward_nodes_[cnode];
    MS_LOG(DEBUG) << "node_user_in_bprop_info size: " << node_user_in_bprop_info.size();
    for (auto info : node_user_in_bprop_info) {
      auto node_user = info.first;
      auto index = info.second;
      MS_LOG(DEBUG) << "node_user: " << node_user->DebugString();
      MS_LOG(DEBUG) << "index: " << index;
      manager_->SetEdge(node_user, index, unpack_caller);
    }
    std::vector<AnfNodePtr> depend_pack_inputs{NewValueNode(prim::kPrimDepend), cnode, pack_caller};
    auto cur_dunc = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cur_dunc);
    auto depend_pack = cur_dunc->NewCNode(depend_pack_inputs);
    MS_LOG(DEBUG) << "depend_pack: " << depend_pack->DebugString();
    auto &node_users = manager_->node_users();
    auto iter = node_users.find(cnode);
    if (iter == node_users.end()) {
      continue;
    }
    for (auto &user : iter->second) {
      auto &user_node = user.first;
      auto fprop_user_index = user.second;
      MS_LOG(DEBUG) << "user_node: " << user_node->DebugString() << " fprop_user_index: " << fprop_user_index;
      if (user_node != depend_pack && user_node != pack_caller) {
        if (cur_dunc == user_node->func_graph()) {
          user_node->cast<CNodePtr>()->set_input(fprop_user_index, depend_pack);
        }
      }
    }
  }
}

void OffLoadFuncGraph::AddAttrForUnpackFuncNodes() {
  MS_EXCEPTION_IF_NULL(unpack_func_);
  const auto &all_nodes = TopoSort(unpack_func_->return_node());
  for (auto &node : all_nodes) {
    MS_LOG(DEBUG) << "node: " << node->DebugString();
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    MS_LOG(DEBUG) << "Add prefetch_ attribute for node: " << cnode->DebugString();
    cnode->AddAttr("count", MakeValue<int64_t>(prefetch_));
  }
}

void OffLoadFuncGraph::ClearOffloadFlags() {
  MS_EXCEPTION_IF_NULL(offload_graph_);
  offload_graph_->erase_flag(FUNC_GRAPH_FLAG_NO_INLINE);
  offload_graph_->erase_flag(FUNC_GRAPH_FLAG_PACK_FN_GRAD);
  offload_graph_->erase_flag(FUNC_GRAPH_FLAG_UNPACK_FN_GRAD);
  offload_graph_->erase_flag(FUNC_GRAPH_FLAG_PREFETCH_GRAD);
}

void OffLoadFuncGraph::AddNotCheckFlags() {
  MS_EXCEPTION_IF_NULL(offload_graph_);
  offload_graph_->set_flag(FUNC_GRAPH_FLAG_NOT_CHECK, true);
}

bool OffLoad(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto all_func_graphs = root->func_graphs_used_total();
  for (auto &fg : all_func_graphs) {
    MS_EXCEPTION_IF_NULL(fg);
    bool is_offload_func = CheckFuncGraphOffLoadFlag(fg);
    MS_LOG(DEBUG) << "The current graph needs offload processing: " << fg->ToString();
    if (is_offload_func) {
      OffLoadFuncGraph off_load_func(fg);
      off_load_func.manager_ = manager;

      // Process the offload information(pack_fn, unpack_fn, prefetch)
      off_load_func.GetPackFunc();
      off_load_func.GetUnPackFunc();
      off_load_func.GetPrefetch();

      // Get the forward nodes and backward nodes.
      off_load_func.InsertNodesPreprocess();

      // Insert the offload nodes for the forward nodes and backward nodes.
      // (pack_func callers, unpack_func callers and depend nodes)
      off_load_func.InsertOffloadNodes();

      // Add prefetch(count) attr for cnodes of unpack_func.
      off_load_func.AddAttrForUnpackFuncNodes();

      // Clear flag for offload_func.
      off_load_func.ClearOffloadFlags();

      // Add not check flag for offload_func.
      off_load_func.AddNotCheckFlags();
    }
  }
  return false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_OFFLOAD_H_
