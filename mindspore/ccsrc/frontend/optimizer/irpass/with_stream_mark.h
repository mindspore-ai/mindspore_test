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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_WITH_STREAM_MARK_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_WITH_STREAM_MARK_H_

#include "ir/graph_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {
int64_t GetStreamIdFuncGraphWithStreamCtx(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto value = func_graph->get_attr(kFuncGraphFlagStreamId);
  if (value != nullptr) {
    const auto &stream_id = GetValue<size_t>(value);
    MS_LOG(DEBUG) << "stream_id:" << stream_id << " func_graph:" << func_graph->ToString();
    auto fg_used_total = func_graph->func_graphs_used_total();
    for (const auto &fg : fg_used_total) {
      auto stream_limit_id_value = fg->get_attr(kFuncGraphFlagStreamLimitId);
      if (stream_limit_id_value != nullptr) {
        MS_LOG(DEBUG) << "Pass labels to the subgraph: " << fg->ToString();
        fg->set_attr(kFuncGraphFlagStreamId, MakeValue(stream_id));
      }
    }
    return stream_id;
  }
  return -1;
}

bool CheckNeedMark(const CNodePtr &cnode, size_t stream_id) {
  if (IsPrimitiveCNode(cnode, prim::kPrimDepend)) {
    auto need_check_node = cnode->input(1);
    if (!need_check_node->isa<CNode>()) {
      return false;
    }
    return CheckNeedMark(need_check_node->cast<CNodePtr>(), stream_id);
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimReturn)) {
    auto need_check_node = cnode->input(1);
    if (!need_check_node->isa<CNode>()) {
      return false;
    }
    return CheckNeedMark(need_check_node->cast<CNodePtr>(), stream_id);
  }
  return true;
}

void MarkWithStreamCtx(const FuncGraphPtr &func) {
  if (func->has_attr("marked_stream_ctx")) {
    return;
  }
  func->set_flag("marked_stream_ctx", true);
  const auto &all_nodes = TopoSort(func->return_node(), SuccDeeperSimple, AlwaysInclude);
  MS_LOG(DEBUG) << "all_nodes size: " << all_nodes.size();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto cur_func = cnode->func_graph();
    int64_t stream_id = GetStreamIdFuncGraphWithStreamCtx(cur_func);
    auto need_mark = (stream_id != -1) && CheckNeedMark(cnode, stream_id);
    if (need_mark) {
      MS_LOG(DEBUG) << "The cnode need mark: " << cnode->DebugString() << " need_mark:" << need_mark;
      cnode->AddAttr(kFuncGraphFlagStreamId, MakeValue(static_cast<int64_t>(stream_id)));
    } else {
      MS_LOG(DEBUG) << "The cnode do not need mark: " << cnode->DebugString() << " need_mark:" << need_mark;
    }
    // call node
    auto cnode_input_0 = cnode->input(0);
    if (IsValueNode<FuncGraph>(cnode_input_0)) {
      auto sub_func = GetValueNode<FuncGraphPtr>(cnode_input_0);
      if (!sub_func->get_attr(kFuncGraphFlagStreamCtxAfter)) {
        MarkWithStreamCtx(sub_func);
      }
    }
  }
}

int64_t GetStreamLimitId(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto value = func_graph->get_attr(kFuncGraphFlagStreamLimitId);
  if (value != nullptr) {
    const auto &stream_limit_id = GetValue<size_t>(value);
    return stream_limit_id;
  }
  return -1;
}

void MarkWithStreamLimitCtx(const FuncGraphPtr &func) {
  if (func->has_attr("marked_stream_limit_ctx")) {
    return;
  }
  func->set_flag("marked_stream_limit_ctx", true);
  const auto &all_nodes = TopoSort(func->return_node(), SuccDeeperSimple, AlwaysInclude);
  MS_LOG(DEBUG) << "all_nodes size: " << all_nodes.size();
  for (auto &node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto cur_func = cnode->func_graph();
    int64_t stream_limit_id = GetStreamLimitId(cur_func);
    int64_t stream_id = -1;
    if (cnode->HasAttr(kFuncGraphFlagStreamId)) {
      stream_id = GetValue<int64_t>(cnode->GetAttr(kFuncGraphFlagStreamId));
    }
    MS_LOG(DEBUG) << "stream_limit_id: " << stream_limit_id << " stream_id: " << stream_id;
    if (stream_limit_id != -1 && stream_id != -1 && stream_limit_id == stream_id) {
      auto cube_num_value = cur_func->get_attr(kFuncGraphFlagCubeNum);
      MS_EXCEPTION_IF_NULL(cube_num_value);
      int64_t cube_num = GetValue<int64_t>(cube_num_value);
      auto vector_num_value = cur_func->get_attr(kFuncGraphFlagVectorNum);
      MS_EXCEPTION_IF_NULL(vector_num_value);
      int64_t vector_num = GetValue<int64_t>(vector_num_value);
      cnode->AddAttr(kFuncGraphFlagCubeNum, MakeValue(static_cast<int64_t>(cube_num)));
      cnode->AddAttr(kFuncGraphFlagVectorNum, MakeValue(static_cast<int64_t>(vector_num)));
      // call node
      auto cnode_input_0 = cnode->input(0);
      if (IsValueNode<FuncGraph>(cnode_input_0)) {
        auto sub_func = GetValueNode<FuncGraphPtr>(cnode_input_0);
        if (!sub_func->get_attr(kFuncGraphFlagStreamLimitCtxAfter)) {
          MarkWithStreamLimitCtx(sub_func);
        }
      }
    }
  }
}

bool WithStreamMark(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_LOG(DEBUG) << "root fg: " << root->ToString();
  MarkWithStreamCtx(root);
  MarkWithStreamLimitCtx(root);

  auto all_func_graphs = root->func_graphs_used_total();
  for (auto &fg : all_func_graphs) {
    MS_EXCEPTION_IF_NULL(fg);
    bool is_with_stream_func = (GetStreamIdFuncGraphWithStreamCtx(fg) != -1);
    bool is_with_stream_after_func = fg->has_flag(kFuncGraphFlagStreamCtxAfter);
    if (is_with_stream_func || is_with_stream_after_func) {
      fg->erase_flag(FUNC_GRAPH_FLAG_NO_INLINE);
      fg->erase_flag(kFuncGraphFlagStreamId);
      fg->erase_flag(kFuncGraphFlagStreamLimitId);
      fg->erase_flag(kFuncGraphFlagCubeNum);
      fg->erase_flag(kFuncGraphFlagVectorNum);
      fg->erase_flag(kFuncGraphFlagStreamCtxAfter);
      fg->erase_flag(kFuncGraphFlagStreamLimitCtxAfter);
      fg->erase_flag("marked_stream_ctx");
      fg->erase_flag("marked_stream_limit_ctx");
    }
  }
  root->erase_flag("marked_stream_ctx");
  root->erase_flag("marked_stream_limit_ctx");
  return false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_WITH_STREAM_MARK_H_
