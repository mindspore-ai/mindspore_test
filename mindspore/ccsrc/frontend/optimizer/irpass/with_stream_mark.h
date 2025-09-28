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
  auto value = func_graph->get_attr(FUNC_GRAPH_FLAG_NO_INLINE_WITH_STREAM_CTX);
  if (value != nullptr && value->isa<Int64Imm>()) {
    const auto &stream_id = GetValue<int64_t>(value);
    return stream_id;
  }
  return -1;
}

int64_t GetStreamIdFuncGraphWithStreamCtxAfter(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto value = func_graph->get_attr(FUNC_GRAPH_FLAG_NO_INLINE_WITH_STREAM_CTX_AFTER);
  if (value != nullptr && value->isa<Int64Imm>()) {
    const auto &stream_id = GetValue<int64_t>(value);
    return stream_id;
  }
  return -1;
}

bool CheckNeedMark(const CNodePtr &cnode, int64_t stream_id) {
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
  auto func_caller = GetValueNode<FuncGraphPtr>(cnode->input(0));
  if (func_caller != nullptr) {
    auto cur_node_stream_id = GetStreamIdFuncGraphWithStreamCtxAfter(func_caller);
    if (cur_node_stream_id != -1 && cur_node_stream_id == stream_id) {
      MS_LOG(DEBUG) << "Do not mark";
      return false;
    }
  }
  return true;
}

bool WithStreamMark(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(opt);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_LOG(DEBUG) << "root fg: " << root->ToString();

  const auto &all_nodes = TopoSort(root->return_node(), SuccDeeperSimple, AlwaysInclude);
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
      cnode->AddAttr("stream_name", MakeValue(static_cast<int64_t>(stream_id)));
    } else {
      MS_LOG(DEBUG) << "The cnode do not need mark: " << cnode->DebugString() << " need_mark:" << need_mark;
    }
  }

  auto all_func_graphs = root->func_graphs_used_total();
  for (auto &fg : all_func_graphs) {
    MS_EXCEPTION_IF_NULL(fg);
    bool is_with_stream_after_func = (GetStreamIdFuncGraphWithStreamCtxAfter(fg) != -1);
    if (is_with_stream_after_func) {
      MS_LOG(DEBUG) << "is_with_stream_after_func fg: " << fg->ToString();
      fg->erase_flag(FUNC_GRAPH_FLAG_NO_INLINE_WITH_STREAM_CTX_AFTER);
      fg->erase_flag(FUNC_GRAPH_FLAG_NO_INLINE);
    }
    bool is_with_stream_func = (GetStreamIdFuncGraphWithStreamCtx(fg) != -1);
    MS_LOG(DEBUG) << "is_with_stream_func: " << is_with_stream_func;
    if (is_with_stream_func) {
      MS_LOG(DEBUG) << "is_with_stream_func fg: " << fg->ToString();
      fg->erase_flag(FUNC_GRAPH_FLAG_NO_INLINE_WITH_STREAM_CTX);
      fg->erase_flag(FUNC_GRAPH_FLAG_NO_INLINE);
    }
  }
  return false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_WITH_STREAM_MARK_H_
