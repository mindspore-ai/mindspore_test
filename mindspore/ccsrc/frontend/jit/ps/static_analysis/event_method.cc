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

#include "frontend/jit/ps/static_analysis/event_method.h"

#include <vector>
#include <string>
#include "mindspore/ops/op_def/framework_ops.h"
#include "utils/anf_utils.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "include/common/utils/anfalgo.h"
#include "frontend/jit/ps/debug/trace.h"

namespace mindspore {
namespace pipeline {
mindspore::HashMap<uint32_t, std::vector<AnfNodePtr>> event_method_nodes;

void PreprocessForEventMethod(const FuncGraphPtr &func_graph) {
  if (func_graph->has_flag("PROCESS_EVENT")) {
    return;
  }
  func_graph->set_flag("PROCESS_EVENT", true);
  auto nodes = func_graph->order_list();
  for (auto &weak_cnode : nodes) {
    const auto &cnode = weak_cnode.lock();
    MS_LOG(DEBUG) << "cnode: " << cnode->DebugString();
    if (IsPrimitiveCNode(cnode, prim::kPrimStreamSend) || IsPrimitiveCNode(cnode, prim::kPrimStreamRecv)) {
      auto event_method_node = cnode->cast<CNodePtr>();
      auto input = event_method_node->input(1);
      while (IsPrimitiveCNode(input, prim::kPrimDepend)) {
        input = input->cast<CNodePtr>()->input(1);
      }
      auto event_value = GetValueNode<EventPtr>(input);
      MS_EXCEPTION_IF_NULL(event_value);
      auto event_id = event_value->value();
      event_method_nodes[event_id].emplace_back(cnode);
      common::AnfAlgo::SetNodeAttrSafely(kAttrEventId, MakeValue(static_cast<uint32_t>(event_id)), cnode);
    }
  }
  for (auto &fg : func_graph->func_graphs_used_total()) {
    PreprocessForEventMethod(fg);
  }
}

void CheckAndReplace(const FuncGraphPtr &func_graph) {
  for (auto iter : event_method_nodes) {
    auto event_id = iter.first;
    MS_LOG(DEBUG) << "The id of event: " << event_id;
    auto cur_event_method_nodes = iter.second;
    if (cur_event_method_nodes.size() % 2 != 0) {
      MS_LOG(EXCEPTION) << "Incorrect use of event, the id of event: " << event_id;
    }
    // Check: record, wait is true; wait, record is wrong.
    for (size_t i = 0; i < cur_event_method_nodes.size(); ++i) {
      auto event_node = cur_event_method_nodes[i];
      if ((i % 2 == 0 && !IsPrimitiveCNode(event_node, prim::kPrimStreamSend)) ||
          (i % 2 == 1 && !IsPrimitiveCNode(event_node, prim::kPrimStreamRecv))) {
        MS_LOG(EXCEPTION) << "Incorrect use of event, the id of event: " << event_id
                          << "the event node:" << event_node->DebugString()
                          << ", location:" << trace::GetDebugInfoStr(event_node->debug_info());
      }
    }
    // %0 = StreamSendInner(event1)
    // %1 = StreamRecvInner(event1)
    // After Replace
    // %0 = StreamSendInner(event1)
    // %1 = StreamRecvInner(%0)
    for (size_t index = cur_event_method_nodes.size() - 1; index > 0; --index) {
      auto after_node = cur_event_method_nodes[index]->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(after_node);
      const auto &before_node = cur_event_method_nodes[index - 1];
      MS_EXCEPTION_IF_NULL(before_node);
      after_node->set_input(1, before_node);
    }
  }
}

void ClearEventFuncFlag(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!func_graph->has_flag("PROCESS_EVENT")) {
    return;
  }
  func_graph->erase_flag("PROCESS_EVENT");
  for (auto &fg : func_graph->func_graphs_used_total()) {
    fg->erase_flag("PROCESS_EVENT");
  }
}

void EventMethod(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->manager());
  PreprocessForEventMethod(func_graph);
  CheckAndReplace(func_graph);
  ClearEventFuncFlag(func_graph);
}
}  // namespace pipeline
}  // namespace mindspore
