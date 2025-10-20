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
using EventMap = mindspore::HashMap<uint32_t, std::vector<AnfNodePtr>>;
void PreprocessForEventMethod(const FuncGraphPtr &func_graph, EventMap *event_method_nodes) {
  if (func_graph->has_flag("PROCESS_EVENT")) {
    return;
  }
  func_graph->set_flag("PROCESS_EVENT", true);
  auto nodes = func_graph->order_list();
  for (auto &weak_cnode : nodes) {
    const auto &cnode = weak_cnode.lock();
    auto node_abs = cnode->abstract();
    MS_EXCEPTION_IF_NULL(node_abs);
    if (node_abs->isa<abstract::AbstractEvent>()) {
      auto event_method_node = cnode->cast<CNodePtr>();
      const auto &input = event_method_node->input(1);
      const auto &abs = input->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      auto event_abs = abs->cast<abstract::AbstractEventPtr>();
      MS_EXCEPTION_IF_NULL(event_abs);
      auto event_id = event_abs->event_id();
      (*event_method_nodes)[event_id].emplace_back(cnode);
      common::AnfAlgo::SetNodeAttrSafely(kAttrEventId, MakeValue(static_cast<uint32_t>(event_id)), cnode);
    }
  }
  for (auto &fg : func_graph->func_graphs_used_total()) {
    PreprocessForEventMethod(fg, event_method_nodes);
  }
}

void CheckAndReplace(const EventMap &event_method_nodes) {
  for (auto iter : event_method_nodes) {
    auto event_id = iter.first;
    MS_LOG(DEBUG) << "The id of event: " << event_id;
    auto cur_event_method_nodes = iter.second;
    // %0 = StreamSend(event1)
    // %1 = Depend(event1, x)
    // %2 = StreamRecv(event1)
    // After Replace
    // %0 = StreamSend(event1)
    // %1 = Depend(%0, x)
    // %2 = StreamRecv(%1)
    for (size_t index = cur_event_method_nodes.size() - 1; index > 0; --index) {
      auto after_node = cur_event_method_nodes[index]->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(after_node);
      const auto &before_node = cur_event_method_nodes[index - 1];
      MS_EXCEPTION_IF_NULL(before_node);
      auto after_cnode = after_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(after_cnode);
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
  EventMap event_method_nodes;
  PreprocessForEventMethod(func_graph, &event_method_nodes);
  CheckAndReplace(event_method_nodes);
  ClearEventFuncFlag(func_graph);
}
}  // namespace pipeline
}  // namespace mindspore
