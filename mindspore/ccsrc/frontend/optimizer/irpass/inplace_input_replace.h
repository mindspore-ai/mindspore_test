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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INPLACE_INPUT_REPLACE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INPLACE_INPUT_REPLACE_H_

#include <map>
#include <string>
#include <vector>

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
class InplaceInputReplace {
 public:
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
    bool change = ChangeInplaceInputInner(func_graph);
    const auto &sub_graphs = func_graph->func_graphs_used_total();
    for (auto sub_graph : sub_graphs) {
      change = ChangeInplaceInputInner(sub_graph) || change;
    }
    return change;
  }

 private:
  std::vector<int64_t> inplace_input_indexes(const PrimitivePtr &prim) const {
    auto op_def = mindspore::ops::GetOpDef(prim->name());
    std::vector<int64_t> indexes;
    if (op_def != nullptr) {
      // Get inplace_indexes for a primtive defined by yaml.
      size_t output_size = op_def->returns_.size();
      for (size_t index = 0; index < output_size; ++index) {
        auto inplace_index = op_def->returns_[index].inplace_input_index_;
        (void)indexes.emplace_back(inplace_index);
      }
      MS_LOG(DEBUG) << "For Primitive '" << prim->name() << "', the inplace_input_indexes is " << indexes;
      return indexes;
    }
    // Try to get inplace_indexes for a Python primitive.
    auto input_names = prim->GetAttr("input_names");
    auto output_names = prim->GetAttr("output_names");
    if ((input_names == nullptr) || (output_names == nullptr)) {
      return indexes;
    }
    const auto &input_name_list = GetValue<std::vector<std::string>>(input_names);
    std::vector<std::string> output_name_list{};
    if (output_names->isa<StringImm>()) {
      (void)output_name_list.emplace_back(GetValue<std::string>(output_names));
    } else {
      output_name_list = GetValue<std::vector<std::string>>(output_names);
    }
    for (const auto &output : output_name_list) {
      const auto &rw_write_indexes = prim->rw_write_input_indexes();
      auto iter = std::find(input_name_list.begin(), input_name_list.end(), output);
      auto index = std::distance(input_name_list.begin(), iter);
      // Record the ref index when output's name is one of inputs' names and this input is rw_write.
      bool is_ref = (iter != input_name_list.end()) &&
                    (std::find(rw_write_indexes.begin(), rw_write_indexes.end(), index) != rw_write_indexes.end());
      auto inplace_index = is_ref ? index : -1;
      (void)indexes.emplace_back(inplace_index);
    }
    MS_LOG(DEBUG) << "For Primitive '" << prim->name() << "', the inplace_input_indexes is " << indexes;
    return indexes;
  }

  // Change from:
  // %0 = InplaceOp(param_x, param_y)
  // %1 = UpdataState(U, %0)
  // %2 = Depend(param_x, %1)
  // To:
  // %0 = InplaceOp(param_x, param_y)
  // %1 = UpdataState(U, %0)
  // %2 = Depend(%0, %1)
  bool ChangeInplaceInputInner(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    std::map<AnfNodePtr, AnfNodePtr> inplace_input;
    bool change = false;
    for (auto node : TopoSort(func_graph->return_node())) {
      if (!IsCNode(node) || IsPrimitiveCNode(node, prim::kPrimVirtualAssignAdd) || node->func_graph() != func_graph) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();

      for (size_t i = 1; i < cnode->size(); i++) {
        auto original_input = cnode->input(i);
        if (inplace_input.count(original_input) == 0 || original_input->func_graph() != func_graph) {
          continue;
        }
        // Find the final inplaced cnode to replace
        // For example:
        // %1 = Inplace(%0)
        // %2 = Inplace(%1)
        // %3 = Depend(%0, U) ==> %3 = Depend(%2, U)
        AnfNodePtr repalced_node = inplace_input[original_input];
        while (inplace_input.count(repalced_node) != 0) {
          repalced_node = inplace_input[repalced_node];
        }
        MS_LOG(INFO) << "Replace cnode : " << cnode->DebugString() << " input from: " << original_input->DebugString()
                     << " to: " << repalced_node->DebugString() << " for inplace ops replacement.";
        cnode->set_input(i, repalced_node);
        change = true;
      }
      const auto &prim = GetCNodePrimitive(cnode);
      if (prim != nullptr && prim->inplace_prim()) {
        const auto &indexes = inplace_input_indexes(prim);
        if (indexes.size() != 1) {
          continue;
        }
        inplace_input[cnode->input(LongToSize(indexes[0] + 1))] = cnode;
        MS_LOG(INFO) << "Record cnode as inplace node: " << cnode->DebugString();
      }
    }
    return change;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INPLACE_INPUT_REPLACE_H_
