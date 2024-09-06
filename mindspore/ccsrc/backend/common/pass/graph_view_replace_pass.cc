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

#include "backend/common/pass/graph_view_replace_pass.h"

#include <map>
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <utility>
#include "include/common/utils/anfalgo.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ops/op_def/other_op_name.h"
#include "mindspore/ops/op_def/array_op_name.h"
#include "mindspore/ops/op_def/math_op_name.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/core/include/utils/ms_utils.h"

namespace mindspore {
namespace opt {
using mindspore::ops::op_enabled_aclnn;
static const std::set<std::string> op_reverse = {kConcatOpName};
static const std::set<std::string> multi_out = {kSplitOpName};
constexpr size_t kAlignSize = 512;

size_t GetOutputMemSize(const AnfNodePtr &node, size_t output_index) {
  MS_EXCEPTION_IF_NULL(node);
  if (output_index >= AnfUtils::GetOutputTensorNum(node)) {
    MS_EXCEPTION(ArgumentError) << "output index [" << output_index << "] large than the output size ["
                                << AnfUtils::GetOutputTensorNum(node) << "] of node!";
  }
  TypeId output_typeid = common::AnfAlgo::GetOutputInferDataType(node, output_index);
  size_t type_size = GetTypeByte(TypeIdToType(output_typeid));
  auto shape = common::AnfAlgo::GetOutputInferShape(node, output_index);
  size_t tensor_size = type_size * SizeOf(shape);
  return tensor_size;
}

bool IsNodeBoundary(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  if (func_graph->output() == nullptr) {
    return false;
  }
  const auto &outputs = common::AnfAlgo::GetAllOutput(func_graph->output());
  auto it = std::find(outputs.begin(), outputs.end(), node);
  return (it != outputs.end());
}

bool IsOutSuit(const AnfNodePtr &node, const mindspore::FuncGraphManagerPtr &manager) {
  auto name = AnfUtils::GetCNodeName(node);
  if (multi_out.count(name) || op_reverse.count(name)) {
    return true;
  }
  auto users = manager->node_users()[node];
  for (const auto user : users) {
    auto out = user.first;
    if (!out->cast<CNodePtr>()) {
      return false;
    }
    // Out is not aclnn kernel: OPAPI_KERNEL
    auto out_name = AnfUtils::GetCNodeName(out);
    if (out_name != kMatMulOpName) {
      return false;
    }
  }
  return true;
}

bool CheckReverseOp(const CNodePtr &cnode) {
  auto kernel_name = AnfUtils::GetCNodeName(cnode);
  if (!op_reverse.count(kernel_name)) {
    return true;
  }
  // not support dynamic shape
  if (common::AnfAlgo::IsDynamicShape(cnode)) {
    return false;
  }
  // check all input is matmul
  auto inputs = cnode->inputs();
  if (inputs.empty()) {
    return false;
  }
  // check axis is 0
  auto axis = GetValue<int64_t>(inputs[inputs.size() - 1]->cast<ValueNodePtr>()->value());
  if (axis != 0) {
    return false;
  }
  for (size_t i = 1; i < inputs.size() - 1; ++i) {
    if (!inputs[i]->isa<CNode>()) {
      return false;
    }
    if (AnfUtils::GetCNodeName(inputs[i]) != kMatMulOpName) {
      return false;
    }
    // checkout input size aligned 512
    auto input_size = GetOutputMemSize(cnode, 0);
    if (input_size % kAlignSize != 0) {
      return false;
    }
  }
  return true;
}

bool CheckMultiOut(const CNodePtr &cnode, const mindspore::FuncGraphManagerPtr &manager) {
  auto kernel_name = AnfUtils::GetCNodeName(cnode);
  if (!multi_out.count(kernel_name)) {
    return true;
  }
  // not support dynamic shape
  if (common::AnfAlgo::IsDynamicShape(cnode)) {
    return false;
  }
  auto inputs = cnode->inputs();
  if (inputs.empty()) {
    return false;
  }
  // check axis is 0  Split(x, axis, out_num)
  size_t axis_pos = inputs.size() - kIndex2;
  if (axis_pos < 1) {
    return false;
  }
  auto axis = GetValue<int64_t>(inputs[axis_pos]->cast<ValueNodePtr>()->value());
  if (axis != 0) {
    return false;
  }
  auto users = manager->node_users()[cnode];
  // skip tuple getitem
  for (auto out : users) {
    auto out_node = out.first;
    auto out_user = manager->node_users()[out_node];
    // check out size aligned 512
    size_t out_size = GetOutputMemSize(out_node, 0);
    if (out_size % kAlignSize != 0) {
      return false;
    }
    for (auto out_out : out_user) {
      auto out_out_node = out_out.first;
      if (!out_out_node->cast<CNodePtr>()) {
        return false;
      }
      // need check optype when make it common : GetKernelType(out_out_node) != OPAPI_KERNEL
      auto name = AnfUtils::GetCNodeName(out_out_node);
      if (name != kMatMulOpName) {
        return false;
      }
    }
  }
  return true;
}

void MakeRefPairForViewNode(const CNodePtr &view_node, const CNodePtr &origin_node, const KernelGraphPtr &kg,
                            bool need_reverse = false) {
  if (need_reverse) {
    auto input_num = common::AnfAlgo::GetInputNum(origin_node);
    for (size_t i = 0; i < input_num; ++i) {
      auto input = common::AnfAlgo::GetInputNode(origin_node, i);
      if (input->isa<CNode>()) {
        kg->AddRefCorrespondPairs(std::make_pair(input, 0), std::make_pair(view_node, 0));
      }
    }
  } else {
    auto output_num = AnfUtils::GetOutputTensorNum(origin_node);
    for (size_t i = 0; i < output_num; ++i) {
      auto origin_pair = common::AnfAlgo::VisitKernel(common::AnfAlgo::GetInputNode(origin_node, 0), 0);
      kg->AddRefCorrespondPairs(std::make_pair(view_node, i), origin_pair);
    }
  }
}

bool GraphViewReplacePass::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto kernel_name = AnfUtils::GetCNodeName(node);
    if (!mindspore::common::IsEnableAclnnViewOp(kernel_name)) {
      continue;
    }
    auto ops = op_enabled_aclnn.find(kernel_name);
    if (ops == op_enabled_aclnn.end() || IsNodeBoundary(func_graph, node) || !IsOutSuit(node, manager)) {
      continue;
    }
    if (!CheckReverseOp(cnode)) {
      continue;
    }
    if (!CheckMultiOut(cnode, manager)) {
      continue;
    }
    MS_LOG(INFO) << "Process view for " << kernel_name;
    // Create view node
    auto inputs = cnode->inputs();
    inputs[0] = NewValueNode(std::make_shared<Primitive>(ops->second));
    auto view_node = func_graph->NewCNode(inputs);
    // Copy attributes
    common::AnfAlgo::CopyNodeAttrs(node, view_node);
    view_node->AddAttr("enable_view", MakeValue(true));
    // Set node abstract
    view_node->set_abstract(node->abstract());
    auto kernel_graph = func_graph->cast<KernelGraphPtr>();
    MS_EXCEPTION_IF_NULL(kernel_graph);

    MakeRefPairForViewNode(view_node, cnode, kernel_graph, op_reverse.count(kernel_name));

    // Replace node
    (void)manager->Replace(cnode, view_node);
  }
  return True;
}
}  // namespace opt
}  // namespace mindspore
