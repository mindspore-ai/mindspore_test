/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "kernel/ascend/opapi/aclnn_kernel_build.h"
#include "mindspore/core/include/ops/op_def.h"

namespace mindspore {
namespace opt {

bool IsInputsFromView(const CNodePtr &origin_node) {
  auto inputs = origin_node->inputs();
  for (auto input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (!input->isa<CNode>()) {
      continue;
    }
    auto input_cnode = input->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(input_cnode);
    if (common::AnfAlgo::IsViewNode(input_cnode)) {
      return true;
    }
  }
  return false;
}

bool TransposePattern(const AnfNodePtr &node, const mindspore::FuncGraphManagerPtr &manager) {
  auto users = manager->node_users()[node];
  for (const auto &user : users) {
    auto out = user.first;
    if (!out->cast<CNodePtr>()) {
      return false;
    }
    // Out is not aclnn kernel: OPAPI_KERNEL
    auto out_name = AnfUtils::GetCNodeName(out);
    std::set<std::string> white_list{ops::kNameMatMul, ops::kNameGroupedMatmul, ops::kNameGroupedMatmulV2,
                                     ops::kNameGroupedMatmulV4};
    if (white_list.find(out_name) == white_list.end()) {
      return false;
    }
  }
  return true;
}

void MakeRefPairForNode(const CNodePtr &origin_node) {
  auto output_num = AnfUtils::GetOutputTensorNum(origin_node);
  auto kernel_info = dynamic_cast<device::KernelInfo *>(origin_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  for (size_t i = 0; i < output_num; ++i) {
    kernel_info->AddRefMap(i, 0);
  }
}

void CreateViewNode(const std::string &name, const AnfNodePtr &origin_node,
                    const mindspore::FuncGraphManagerPtr &manager, const FuncGraphPtr &func_graph) {
  auto ops = name;
  if (ops == "Transpose") {
    if (TransposePattern(origin_node, manager)) {
      ops = ops.append("View");
    } else {
      return;
    }
  }
  auto cnode = origin_node->cast<CNodePtr>();
  auto inputs = cnode->inputs();
  inputs[0] = NewValueNode(std::make_shared<Primitive>(ops));
  auto view_node = func_graph->NewCNode(inputs);
  // Copy attributes
  common::AnfAlgo::CopyNodeAttrs(origin_node, view_node);
  // Set node abstract
  view_node->set_abstract(origin_node->abstract());
  view_node->set_kernel_info(origin_node->kernel_info_ptr());
  const auto &kernel_build_info = AnfAlgo::GetSelectKernelBuildInfo(view_node);
  kernel_build_info->set_kernel_type(OPAPI_KERNEL);
  view_node->AddAttr("enable_view", MakeValue(true));
  MakeRefPairForNode(view_node);
  // Replace node
  (void)manager->Replace(cnode, view_node);
}

bool GraphViewReplacePass::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return());
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->cast<CNodePtr>()) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto kernel_name = AnfUtils::GetCNodeName(node);

    // The view op list defined in yamls. Spacial case: Transpose + GroupMatmul/Matmul
    if (!(common::AnfAlgo::IsViewNode(node) || kernel_name == "Transpose")) {
      continue;
    }
    // Skip reshapeview when input is from view. Need to be done when the ref count is ready.
    if (kernel_name == "Reshape" && IsInputsFromView(cnode)) {
      continue;
    }
    MS_LOG(INFO) << "Process view for " << kernel_name;
    CreateViewNode(kernel_name, node, manager, func_graph);
  }
  return True;
}
}  // namespace opt
}  // namespace mindspore
