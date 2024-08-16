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

#include "include/common/utils/anfalgo.h"
#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/core/graph_kernel_pass_manager.h"

namespace mindspore::graphkernel::test {
class EmptyPass : public mindspore::opt::Pass {
 public:
  using Pass::Pass;
  bool Run(const FuncGraphPtr &) override { return false; }
};

void GraphKernelCommonTestSuite::RunPass(const FuncGraphPtr &graph, const std::vector<opt::PassPtr> &passes) {
  UT_CHECK_NULL(graph);
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<GraphKernelPassManager>(pass_stage_++, "ut");
  // when running UT, user can use the environment "export MS_DEV_SAVE_GRAPHS=2" to dump ir with PassManager.
  // add an empty pass to dump the original graph before running.
  pm->Add(std::make_shared<EmptyPass>("ir_before_running"), 0, true);
  for (const auto &pass : passes) {
    UT_CHECK_NULL(pass);
    pm->Add(pass, 0, true);
  }
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(graph);
}

AnfNodePtrList GraphKernelCommonTestSuite::GetAllNodes(const FuncGraphPtr &fg) { return TopoSort(fg->output()); }

CNodePtrList GraphKernelCommonTestSuite::GetAllCNodes(const FuncGraphPtr &fg) {
  CNodePtrList cnodes;
  for (auto &node : GetAllNodes(fg)) {
    if (node->isa<CNode>()) {
      (void)cnodes.emplace_back(node->cast<CNodePtr>());
    }
  }
  return cnodes;
}

CNodePtrList GraphKernelCommonTestSuite::GetAllGKNodes(const FuncGraphPtr &fg) {
  auto cnodes = GetAllCNodes(fg);
  CNodePtrList gk_nodes;
  std::copy_if(cnodes.begin(), cnodes.end(), std::back_inserter(gk_nodes),
               [](const CNodePtr &node) { return common::AnfAlgo::HasNodeAttr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, node); });
  return gk_nodes;
}

void GraphKernelCommonTestSuite::SetGraphKernelFlags(const std::string &flags) {
  std::map<std::string, std::string> jit_config;
  jit_config["graph_kernel_flags"] = flags;
  graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
}

void GraphKernelCommonTestSuite::SetDeviceTarget(const std::string &device) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  context->set_param<std::string>(MS_CTX_DEVICE_TARGET, device);
}
}  // namespace mindspore::graphkernel::test
