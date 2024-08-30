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

#include <map>
#include <string>
#include <iostream>
#include "common/mockcpp.h"
#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/proactive_fallback_expander.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "graph_kernel/expander/base.h"

namespace mindspore::graphkernel {
void SetDeviceInfo(const CNodePtr &cnode);
}

namespace mindspore::graphkernel::test {
namespace {
struct BatchMatMulExtParams {
  ShapeVector x1_shape;
  ShapeVector x2_shape;
  ShapeVector expect_shape;
  TypePtr type;
};
}  // namespace

/// Feature: Test graph kernel BatchMatMulExt expander
/// Description: BatchMatMulExt will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestBatchMatMulExtExpander : public TestGraphKernelExpander,
                                   public testing::WithParamInterface<BatchMatMulExtParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestBatchMatMulExtExpander, BatchMatMulExt) {
  MOCKER_CPP(SetDeviceInfo, void (*)(const CNodePtr &cnode)).stubs().will(ignoreReturnValue());
  const auto &param = GetParam();
  ConstructGraph c;
  auto x1_shape = c.NewTensorInput("x1_shape", param.type, param.x1_shape);
  auto x2_shape = c.NewTensorInput("x2_shape", param.type, param.x2_shape);
  auto op = c.NewCNodeWithBuildInfo("BatchMatMulExt", {x1_shape, x2_shape}, {});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::ProactiveFallbackExpander>()});
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      CompareShapeAndType(node, 0, param.expect_shape, param.type->type_id());
    }
  }
  mindspore::opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1").AddVar("x2").AddVar("trans_a").AddVar("trans_b").AddCNode(
    "bmm", {std::make_shared<Primitive>("BatchMatMul"), "x1", "x2", "trans_a", "trans_b"});
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  EXPECT_TRUE(checker.build_pattern_map(g->output()));
  GlobalMockObject::verify();
}

INSTANTIATE_TEST_CASE_P(TestOpBatchMatMulExt, TestBatchMatMulExtExpander,
                        testing::Values(BatchMatMulExtParams{{4, 4, 4}, {4, 4, 4}, {4, 4, 4}, kFloat16},
                                        BatchMatMulExtParams{{4, 4, 4}, {4, 4, 4}, {4, 4, 4}, kFloat32}));
}  // namespace mindspore::graphkernel::test