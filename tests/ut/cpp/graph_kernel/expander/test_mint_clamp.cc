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
struct ClampScalarParams {
  ShapeVector x1_shape;
  float min;
  float max;
  ShapeVector expect_shape;
  TypePtr type;
};
}  // namespace

/// Feature: Test graph kernel ClampScalar expander
/// Description: ClampScalar will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestClampScalarExpander : public TestGraphKernelExpander, public testing::WithParamInterface<ClampScalarParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestClampScalarExpander, ClampScalar) {
  MOCKER_CPP(SetDeviceInfo, void (*)(const CNodePtr &cnode)).stubs().will(ignoreReturnValue());
  const auto &param = GetParam();
  ConstructGraph c;
  auto x1_shape = c.NewTensorInput("x1_shape", param.type, param.x1_shape);
  auto min = c.NewScalarInput("min", MakeValue(param.min), param.type);
  auto max = c.NewScalarInput("max", MakeValue(param.max), param.type);
  auto op = c.NewCNodeWithBuildInfo("ClampScalar", {x1_shape, min, max}, {});
  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::ProactiveFallbackExpander>()});
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      CompareShapeAndType(node, 0, param.expect_shape, param.type->type_id());
    }
  }
  mindspore::opt::CheckPattern checker;
  checker.src_pattern_.AddVar("x1")
    .AddVar("min")
    .AddVar("max")
    .AddCNode("Max", {std::make_shared<Primitive>("Maximum"), "x1", "max"})
    .AddCNode("Min", {std::make_shared<Primitive>("Minimum"), "Max", "min"});
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  EXPECT_TRUE(checker.build_pattern_map(g->output()));
  GlobalMockObject::verify();
}

INSTANTIATE_TEST_CASE_P(TestOpClampScalar, TestClampScalarExpander,
                        testing::Values(ClampScalarParams{{4, 4, 4}, 1.0, 3.0, {4, 4, 4}, kFloat16},
                                        ClampScalarParams{{4, 4, 4}, 1.0, 3.0, {4, 4, 4}, kFloat32}));
}  // namespace mindspore::graphkernel::test