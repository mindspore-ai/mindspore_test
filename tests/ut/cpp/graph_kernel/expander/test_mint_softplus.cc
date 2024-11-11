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
#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_expander_cloud.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "pre_activate/common/pattern_to_pattern_pass_utils.h"
#include "graph_kernel/expander/base.h"

namespace mindspore::graphkernel::test {
namespace {
struct SoftplusExtParams {
  ShapeVector input_shape;
  int64_t beta;
  int64_t threshold;
  ShapeVector expect_shape;
  TypePtr type;
};
}  // namespace

/// Feature: Test graph kernel SoftplusExt expander
/// Description: SoftplusExt will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestSoftplusExtExpander : public TestGraphKernelExpander, public testing::WithParamInterface<SoftplusExtParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestSoftplusExtExpander, SoftplusExt) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto shape = c.NewTensorInput("input_shape", param.type, param.input_shape);
  auto beta = c.NewScalarInput("beta", MakeValue(param.beta), kFloat32);
  auto threshold = c.NewScalarInput("threshold", MakeValue(param.threshold), kFloat32);
  auto op = c.NewCNodeWithBuildInfo("SoftplusExt", {shape, beta, threshold}, {});

  c.SetOutput(op);
  RunPass(c.GetGraph(), {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
  auto nodes = TopoSort(c.GetGraph()->get_return());
  for (const auto &node : nodes) {
    if (node != nullptr && AnfUtils::IsGraphKernel(node)) {
      CompareShapeAndType(node, 0, param.expect_shape, param.type->type_id());
    }
  }
  auto g = c.GetGraph();
  UT_CHECK_NULL(g);
  auto gknodes = GetAllGKNodes(g);
  EXPECT_EQ(gknodes.size(), 1);
}

INSTANTIATE_TEST_CASE_P(TestOpSoftplusExt, TestSoftplusExtExpander,
                        testing::Values(SoftplusExtParams{{16, 16}, 1, 20, {16, 16}, kFloat16},
                                        SoftplusExtParams{{16, 16}, 1, 20, {16, 16}, kFloat32}));
}  // namespace mindspore::graphkernel::test