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
#include "mindspore/ops/op_def/op_enum.h"

namespace mindspore::graphkernel::test {
namespace {
struct DivModParams {
  ShapeVector x1_shape;
  ShapeVector x2_shape;
  ops::RoundingMode rounding_mode;
  ShapeVector expect_shape;
  TypePtr type;
};
}  // namespace

/// Feature: Test graph kernel DivMod expander
/// Description: DivMod will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestDivModExpander : public TestGraphKernelExpander, public testing::WithParamInterface<DivModParams> {
  void SetUp() override {
    SetDeviceTarget(kAscendDevice);
    std::map<std::string, std::string> jit_config;
    jit_config["graph_kernel_flags"] = "--kernel_generator=DVM";
    graphkernel::GraphKernelFlags::SaveJitConfig(jit_config);
  }
};

TEST_P(TestDivModExpander, DivMod) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto x1 = c.NewTensorInput("x1_shape", param.type, param.x1_shape);
  auto x2 = c.NewTensorInput("x2_shape", param.type, param.x2_shape);
  auto rounding_mode = c.NewScalarInput("rounding_mode", MakeValue(static_cast<int64_t>(param.rounding_mode)), kInt64);
  auto op = c.NewCNodeWithBuildInfo("DivMod", {x1, x2, rounding_mode}, {});
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

INSTANTIATE_TEST_CASE_P(TestOpDivMod, TestDivModExpander,
                        testing::Values(DivModParams{{16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kFloat16},
                                        DivModParams{{16, 16}, {16, 16}, ops::RoundingMode::FLOOR, {16, 16}, kFloat32},
                                        DivModParams{{16, 16}, {16, 16}, ops::RoundingMode::TRUNC, {16, 16}, kFloat16},
                                        DivModParams{
                                          {16, 16}, {16, 16}, ops::RoundingMode::TRUNC, {16, 16}, kFloat32}));
}  // namespace mindspore::graphkernel::test