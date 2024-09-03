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
struct AddcmulParams {
  ShapeVector a_shape;
  ShapeVector x1_shape;
  ShapeVector x2_shape;
  ShapeVector v_shape;
  ShapeVector expect_shape;
  TypePtr type;
};
}  // namespace

/// Feature: Test graph kernel Addcmul expander
/// Description: Addcmul will expanded
/// Expectation: After expand, the output shape and data type of sub graph should match expect
class TestAddcmulExpander : public TestGraphKernelExpander, public testing::WithParamInterface<AddcmulParams> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestAddcmulExpander, addcmul) {
  const auto &param = GetParam();
  ConstructGraph c;
  auto a = c.NewTensorInput("a", param.type, param.a_shape);
  auto x1 = c.NewTensorInput("x1", param.type, param.x1_shape);
  auto x2 = c.NewTensorInput("x2", param.type, param.x2_shape);
  auto v = c.NewTensorInput("v", param.type, param.v_shape);
  auto op = c.NewCNodeWithBuildInfo("Addcmul", {a, x1, x2, v}, {});
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

INSTANTIATE_TEST_CASE_P(TestOpAddcmul, TestAddcmulExpander,
                        testing::Values(AddcmulParams{{16, 128}, {16, 128}, {16, 128}, {16, 128}, {16, 128}, kFloat16},
                                        AddcmulParams{{16, 8}, {16, 8}, {16, 8}, {1}, {16, 8}, kFloat32},
                                        AddcmulParams{{16, 32}, {16, 32}, {1, 32}, {16, 32}, {16, 32}, kFloat16}));
}  // namespace mindspore::graphkernel::test
