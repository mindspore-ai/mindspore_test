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

#include <mindspore/core/include/ir/core_ops_primitive.h>
#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "backend/common/graph_kernel/core/split_umonad.h"
#include "backend/common/graph_kernel/convert_input_and_attr.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_expander_cloud.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_cluster_cloud.h"
#include "utils/anf_utils.h"

namespace mindspore::graphkernel::test {
struct ViewParams {
  std::string view_op;
  bool used_by_inplace;
  ShapeArray shapes;
};

/// Feature: test view op fusion in graph kernel
/// Description: view + non-inplace, view + inplace
/// Expectation: 1) view + non-inplace, view can be fused 2) view + inplace, view can not be fused
class TestViewFuse : public GraphKernelCommonTestSuite, public testing::WithParamInterface<ViewParams> {};

TEST_P(TestViewFuse, view_op_fuse) {
  SetDeviceTarget(kAscendDevice);
  SetGraphKernelFlags("--enable_cluster_ops_only=BroadcastTo,Abs,Add,ReduceSum --enable_expand_ops_only=ExpandDims");
  const auto &param = GetParam();
  ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, param.shapes[0]);
  auto x1 = c.NewTensorInput("x1", kFloat32, param.shapes[1]);
  auto u = c.NewValueNode(kUMonad);
  u->set_abstract(kUMonad->ToAbstract());
  auto y0 = c.NewCNodeWithBuildInfo("Abs", {x0});
  CNodePtr y1;
  if (param.view_op == "BroadcastTo") {
    y1 = c.NewCNodeWithBuildInfo(param.view_op, {y0, c.NewValueNode<std::vector<int64_t>>(param.shapes[1]), u});
  } else {
    y1 = c.NewCNodeWithBuildInfo(param.view_op, {y0, c.NewValueNode<int64_t>(0), u});
  }
  auto y2 = c.NewCNodeWithBuildInfo("Add", {y0, y1});
  auto axis = c.NewValueNode(MakeValue(std::vector<int64_t>{0}));
  auto keep_dims = c.NewValueNode(MakeValue<bool>(false));
  auto skip_mode = c.NewValueNode(MakeValue<bool>(true));
  auto y3 = c.NewCNodeWithBuildInfo("ReduceSum", {y2, axis, keep_dims, skip_mode});
  CNodePtr y4;
  if (param.used_by_inplace) {
    y4 = c.NewCNodeWithBuildInfo("Assign", {y1, x1, u}, {{GRAPH_FLAG_SIDE_EFFECT_MEM, MakeValue(true)}});
  } else {
    y4 = c.NewCNodeWithBuildInfo("Mul", {y1, x1});
  }
  auto y5 = c.NewCNodeWithBuildInfo("Sub", {y0, y4});
  auto y6 = c.NewCNodeWithBuildInfo("Mul", {y3, y5});
  c.SetOutput(y6);
  auto fg = c.GetGraph();
  RunPass(fg, {std::make_shared<graphkernel::ConvertFrontEndToGraphKernel>(),
               std::make_shared<graphkernel::SplitUMonad>(), std::make_shared<graphkernel::GraphKernelExpanderCloud>(),
               std::make_shared<graphkernel::StaticShapeCluster>()});
  AnfNodePtr view_node = nullptr;
  for (auto &node : TopoSort(fg->output())) {
    if (node->isa<CNode>() && AnfUtils::GetCNodeName(node) == param.view_op) {
      view_node = node;
    }
  }
  ASSERT_TRUE((param.used_by_inplace && view_node != nullptr) || (!param.used_by_inplace && view_node == nullptr));
}

INSTANTIATE_TEST_CASE_P(TestViewOpFuse, TestViewFuse,
                        testing::Values(ViewParams{"BroadcastTo", false, {{2, 4, 1}, {2, 4, 6}}},
                                        ViewParams{"BroadcastTo", true, {{2, 4, 1}, {2, 4, 6}}},
                                        ViewParams{"ExpandDims", false, {{4, 6}, {1, 4, 6}}},
                                        ViewParams{"ExpandDims", true, {{4, 6}, {1, 4, 6}}}));
}  // namespace mindspore::graphkernel::test