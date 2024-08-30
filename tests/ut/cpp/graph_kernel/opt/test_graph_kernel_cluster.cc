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
#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "common/graph_optimizer_test_framework.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "backend/common/graph_kernel/adapter/graph_kernel_cluster_cloud.h"

namespace mindspore::graphkernel::test {
class TestGraphCluster : public GraphKernelCommonTestSuite {};

namespace {
FuncGraphPtr ConstructGraph_1() {
  ConstructGraph gb;
  test::ConstructGraph c;
  auto x0 = c.NewTensorInput("x0", kFloat32, {-1, 32});
  auto x1 = c.NewTensorInput("x1", kFloat32, {-1, 32});
  auto op = c.NewCNodeWithBuildInfo("Sub", {x0, x1}, {});
  auto op1 = c.NewCNodeWithBuildInfo("Add", {op, op}, {});
  auto op2 = c.NewCNodeWithBuildInfo("Sub", {op1, op}, {});
  auto op3 = c.NewCNodeWithBuildInfo("Add", {op1, op2}, {});
  c.SetOutput(op3);
  return c.GetGraph();
}
}  // namespace

/// Feature: Test graph kernel cluster pass
/// Description: op will cluster, check cluster and not cluster.
/// Expectation: After cluster pass, the gk node and main node should be correct.
TEST_F(TestGraphCluster, cluster_cut_max_id) {
  SetGraphKernelFlags("--disable_cluster_ops=Sub");
  SetDeviceTarget(kAscendDevice);
  auto fg = ConstructGraph_1();
  RunPass(fg, {std::make_shared<graphkernel::StaticShapeCluster>()});
  ASSERT_EQ(GetAllGKNodes(fg).size(), 2);
  ASSERT_EQ(GetAllCNodes(fg).size(), 4);
}
}  // namespace mindspore::graphkernel::test
