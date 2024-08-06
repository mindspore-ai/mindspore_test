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

#include "graph_kernel/kernel_packet/kernel_packet_common_test_suite.h"

namespace mindspore::graphkernel::test {
/// Feature: KernelPacket
/// Description: the "Range"'s value-depend node comes from parameter
/// Expectation: fuse until parameter
TEST_F(TestKernelPacket, depend_param_value) {
  ConstructGraph gb;
  auto p1 = gb.NewScalarInput("p1", kInt64);
  auto p2 = gb.NewScalarInput("p2", kInt64);
  auto p3 = gb.NewScalarInput("p3", kInt64);
  auto p4 = gb.NewScalarInput("p4", kInt64);
  auto t = gb.NewCNodeWithBuildInfo("ScalarToTensor", {p2, gb.NewValueNode(MakeValue<int64_t>(kNumberTypeInt64))});
  auto s = gb.NewCNodeWithBuildInfo("TensorToScalar", {t});
  auto range = gb.NewCNodeWithBuildInfo("Range", {p1, s, p3, p4});
  gb.SetOutput(range);
  auto fg = gb.GetGraph();
  RunPass(fg, {std::make_shared<packet::SymbolEngineExtender>(), std::make_shared<ConvertCallToPrim>()});
  EXPECT_EQ(GetAllCNodes(fg).size(), 1);
  EXPECT_EQ(GetAllPacketNodes(fg).size(), 1);
}
}  // namespace mindspore::graphkernel::test
