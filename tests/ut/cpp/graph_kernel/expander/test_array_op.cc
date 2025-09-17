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

#include "graph_kernel/common/graph_kernel_common_test_suite.h"
#include "backend/ms_backend/graph_fusion/adapter/graph_kernel_expander_cloud.h"
#include "graph_kernel/expander/base.h"

namespace mindspore::graphkernel::test {
namespace {
struct Params {
  bool can_expand;
  TypePtr input_type;
  ShapeVector input_shape;
  TypePtr dst_type{nullptr};
};
}  // namespace

/// Feature: Test ArrayOp expander
/// Description: test op with different inputs
/// Expectation: Can be expanded only when its input data types are supported.
class TestArrayOpExpander : public TestGraphKernelExpander, public testing::WithParamInterface<Params> {
  void SetUp() override { SetDeviceTarget(kAscendDevice); }
};

TEST_P(TestArrayOpExpander, array_op) {
  const auto &param = GetParam();
  if (param.dst_type == nullptr) {
    std::vector<std::string> op_names{"OnesLike", "ZerosLike"};
    for (const auto &op_name : op_names) {
      ConstructGraph c;
      auto x = c.NewTensorInput("x", param.input_type, param.input_shape);
      auto op = c.NewCNodeWithBuildInfo(op_name, {x});
      c.SetOutput(op);
      auto fg = c.GetGraph();
      RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
      size_t gk_size = param.can_expand ? 1 : 0;
      ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
    }
  } else {
    ConstructGraph c;
    auto x = c.NewTensorInput("x", param.input_type, param.input_shape);
    auto dst_type = c.NewValueNode<int64_t>(param.dst_type->type_id());
    auto op = c.NewCNodeWithBuildInfo("ZerosLikeExt", {x, dst_type});
    c.SetOutput(op);
    auto fg = c.GetGraph();
    RunPass(fg, {std::make_shared<graphkernel::GraphKernelExpanderCloud>()});
    size_t gk_size = param.can_expand ? 1 : 0;
    ASSERT_EQ(GetAllGKNodes(fg).size(), gk_size);
  }
}

INSTANTIATE_TEST_CASE_P(TestOpArrayOp, TestArrayOpExpander,
                        testing::Values(
                          // OnesLike/ZerosLike
                          Params{true, kFloat16, {16, 16}}, Params{true, kFloat32, {16, 16}},
                          Params{true, kBFloat16, {16, 16}}, Params{true, kInt32, {16, 16}},
                          Params{false, kFloat16, {-2}}, Params{false, kFloat16, {-1, 1, 2}},
                          Params{false, kFloat16, {-1, -1, 2}}, Params{false, kFloat16, {2, 0, 16}},
                          Params{false, kFloat64, {16, 16}}, Params{false, kInt8, {16, 16}},
                          Params{false, kInt16, {16, 16}}, Params{false, kInt64, {16, 16}},
                          Params{false, kBool, {16, 16}},
                          // ZerosLikeExt
                          Params{true, kFloat16, {16, 16}, kFloat16}, Params{true, kFloat32, {16, 16}, kFloat32},
                          Params{true, kBFloat16, {16, 16}, kBFloat16}, Params{true, kInt32, {16, 16}, kInt32},
                          Params{true, kFloat16, {16, 16}, kFloat32}, Params{true, kFloat16, {16, 16}, kBFloat16},
                          Params{true, kFloat16, {16, 16}, kInt32}, Params{false, kFloat16, {-2}, kFloat16},
                          Params{false, kFloat16, {-1, 1, 2}, kFloat16}, Params{false, kFloat16, {-1, -1, 2}, kFloat16},
                          Params{false, kFloat16, {2, 0, 16}, kFloat16}, Params{false, kFloat64, {16, 16}, kFloat64},
                          Params{false, kInt8, {16, 16}, kInt8}, Params{false, kInt16, {16, 16}, kInt16},
                          Params{false, kInt64, {16, 16}, kInt64}, Params{false, kBool, {16, 16}, kBool}));
}  // namespace mindspore::graphkernel::test
