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

#include "symbol_engine/ops/symbolic_shape_test_utils.h"
#include "common/graph_optimizer_test_framework.h"

namespace mindspore::symshape::test {
struct ConvTestParam {
  ShapeVector x_shape;
  ShapeVector kernel_size;
  int64_t mode;
  int64_t out_channel;
  ShapeVector pad;
  int64_t pad_mode;
  std::string format;
  int64_t groups;
  ShapeVector stride;
  ShapeVector dilation;
};
class TestConv : public TestSymbolEngine, public testing::WithParamInterface<ConvTestParam> {};
class TestConv2D : public TestConv {};

/// Feature: symbolic shape
/// Description: test infer symbolic shape for Conv2D
/// Expectation: symbolic shape match the digital shape
TEST_P(TestConv2D, dyn_shape) {
  mindspore::test::ConstructGraph cg;
  const auto &param = GetParam();
  ShapeVector w_shape = {param.out_channel, -1, param.kernel_size[0], param.kernel_size[1]};
  size_t c_in_axis = param.format == "NHWC" ? 3 : 1;
  if (param.x_shape[c_in_axis] > 0) {
    w_shape[c_in_axis] = param.x_shape[c_in_axis] / param.groups;
  }
  auto x = cg.NewTensorInput("x", kFloat32, param.x_shape);
  auto w = cg.NewTensorInput("w", kFloat32, w_shape);
  auto node = cg.NewCNode("Conv2D", {x, w},
                          {{"kernel_size", MakeValue<ShapeVector>(param.kernel_size)},
                           {"mode", MakeValue<int64_t>(param.mode)},
                           {"out_channel", MakeValue<int64_t>(param.out_channel)},
                           {"pad", MakeValue<ShapeVector>(param.pad)},
                           {"pad_mode", MakeValue<int64_t>(param.pad_mode)},
                           {"format", MakeValue<std::string>(param.format)},
                           {"group", MakeValue<int64_t>(param.groups)},
                           {"stride", MakeValue<ShapeVector>(param.stride)},
                           {"dilation", MakeValue<ShapeVector>(param.dilation)}});
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}

INSTANTIATE_TEST_CASE_P(
  TestSymShape, TestConv2D,
  testing::Values(
    // pad_mode = valid
    ConvTestParam{{4, 3, -1, -1}, {14, 14}, 1, 1280, {0, 0, 0, 0}, 1, "NCHW", 1, {1, 1, 14, 14}, {1, 1, 1, 1}},
    // pad_mode = pad
    ConvTestParam{{-1, 128, 256, 256}, {3, 3}, 1, 3, {1, 1, 1, 1}, 0, "NCHW", 1, {1, 1, 1, 1}, {1, 1, 1, 1}}));

class TestConv3D : public TestConv {};
/// Feature: symbolic shape
/// Description: test infer symbolic shape for Conv3D
/// Expectation: symbolic shape match the digital shape
TEST_P(TestConv3D, dyn_shape) {
  mindspore::test::ConstructGraph cg;
  const auto &param = GetParam();
  ShapeVector w_shape = {param.out_channel, -1, param.kernel_size[0], param.kernel_size[1], param.kernel_size[2]};
  if (param.x_shape[1] > 0) {
    w_shape[1] = param.x_shape[1] / param.groups;
  }
  auto x = cg.NewTensorInput("x", kFloat32, param.x_shape);
  auto w = cg.NewTensorInput("w", kFloat32, w_shape);
  auto node = cg.NewCNode("Conv3D", {x, w},
                          {{"kernel_size", MakeValue<ShapeVector>(param.kernel_size)},
                           {"mode", MakeValue<int64_t>(param.mode)},
                           {"out_channel", MakeValue<int64_t>(param.out_channel)},
                           {"pad", MakeValue<ShapeVector>(param.pad)},
                           {"pad_mode", MakeValue<int64_t>(param.pad_mode)},
                           {"format", MakeValue<std::string>(param.format)},
                           {"group", MakeValue<int64_t>(param.groups)},
                           {"strides", MakeValue<ShapeVector>(param.stride)},
                           {"dilations", MakeValue<ShapeVector>(param.dilation)}});
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}

INSTANTIATE_TEST_CASE_P(
  TestSymShape, TestConv3D,
  testing::Values(
    // pad_mode = same
    ConvTestParam{
      {4, -1, 13, 14, 15}, {15, 15, 15}, 1, 8, {0, 0, 0, 0, 0, 0}, 2, "NCDHW", 1, {1, 1, 2, 2, 2}, {0, 0, 0, 0, 0}},
    // pad_mode = valid
    ConvTestParam{
      {4, -1, 28, 28, 28}, {13, 14, 15}, 1, 8, {0, 0, 0, 0, 0, 0}, 1, "NCDHW", 1, {1, 1, 14, 14, 14}, {1, 1, 1, 1, 1}},
    // pad_mode = pad
    ConvTestParam{
      {-1, 128, 256, 256, 256}, {3, 3, 3}, 1, 3, {1, 1, 1, 1, 1, 1}, 0, "NCDHW", 1, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}}));

/// Feature: Symbolic shape for Conv2D
/// Description: cond2d with padmode="valid"
/// Expectation: success.
TEST_F(TestSymbolEngine, conv2d_validpad_1) {
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, {1, 128, -1, -1});
  auto w = cg.NewTensorInput("w", kFloat32, {128, 128, 3, 3});
  auto node = cg.NewCNode("Conv2D", {x, w},
                          {{"kernel_size", MakeValue<ShapeVector>({3, 3})},
                           {"mode", MakeValue<int64_t>(1)},
                           {"out_channel", MakeValue<int64_t>(128)},
                           {"pad", MakeValue<ShapeVector>({0, 0, 0, 0})},
                           {"pad_mode", MakeValue<std::string>("valid")},
                           {"format", MakeValue<std::string>("NCHW")},
                           {"group", MakeValue<int64_t>(1)},
                           {"stride", MakeValue<ShapeVector>({1, 1, 2, 2})},
                           {"dilation", MakeValue<ShapeVector>({1, 1, 1, 1})}});
  helper_->InitSymbolEngine(cg.GetGraph());
  IntSymbolInfo sym_h;
  sym_h.divisor = 64;
  sym_h.remainder = 2;
  IntSymbolInfo sym_w;
  sym_w.divisor = 64;
  sym_w.remainder = 3;
  helper_->SetSymbolicShapeInfo(x, {{}, {}, sym_h, sym_w});
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  auto out_h = out_shape->item_as<IntSymbol>(2);
  auto out_w = out_shape->item_as<IntSymbol>(3);
  EXPECT_TRUE(out_h->is_divisible_by(32));
  EXPECT_FALSE(out_w->is_divisible_by(32));
}
}  // namespace mindspore::symshape::test
