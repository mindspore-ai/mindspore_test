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

#include <string>

#include "symbol_engine/ops/symbolic_shape_test_utils.h"
#include "common/graph_optimizer_test_framework.h"
#include "abstract/dshape.h"
#include "common/mockcpp.h"

namespace mindspore::symshape::test {
using abstract::TensorShape;
struct SliceOp {
  ShapeVector x_shape;
  ShapeVector begin;
  ShapeVector size;
};

class TestSlice : public TestSymbolEngine, public testing::WithParamInterface<SliceOp> {};

/// Feature: symbolic shape
/// Description: test infer symbolic shape for Slice
/// Expectation: symbolic shape match the digital shape
TEST_P(TestSlice, slice_cases) {
  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, param.x_shape);
  auto begin = cg.NewTensorInput("begin", kInt32, {static_cast<int64_t>(param.begin.size())});
  auto size = cg.NewTensorInput("size", kInt32, {static_cast<int64_t>(param.size.size())});
  auto node = cg.NewCNode("Slice", {x, begin, size});
  cg.GetGraph()->set_output(node);

  MOCKER_CPP(&TensorShape::IsDynamic, bool (*)(const TensorShape *)).stubs().will(returnValue(true));
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}
INSTANTIATE_TEST_CASE_P(TestSymShape, TestSlice,
                        testing::Values(SliceOp{{10, 10, 10, 10}, {0, 1, 2, 3}, {10, 9, 8, 7}},
                                        SliceOp{{10, 10, 10, 10}, {0, 1, 2, 3}, {-1, -1, -1, -1}},
                                        SliceOp{{10, 10, 10, 10}, {0, -10, -8, -7}, {10, 9, 8, 7}}));

/// Feature: symbolic shape
/// Description: test infer symbolic shape for Slice, inputs are dynamic rank
/// Expectation: output is dynamic rank
TEST_F(TestSymbolEngine, slice_dynrank) {
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, {-2LL});
  auto begin = cg.NewTensorInput("begin", kInt64, {-2LL});
  auto size = cg.NewTensorInput("size", kInt64, {-2LL});
  auto node = cg.NewCNode("Slice", {x, begin, size});
  cg.GetGraph()->set_output(node);
  MOCKER_CPP(&TensorShape::IsDynamic, bool (*)(const TensorShape *)).stubs().will(returnValue(true));
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}
}  // namespace mindspore::symshape::test
