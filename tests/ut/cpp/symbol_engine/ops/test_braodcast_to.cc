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

namespace mindspore::symshape::test {
class TestBroadcastTo : public TestSymbolEngine {};

/// Feature: Symbolic shape for BroadcastTo
/// Description: output shape has -1
/// Expectation: success.
TEST_F(TestBroadcastTo, case1_const_out) {
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, {-1, -1});
  auto out = cg.NewValueNode(MakeValue<ShapeVector>({16, -1}));
  auto opname = "BroadcastTo";
  auto node = cg.NewCNode(opname, {x, out});
  cg.SetOutput(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  abstract::AbstractBasePtrList inputs_args{std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1, 32})};
  ASSERT_TRUE(helper_->Infer(inputs_args));
  node->abstract()->set_shape(std::make_shared<abstract::TensorShape>(ShapeVector{16, 32}));
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}

/// Feature: Symbolic shape for BroadcastTo
/// Description: output shape has unknown value
/// Expectation: success.
TEST_F(TestBroadcastTo, case2_dyn_out) {
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, {1, -1, 32});
  auto out = cg.NewTensorInput("out", kInt64, {4});
  auto opname = "BroadcastTo";
  auto node = cg.NewCNode(opname, {x, out});
  cg.SetOutput(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->SupportInfer());
  // ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  abstract::AbstractBasePtrList inputs_args{
    std::make_shared<abstract::AbstractTensor>(kFloat32, ShapeVector{1, 16, 32}),
    MakeValue<ShapeVector>({8, 4, 16, 32})->ToAbstract()};
  ASSERT_TRUE(helper_->Infer(inputs_args));
  node->abstract()->set_shape(std::make_shared<abstract::TensorShape>(ShapeVector{8, 4, 16, 32}));
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
}
}  // namespace mindspore::symshape::test
