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
struct FlattenOp {
  ShapeVector shape;
};
struct FlattenExtOp {
  ShapeVector shape;
  int64_t start;
  int64_t end;
};

class TestFlatten : public TestSymbolEngine, public testing::WithParamInterface<FlattenOp> {};
class TestFlattenExt : public TestSymbolEngine, public testing::WithParamInterface<FlattenExtOp> {};

using abstract::TensorShape;
TEST_P(TestFlatten, compare_shape_succ) {
  // building symbolic shape like a dynamic shape node.
  MOCKER_CPP(&TensorShape::IsDynamic, bool (*)(const TensorShape *)).stubs().will(returnValue(true));

  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, param.shape);
  auto node = cg.NewCNode("Flatten", {x});
  cg.GetGraph()->set_output(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  GlobalMockObject::verify();
}

INSTANTIATE_TEST_CASE_P(TestSymShape, TestFlatten,
                        testing::Values(FlattenOp{{10}},             //
                                        FlattenOp{{10, 20}},         //
                                        FlattenOp{{10, 20, 30}},     //
                                        FlattenOp{{10, 20, 30, 40}}  //
                                        ));

TEST_P(TestFlattenExt, compare_shape_succ) {
  // building symbolic shape like a dynamic shape node.
  MOCKER_CPP(&TensorShape::IsDynamic, bool (*)(const TensorShape *)).stubs().will(returnValue(true));

  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, param.shape);
  auto node = cg.NewCNode("FlattenExt", {x, cg.NewValueNode(param.start), cg.NewValueNode(param.end)});
  cg.GetGraph()->set_output(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  GlobalMockObject::verify();
}

INSTANTIATE_TEST_CASE_P(TestSymShape, TestFlattenExt,
                        testing::Values(FlattenExtOp{{}, 0, 0},                 //
                                        FlattenExtOp{{10}, 0, -1},              //
                                        FlattenExtOp{{10, 20}, 0, -1},          //
                                        FlattenExtOp{{10, 20, 30}, 0, 1},       //
                                        FlattenExtOp{{10, 20, 30}, 1, 1},       //
                                        FlattenExtOp{{10, 20, 30}, 1, 2},       //
                                        FlattenExtOp{{10, 20, 30, 40}, 0, 2},   //
                                        FlattenExtOp{{10, 20, 30, 40}, 0, -1},  //
                                        FlattenExtOp{{10, 20, 30, 40}, 1, 2},   //
                                        FlattenExtOp{{10, 20, 30, 40}, -3, -1}  //
                                        ));
}  // namespace mindspore::symshape::test
