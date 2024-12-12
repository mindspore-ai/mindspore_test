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
struct DropoutExtOp {
  ShapeVector shape;
};

class TestDropoutExt : public TestSymbolEngine, public testing::WithParamInterface<DropoutExtOp> {};

using abstract::TensorShape;
TEST_P(TestDropoutExt, compare_shape_succ) {
  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, param.shape);
  auto p = cg.NewValueNode(MakeValue<float>(0.6));
  auto seed = cg.NewTensorInput("seed", kInt64, {1});
  auto offset = cg.NewTensorInput("offset", kInt64, {1});
  auto node = cg.NewCNode("DropoutExt", {x, p, seed, offset});
  cg.GetGraph()->set_output(node);
  MOCKER_CPP(&TensorShape::IsDynamic, bool (*)(const TensorShape *)).stubs().will(returnValue(true));
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  GlobalMockObject::verify();
}

INSTANTIATE_TEST_CASE_P(TestSymShape, TestDropoutExt,
                        testing::Values(DropoutExtOp{{32}},      //
                                        DropoutExtOp{{32, 64}},  //
                                        DropoutExtOp{{127}},     //
                                        DropoutExtOp{{16}},      //
                                        DropoutExtOp{{256, 1}},  //
                                        DropoutExtOp{{250, 2}}   //
                                        ));
}  // namespace mindspore::symshape::test
