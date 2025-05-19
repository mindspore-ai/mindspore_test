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
struct GatherNdOp {
  ShapeVector x_shape;
  ShapeVector indices_shape;
};

class TestGatherNd : public TestSymbolEngine, public testing::WithParamInterface<GatherNdOp> {};

using abstract::TensorShape;
TEST_P(TestGatherNd, compare_shape_succ) {
  // building symbolic shape like a dynamic shape node.
  MOCKER_CPP(&TensorShape::IsDynamic, bool (*)(const TensorShape *)).stubs().will(returnValue(true));

  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kFloat32, param.x_shape);
  auto indices = cg.NewTensorInput("indices", kInt64, param.indices_shape);
  auto node = cg.NewCNode("GatherNd", {x, indices});
  cg.GetGraph()->set_output(node);
  helper_->InitSymbolEngine(cg.GetGraph());
  auto out_shape = helper_->BuildSymbolicShape(node);
  UT_CHECK_NULL(out_shape);
  ASSERT_TRUE(helper_->SupportInfer());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  GlobalMockObject::verify();
}

INSTANTIATE_TEST_CASE_P(TestSymShape, TestGatherNd,
                        testing::Values(GatherNdOp{{20, 30, 40}, {2, 2, 3}},  //
                                        GatherNdOp{{20, 30, 40}, {2, 3}},     //
                                        GatherNdOp{{30, 40}, {}},             //
                                        GatherNdOp{{10}, {1}},                //
                                        GatherNdOp{{10}, {}}                  //
                                        ));
}  // namespace mindspore::symshape::test
