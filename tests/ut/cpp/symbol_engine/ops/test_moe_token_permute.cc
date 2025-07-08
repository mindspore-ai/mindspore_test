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
#include <optional>
#include "symbol_engine/ops/symbolic_shape_test_utils.h"
#include "common/graph_optimizer_test_framework.h"
#include "abstract/dshape.h"
#include "common/mockcpp.h"

namespace mindspore::symshape::test {
struct MoeTokenPermuteOp {
  ShapeVector tokens;
  ShapeVector indices;
  std::optional<int64_t> num_out_tokens;
  bool padded_mode;
};

class TestMoeTokenPermute : public TestSymbolEngine, public testing::WithParamInterface<MoeTokenPermuteOp> {};

using abstract::TensorShape;
TEST_P(TestMoeTokenPermute, compare_shape_succ) {
  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto tokens = cg.NewTensorInput("tokens", kBFloat16, param.tokens);
  auto indices = cg.NewTensorInput("indices", kInt64, param.indices);
  auto num_out_tokens =
    cg.NewValueNode(param.num_out_tokens.has_value() ? MakeValue(param.num_out_tokens.value()) : kNone);
  auto padded_mode = cg.NewValueNode<bool>(param.padded_mode);
  auto node = cg.NewCNode("MoeTokenPermute", {tokens, indices, num_out_tokens, padded_mode});
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

INSTANTIATE_TEST_CASE_P(TestSymShape, TestMoeTokenPermute,
                        testing::Values(MoeTokenPermuteOp{{10, 20}, {10}, std::nullopt, False},  //
                                        MoeTokenPermuteOp{{10, 20}, {10}, 5, False},             //
                                        MoeTokenPermuteOp{{10, 20}, {10, 10}, 10, False}         //
                                        ));
}  // namespace mindspore::symshape::test
