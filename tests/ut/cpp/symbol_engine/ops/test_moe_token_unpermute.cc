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
struct MoeTokenUnpermuteOp {
  int64_t num_tokens;
  int64_t topk;
  int64_t hidden_size;
  bool has_probs;
  bool padded_mode;
};

class TestMoeTokenUnpermute : public TestSymbolEngine, public testing::WithParamInterface<MoeTokenUnpermuteOp> {};

using abstract::TensorShape;
TEST_P(TestMoeTokenUnpermute, compare_shape_succ) {
  const auto &param = GetParam();
  mindspore::test::ConstructGraph cg;
  auto permuted_tokens =
    cg.NewTensorInput("permuted_tokens", kBFloat16, {param.num_tokens * param.topk, param.hidden_size});
  auto sorted_indices = cg.NewTensorInput("sorted_indices", kInt32, {param.num_tokens * param.topk});
  AnfNodePtr probs;
  if (param.has_probs) {
    probs = cg.NewTensorInput("probs", kBFloat16, {param.num_tokens, param.topk});
  } else {
    probs = cg.NewValueNode(kNone);
  }
  auto padded_mode = cg.NewValueNode<bool>(param.padded_mode);
  auto restore_shape = cg.NewValueNode(kNone);
  auto node =
    cg.NewCNode("InnerMoeTokenUnpermute", {permuted_tokens, sorted_indices, probs, padded_mode, restore_shape});
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

INSTANTIATE_TEST_CASE_P(TestSymShape, TestMoeTokenUnpermute,
                        testing::Values(MoeTokenUnpermuteOp{10, 20, 5, true, False},   //
                                        MoeTokenUnpermuteOp{10, 20, 10, false, False}  //
                                        ));
}  // namespace mindspore::symshape::test
