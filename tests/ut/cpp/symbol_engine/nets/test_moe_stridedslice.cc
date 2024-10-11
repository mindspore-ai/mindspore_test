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

#include "symbol_engine/nets/test_net.h"
#include "common/graph_optimizer_test_framework.h"

namespace mindspore::symshape::test {
/// Feature: SymbolicShape
/// Description: #IAWB6F, the slice index is calculated from onehot, but it does not support infer value.
/// Expectation: the dynamic output dim of StridedSlice has divisor info.
TEST_F(TestNet, moe_stridedslice) {
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kInt32, {2, 8192});
  auto y = cg.NewTensorInput("y", kFloat32, {2, 8, 8192});
  auto val0 = cg.NewValueNode(MakeValue<int64_t>(0));
  auto val1 = cg.NewValueNode(MakeValue<int64_t>(1));
  auto val2 = cg.NewValueNode(MakeValue<int64_t>(2));
  int64_t on_val = 1LL;
  int64_t off_val = 0LL;
  auto t1 = cg.NewCNode(
    "OneHot",
    {x, cg.NewValueNode(MakeValue<int64_t>(8)), cg.NewValueNode(std::make_shared<tensor::Tensor>(on_val, kInt64)),
     cg.NewValueNode(std::make_shared<tensor::Tensor>(off_val, kInt64)), cg.NewValueNode(MakeValue<int64_t>(-1))});
  auto t2 =
    cg.NewCNode("ReduceSum", {t1, cg.NewValueNode(MakeValue<std::vector<int64_t>>({1})),
                              cg.NewValueNode(MakeValue<bool>(false)), cg.NewValueNode(MakeValue<bool>(false))});
  auto t3 = cg.NewCNode("Max", {t2});
  auto t4 = cg.NewCNode("Cast", {t3, cg.NewValueNode(MakeValue<int64_t>(35))});  // from int32 to int64
  auto t5 = cg.NewCNode("TensorToScalar", {t4});
  auto t6 = cg.NewCNode("ScalarFloorDiv", {t5, val2});
  auto t7 = cg.NewCNode("ScalarAdd", {t6, val1});
  auto t8 = cg.NewCNode("ScalarMul", {t7, val2});
  auto t9 = cg.NewCNode("MakeTuple", {val2, cg.NewValueNode(MakeValue<int64_t>(8)), t8});
  auto m = val0;
  auto node = cg.NewCNode("StridedSlice", {y, cg.NewValueNode(MakeValue<std::vector<int64_t>>({0, 0, 0})), t9,
                                           cg.NewValueNode(MakeValue<std::vector<int64_t>>({1, 1, 1})), m, m, m, m, m});
  cg.GetGraph()->set_output(node);
  helper_->BuildSymbolEngine(cg.GetGraph());
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  ASSERT_EQ(node->abstract()->GetSymbolicShape()->item_as<IntSymbol>(2)->divisor(), 2);
}
}  // namespace mindspore::symshape::test
