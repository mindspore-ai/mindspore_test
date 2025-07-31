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

#include "ir/tensor_new.h"
#include "symbol_engine/nets/test_net.h"
#include "common/graph_optimizer_test_framework.h"

namespace mindspore::symshape::test {
/// Feature: SymbolicShape
/// Description: #IAWB6F, the slice index is calculated from onehot, but it does not support infer value.
/// Expectation: the dynamic output dim of StridedSlice has divisor info.
TEST_F(TestNet, moe_stridedslice_1) {
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
    {x, cg.NewValueNode(MakeValue<int64_t>(8)), cg.NewValueNode(tensor::from_scalar(on_val, kInt64)),
     cg.NewValueNode(tensor::from_scalar(off_val, kInt64)), cg.NewValueNode(MakeValue<int64_t>(-1))});
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

/// Feature: SymbolicShape
/// Description: todo.
/// Expectation: todo.
TEST_F(TestNet, moe_stridedslice_2) {
  mindspore::test::ConstructGraph cg;
  auto x = cg.NewTensorInput("x", kInt32, {8, 4096});
  auto y = cg.NewTensorInput("y", kInt32, {8, 2048, 2});
  auto z = cg.NewTensorInput("z", kFloat16, {8, 2049, 4096});
  auto val0 = cg.NewValueNode<int64_t>(0);
  auto val1 = cg.NewValueNode<int64_t>(1);
  auto val2 = cg.NewValueNode<int64_t>(2);
  auto val4 = cg.NewValueNode<int64_t>(4);
  auto val8 = cg.NewValueNode<int64_t>(8);
  auto val4096 = cg.NewValueNode<int64_t>(4096);
  int64_t on_val = 1LL;
  int64_t off_val = 0LL;
  auto t1 = cg.NewCNode(
    "OneHot", {x, val8, cg.NewValueNode(tensor::from_scalar(on_val, kInt64)),
               cg.NewValueNode(tensor::from_scalar(off_val, kInt64)), cg.NewValueNode<int64_t>(-1)});
  auto t2 = cg.NewCNode("ReduceSum",
                        {t1, cg.NewValueNode(std::vector<int64_t>{1}), cg.NewValueNode(false), cg.NewValueNode(false)});
  auto t3 = cg.NewCNode("Max", {t2});
  auto t4 = cg.NewCNode("Cast", {t3, cg.NewValueNode<int64_t>(35)});  // from int32 to int64
  auto t5 = cg.NewCNode("TensorToScalar", {t4});
  auto t6 = cg.NewCNode("ScalarFloorDiv", {t5, val1});
  auto t7 = cg.NewCNode("ScalarAdd", {t6, val1});
  auto t8 = cg.NewCNode("TopKRouter", {y, t7, val8, val0});
  auto t9 = cg.NewCNode("TupleGetItem", {t8, val0});
  auto t10 = cg.NewCNode("Gather", {z, t9, val1, val1});
  auto t11 = cg.NewCNode("Split", {t10, val2, val2});
  auto t12 = cg.NewCNode("TupleGetItem", {t11, val0});
  auto t13 = cg.NewCNode("Shape", {t9});
  auto t14 = cg.NewCNode("TupleGetItem", {t13, val2});
  auto t15 = cg.NewCNode("ScalarFloorDiv", {t14, val2});
  auto t16 = cg.NewCNode("MakeTuple", {val4, val2, val8, t15, val4096});
  auto t17 = cg.NewCNode("Reshape", {t12, t16});
  auto t18 = cg.NewCNode("Transpose", {t17, cg.NewValueNode(std::vector<int64_t>{0, 2, 1, 3, 4})});
  auto t19 = cg.NewCNode("ScalarMul", {t15, val2});
  auto t20 = cg.NewCNode("MakeTuple", {val1, val4, val8, t19, val4096});
  auto t21 = cg.NewCNode("Reshape", {t18, t20});
  auto m = val0;
  auto node = cg.NewCNode("StridedSlice", {t21, cg.NewValueNode(std::vector<int64_t>(5, 0)), t20,
                                           cg.NewValueNode(std::vector<int64_t>(5, 1)), m, m, m, m, m});
  cg.GetGraph()->set_output(node);
  helper_->BuildSymbolEngine(cg.GetGraph());
  SaveIR(cg.GetGraph());
  ASSERT_TRUE(helper_->CheckSymbolicShapeMatchesDigitalShape(node));
  ASSERT_EQ(node->abstract()->GetSymbolicShape()->item_as<IntSymbol>(3)->divisor(), 2);
}
}  // namespace mindspore::symshape::test
