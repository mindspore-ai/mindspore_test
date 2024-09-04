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
#include "mindspore/ops/infer/symbol_ops_impl/reshape.h"

namespace mindspore::symshape::test {
class TestReshape : public TestOperation {};

/// Feature: Symbolic shape for Reshape
/// Description: output shape items are all positive
/// Expectation: transparent output shape, ops list is empty.
TEST_F(TestReshape, case_1) {
  SymbolPtrList x_list{GenPVInt(), GenPVInt()};
  SymbolPtrList shape_list{GenPVInt(), GenPVInt(), GenPVInt()};
  auto x = GenList(x_list);
  auto shape = GenList(shape_list);
  auto out = helper_->Emit(std::make_shared<ops::Reshape>(x, shape));
  EXPECT_TRUE(out->EqualsTo(shape));
  EXPECT_TRUE(helper_->ops_list_.empty());
}

/// Feature: Symbolic shape for Reshape
/// Description: output shape items are got from inputs (no positive info).
/// Expectation: transparent output shape, ops list is empty, after operation, the symbols are set to positive.
TEST_F(TestReshape, case_2) {
  SymbolPtrList x_list{GenVInt(), GenVInt(), GenVInt()};
  SymbolPtrList shape_list{x_list[2], x_list[0]};
  auto x = GenList(x_list);
  auto shape = GenList(shape_list);
  auto out = helper_->Emit(std::make_shared<ops::Reshape>(x, shape));
  EXPECT_TRUE(out->EqualsTo(shape));
  EXPECT_TRUE(helper_->ops_list_.empty());
  for (auto s : out->as<ListSymbol>()->symbols()) {
    ASSERT_TRUE(s->as<IntSymbol>()->is_positive()) << out->ToString();
  }
}

/// Feature: Symbolic shape for Reshape
/// Description: output shape has only one unknown dim, output const dim is divisible by input const dim
/// Expectation: the unknown dim is calculated by inputs
TEST_F(TestReshape, case_3) {
  SymbolPtrList x_list{GenInt(32), GenPVInt(), GenPVInt(), GenVInt()};
  SymbolPtrList shape_list{GenVInt(), GenInt(8), GenInt(2), x_list[3]};
  auto x = GenList(x_list);
  auto shape = GenList(shape_list);
  auto out = helper_->Emit(std::make_shared<ops::Reshape>(x, shape))->as_sptr<ListSymbol>();
  ASSERT_EQ(out->size(), 4);
  ASSERT_TRUE(out->item_as<IntSymbol>(0)->is_divisible_by(2));
  ASSERT_EQ(helper_->ops_list_.size(), 1);
}

/// Feature: Symbolic shape for Reshape
/// Description: output shape has only one unknown dim, output const dim is not divisible by input const dim
/// Expectation: the unknown dim is calculated by inputs
TEST_F(TestReshape, case_4) {
  SymbolPtrList x_list{GenInt(32), GenPVInt(), GenPVInt(), GenVInt()};
  SymbolPtrList shape_list{GenVInt(), GenInt(5), GenInt(2), x_list[3]};
  auto x = GenList(x_list);
  auto shape = GenList(shape_list);
  auto out = helper_->Emit(std::make_shared<ops::Reshape>(x, shape))->as_sptr<ListSymbol>();
  ASSERT_EQ(out->size(), 4);
  ASSERT_EQ(out->item_as<IntSymbol>(0)->divisor(), 1);
  ASSERT_EQ(helper_->ops_list_.size(), 1);
}

/// Feature: Symbolic shape for Reshape
/// Description: output shape has a "-1" dim, and two variable dims. and test infer interface.
/// Expectation: output use the same pointers of two variable dims.
TEST_F(TestReshape, case_5) {
  SymbolPtrList x_list{GenInt(32), GenPVInt(), GenPVInt(), GenPVInt()};
  SymbolPtrList shape_list{GenVInt(), GenVInt(), GenInt(8), GenInt(-1)};
  auto x = GenList(x_list);
  auto shape = GenList(shape_list);
  auto out = helper_->Emit(std::make_shared<ops::Reshape>(x, shape))->as_sptr<ListSymbol>();
  ASSERT_EQ(out->size(), 4);
  EXPECT_EQ(out->item(0), shape_list[0]);  // use the same ptr
  EXPECT_EQ(out->item(1), shape_list[1]);  // use the same ptr
  ASSERT_EQ(helper_->ops_list_.size(), 1);
  helper_->Infer({{x, {32, 8, 2, 1}}, {shape, {16, 1, 8, -1}}});
  EXPECT_EQ(out->item_as<IntSymbol>(3)->value(), 4);
  helper_->Infer({{x, {32, 8, 2, 1}}, {shape, {16, 2, 8, -1}}});
  EXPECT_EQ(out->item_as<IntSymbol>(3)->value(), 2);
}
}  // namespace mindspore::symshape::test
