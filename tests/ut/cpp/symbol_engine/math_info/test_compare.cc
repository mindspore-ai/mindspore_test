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

#include "symbol_engine/math_info/symbol_test_utils.h"
#include "mindspore/ops/infer/symbol_ops_impl/operator_scope.h"

namespace mindspore::symshape::test {
/// Feature: IntSymbol compare
/// Description: s1 > 0; s2 = s1-1; s3 = s1*2-2
/// Expectation: "s2 <= s3" but not "s2 < s3".
TEST_F(TestMathInfo, cmp_1) {
  auto s1 = IntSymbol::Make();
  s1->SetRangeMin(1);
  ops::OperatorScope h(helper_->emitter());
  auto hs1 = h(s1);
  auto s2 = (hs1 - kSym1).as<IntSymbol>();
  auto s3 = (hs1 * kSym2 - kSym2).as<IntSymbol>();
  EXPECT_TRUE(*s2 <= *s3);
  EXPECT_TRUE(*s3 >= *s2);
  EXPECT_FALSE(*s2 < *s3);
  EXPECT_FALSE(*s3 > *s2);
  EXPECT_FALSE(*s3 < *s2);
  EXPECT_FALSE(*s3 == *s2);
}

/// Feature: IntSymbol compare
/// Description: s1 > 1; s2 = s1-1; s3 = s1*2-2
/// Expectation: s2 < s3
TEST_F(TestMathInfo, cmp_2) {
  auto s1 = IntSymbol::Make();
  s1->SetRangeMin(2);
  ops::OperatorScope h(helper_->emitter());
  auto hs1 = h(s1);
  auto s2 = (hs1 - kSym1).as<IntSymbol>();
  auto s3 = (hs1 * kSym2 - kSym2).as<IntSymbol>();
  EXPECT_TRUE(*s2 <= *s3);
  EXPECT_TRUE(*s3 >= *s2);
  EXPECT_TRUE(*s2 < *s3);
  EXPECT_TRUE(*s3 > *s2);
  EXPECT_FALSE(*s2 == *s3);
}

/// Feature: IntSymbol compare
/// Description: s1 = 8N; s2 = s1 / 8
/// Expectation: s2 < s1
TEST_F(TestMathInfo, cmp_3) {
  auto s1 = IntSymbol::Make();
  s1->SetDivisorRemainder(8, 0);
  auto s2 = helper_->Emit(std::make_shared<ops::ScalarDiv>(s1, IntSymbol::Make(8)))->as_sptr<IntSymbol>();
  EXPECT_EQ(s1->range_min(), 8);
  EXPECT_TRUE(*s1 > *s2);
  EXPECT_TRUE(*s1 >= *s2);
  EXPECT_TRUE(*s2 < *s1);
  EXPECT_TRUE(*s2 <= *s1);
  EXPECT_FALSE(*s2 >= *s1);
}
}  // namespace mindspore::symshape::test
