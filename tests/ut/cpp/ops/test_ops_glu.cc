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

#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
class GluCaseGenerator {
 public:
  GluCaseGenerator() : generator() {}
  auto cases() {
    // scalar raises
    add_case({}, kNumberTypeFloat64, 0);
    add_case({}, kNumberTypeFloat64, kValueAny);
    // Static Shape + Static Value
    add_case({2}, kNumberTypeFloat64, 0, {1});
    add_case({3, 4}, kNumberTypeFloat32, 1, {3, 2});
    add_case({2}, kNumberTypeFloat16, -1, {1});
    add_case({2, 3, 8}, kNumberTypeBFloat16, -3, {1, 3, 8});
    add_case({2, 3, 8}, kNumberTypeFloat16, 2, {2, 3, 4});
    add_case({2}, kNumberTypeFloat16, 1);         // dim out of range
    add_case({2, 3, 8}, kNumberTypeFloat16, -4);  // dim out of range
    add_case({2, 3, 8}, kNumberTypeFloat16, 3);   // dim out of range
    add_case({2, 3, 8}, kNumberTypeFloat16, 1);   // shape[dim] odd
    add_case({2, 3, 9}, kNumberTypeFloat16, -1);  // shape[dim] odd
    add_case({3, 2, 8}, kNumberTypeFloat16, 0);   // shape[dim] odd
    // Dynamic shape
    add_case({-1, -1}, kNumberTypeBFloat16, 1, {-1, -1});
    add_case({-1, -1}, kNumberTypeBFloat16, -2, {-1, -1});
    add_case({-1, -1}, kNumberTypeBFloat16, -3);  // dim out of range
    add_case({-1, -1}, kNumberTypeBFloat16, 2);   // dim out of range
    add_case({-1, 3}, kNumberTypeBFloat16, 1);    // shape[dim] odd
    add_case({-1, 3}, kNumberTypeBFloat16, -1);   // shape[dim] odd
    add_case({-1, 5}, kNumberTypeBFloat16, -2, {-1, 5});
    add_case({4, -1}, kNumberTypeBFloat16, -2, {2, -1});
    add_case({4, -1}, kNumberTypeBFloat16, 1, {4, -1});
    add_case({4, 2}, kNumberTypeBFloat16, kValueAny, {-1, -1});
    add_case({4, 3}, kNumberTypeBFloat16, kValueAny, {-1, -1});
    // Dynamic rank
    add_case({-2}, kNumberTypeBFloat16, 2, {-2});
    add_case({-2}, kNumberTypeBFloat16, -2, {-2});
    add_case({-2}, kNumberTypeBFloat16, kValueAny, {-2});
    return generator.Generate();
  }

 private:
  GeneralInferParamGenerator generator;
  void add_case(ShapeVector &&self, TypeId dtype, int64_t dim) { add_case(std::move(self), dtype, CreateScalar(dim)); }
  void add_case(ShapeVector &&self, TypeId dtype, int64_t dim, ShapeVector &&out) {
    add_case(std::move(self), dtype, CreateScalar(dim), std::move(out));
  }
  void add_case(ShapeVector &&self, TypeId dtype, ValuePtr dim) {
    generator.FeedInputArgs(gen_input(std::move(self), dtype, dim)).CaseShouldThrow();
  }
  void add_case(ShapeVector &&self, TypeId dtype, ValuePtr dim, ShapeVector &&out) {
    generator.FeedInputArgs(gen_input(std::move(self), dtype, dim)).FeedExpectedOutput({out}, {dtype});
  }
  std::vector<mindspore::ops::InferInfoParam> gen_input(ShapeVector &&self, TypeId dtype, ValuePtr dim) {
    return {
      InferInfoParam{self, dtype},                          // self
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, dim}  // dim
    };
  }
};
}  // namespace

INSTANTIATE_TEST_CASE_P(GLU, GeneralInferTest, testing::ValuesIn(GluCaseGenerator().cases()));
}  // namespace mindspore::ops
