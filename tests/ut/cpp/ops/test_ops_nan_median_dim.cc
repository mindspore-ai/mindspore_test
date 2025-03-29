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
class NanMedianDimCaseGenerator {
 public:
  NanMedianDimCaseGenerator() : generator() {}
  auto cases() {
    // A.  Scalar: keepdim does not affect the result
    // A.1 Static Shape + Static Value
    add_case({}, kNumberTypeFloat32, 0, false, {});
    add_case({}, kNumberTypeFloat32, 0, true, {});
    add_case({}, kNumberTypeFloat16, -1, true, {});
    add_case({}, kNumberTypeFloat16, -2, true);  // dim out of range
    add_case({}, kNumberTypeFloat16, 1, false);  // dim out of range

    // A.2 Static Shape + Dynamic Value (Dynamic Shape is treated as Tensor)
    add_case({}, kNumberTypeFloat32, kValueAny, false, {-1});
    add_case({}, kNumberTypeFloat32, kValueAny, kValueAny, {-1});
    add_case({}, kNumberTypeFloat32, 0, kValueAny, {});
    add_case({}, kNumberTypeFloat32, -1, kValueAny, {});
    add_case({}, kNumberTypeFloat32, 1, kValueAny);   // dim out of range
    add_case({}, kNumberTypeFloat32, -2, kValueAny);  // dim out of range

    // B.  Tensor
    // B.1 Static Shape + Static Value
    add_case({2}, kNumberTypeFloat16, 0, false, {});
    add_case({2}, kNumberTypeFloat16, -1, true, {1});
    add_case({2, 3, 4, 5}, kNumberTypeFloat32, 0, false, {3, 4, 5});
    add_case({2, 3, 4, 5}, kNumberTypeFloat32, 2, false, {2, 3, 5});
    add_case({2, 3, 4, 5}, kNumberTypeFloat32, 3, true, {2, 3, 4, 1});
    add_case({2, 3, 4, 5}, kNumberTypeFloat32, -4, true, {1, 3, 4, 5});
    add_case({2, 3, 4, 5}, kNumberTypeFloat32, -4, false, {3, 4, 5});
    add_case({2}, kNumberTypeFloat16, 1, false);         // dim out of range
    add_case({2}, kNumberTypeFloat16, 1, true);          // dim out of range
    add_case({2, 3, 8}, kNumberTypeFloat16, -4, false);  // dim out of range
    add_case({2, 3, 8}, kNumberTypeFloat16, 3, false);   // dim out of range

    // B.2 Static shape to dynamic (rank or shape)
    add_case({2, 3, 4, 5}, kNumberTypeFloat32, 0, kValueAny, {-2});
    add_case({2, 3, 4, 5}, kNumberTypeFloat32, kValueAny, true, {-1, -1, -1, -1});
    add_case({2, 3, 4, 5}, kNumberTypeFloat32, kValueAny, false, {-1, -1, -1});

    // B.3 Other dynamic shape cases
    add_case({-1, -1}, kNumberTypeBFloat16, 1, true, {-1, 1});  // no futrher check about input.shape[dim]
    add_case({-1, -1}, kNumberTypeBFloat16, -2, false, {-1});
    add_case({3, -1}, kNumberTypeBFloat16, -3, true);  // dim out of range
    add_case({3, -1}, kNumberTypeBFloat16, 2, true);   // dim out of range
    add_case({2, 3, -1, 5}, kNumberTypeFloat32, kValueAny, true, {-1, -1, -1, -1});
    add_case({2, -1, 4, 5}, kNumberTypeFloat32, kValueAny, false, {-1, -1, -1});
    add_case({4, -1}, kNumberTypeBFloat16, -1, false, {4});
    // B.4 Dynamic rank
    add_case({-2}, kNumberTypeBFloat16, -2, true, {-2});
    add_case({-2}, kNumberTypeBFloat16, -1, true, {-2});
    add_case({-2}, kNumberTypeBFloat16, 0, false, {-2});
    add_case({-2}, kNumberTypeBFloat16, kValueAny, true, {-2});
    add_case({1, 2, 3}, kNumberTypeBFloat16, 2, kValueAny, {-2});
    add_case({1, 2, 3}, kNumberTypeBFloat16, kValueAny, kValueAny, {-2});
    return generator.Generate();
  }

 private:
  GeneralInferParamGenerator generator;
  static ValuePtr Dim(int64_t v) { return CreateScalar(v); }
  static ValuePtr Keep(const bool &v) { return CreateScalar(v); }
  static ValuePtr Dim(ValuePtr v) { return v; }
  static ValuePtr Keep(ValuePtr v) { return v; }
  template <typename D, typename K>
  void add_case(ShapeVector shape, TypeId dtype, D dim, K keep) {
    generator.FeedInputArgs(gen_input(shape, dtype, dim, keep)).CaseShouldThrow();
  }
  template <typename D, typename K>
  void add_case(ShapeVector shape, TypeId dtype, D dim, K keep, ShapeVector &&out) {
    generator.FeedInputArgs(gen_input(shape, dtype, dim, keep))
      .FeedExpectedOutput({out, out}, {dtype, kNumberTypeInt64});
  }
  template <typename D, typename K>
  std::vector<mindspore::ops::InferInfoParam> gen_input(ShapeVector shape, TypeId dtype, D dim, K keep) {
    return {
      InferInfoParam{shape, dtype},                               // self
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, Dim(dim)},  // dim
      InferInfoParam{ShapeVector{}, kNumberTypeBool, Keep(keep)}  // keepdim
    };
  }
};
}  // namespace

INSTANTIATE_TEST_CASE_P(NanMedianDim, GeneralInferTest, testing::ValuesIn(NanMedianDimCaseGenerator().cases()));
}  // namespace mindspore::ops
