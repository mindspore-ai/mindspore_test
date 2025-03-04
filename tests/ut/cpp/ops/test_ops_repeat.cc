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
ValuePtr scalar(int64_t v) { return CreateScalar(v); }
ValuePtr scalar(ValuePtr v) { return v; }

template <typename... Args>
ValuePtrList scalar_list(Args... v) {
  return {scalar(v)...};
}

std::vector<InferInfoParam> repeat_input(ShapeVector shape, TypeId dtype) {
  return {
    InferInfoParam{ShapeVector{shape}, dtype},
    InferInfoParam{ShapeArray{}, TypeIdList{}, ValuePtrList{}},
  };
}

template <typename... T>
std::vector<InferInfoParam> repeat_input(ShapeVector shape, TypeId dtype, T... dim) {
  return {
    InferInfoParam{ShapeVector{shape}, dtype},
    InferInfoParam{ShapeVector{}, kNumberTypeInt64, scalar_list(dim...)},
  };
}

std::vector<GeneralInferParam> prepare_params() {
  // Only shape and type of the first operand matters. Ignore uncecessary dim/input/value.
  GeneralInferParamGenerator generator;
  return generator
    // empty (scalar)
    .FeedInputArgs(repeat_input({}, kNumberTypeFloat64))
    .FeedExpectedOutput({{}}, {kNumberTypeFloat64})
    // scalar in, tensor out
    .FeedInputArgs(repeat_input({}, kNumberTypeFloat64, 4, 5, 6))
    .FeedExpectedOutput({{4, 5, 6}}, {kNumberTypeFloat64})
    // normal 1 (rank equals)
    .FeedInputArgs(repeat_input({3}, kNumberTypeFloat64, 4))
    .FeedExpectedOutput({{12}}, {kNumberTypeFloat64})
    .FeedInputArgs(repeat_input({6, 2, 3}, kNumberTypeFloat64, 4, 5, 6))
    .FeedExpectedOutput({{24, 10, 18}}, {kNumberTypeFloat64})
    // normal 2 (repeat length larger)
    .FeedInputArgs(repeat_input({2, 3}, kNumberTypeFloat16, 4, 5, 6))
    .FeedExpectedOutput({{4, 10, 18}}, {kNumberTypeFloat16})
    // error: repeat length smaller
    .FeedInputArgs(repeat_input({1}, kNumberTypeFloat16))
    .CaseShouldThrow()
    .FeedInputArgs(repeat_input({1, 2}, kNumberTypeFloat16, 1))
    .CaseShouldThrow()
    .FeedInputArgs(repeat_input({-1, 2}, kNumberTypeFloat16, 1))
    .CaseShouldThrow()
    // dyn shapes (some of self shape unknown)
    .FeedInputArgs(repeat_input({-1, 3}, kNumberTypeFloat16, 4, 5))
    .FeedExpectedOutput({{-1, 15}}, {kNumberTypeFloat16})
    .FeedInputArgs(repeat_input({2, -1, -1}, kNumberTypeFloat16, 3, 4, 5, 6))
    .FeedExpectedOutput({{3, 8, -1, -1}}, {kNumberTypeFloat16})
    // dyn shapes (some of repeats value unknown)
    .FeedInputArgs(repeat_input({2, 3, 4, -1}, kNumberTypeFloat32, kValueAny, 3, 4, kValueAny))
    .FeedExpectedOutput({{-1, 9, 16, -1}}, {kNumberTypeFloat32})
    .FeedInputArgs(repeat_input({2, 3}, kNumberTypeFloat32, 2, kValueAny, kValueAny, 3))
    .FeedExpectedOutput({{2, -1, -1, 9}}, {kNumberTypeFloat32})

    // dyn rank (self shape unknown / repeats unknown)
    .FeedInputArgs(repeat_input({-2}, kNumberTypeBFloat16, 2, 3, 4))
    .FeedExpectedOutput({{-2}}, {kNumberTypeBFloat16})
    .FeedInputArgs({
      InferInfoParam{ShapeVector{{2, 3}}, kNumberTypeDouble},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},
    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeDouble})
    .FeedInputArgs({
      InferInfoParam{ShapeVector{{-2}}, kNumberTypeDouble},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, kValueAny},
    })
    .FeedExpectedOutput({{-2}}, {kNumberTypeDouble})
    // Negative repeats
    // Case 1: Final output is zero, ok
    .FeedInputArgs(repeat_input({2, 0}, kNumberTypeFloat16, 3, 4, -5))
    .FeedExpectedOutput({{3, 8, 0}}, {kNumberTypeFloat16})
    // Case 2: Final output has negative, raise
    .FeedInputArgs(repeat_input({2, 0}, kNumberTypeFloat16, 3, -4, -5))
    .CaseShouldThrow()
    .Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Repeat, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
