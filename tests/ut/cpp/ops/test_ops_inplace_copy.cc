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
std::vector<GeneralInferParam> prepare_params() {
    GeneralInferParamGenerator generator;
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{1}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{1}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{1}}, {kNumberTypeFloat16});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat16},
         InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat16});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},
         InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat64},
         InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat64},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3, 4}}, {kNumberTypeFloat64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt8},
         InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt8});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt16},
         InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt16});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt32},
         InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt32});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeInt64},
         InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeUInt8},
         InferInfoParam{ShapeVector{3}, kNumberTypeUInt8},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeUInt8});
   generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{1, 2}, kNumberTypeUInt16},
         InferInfoParam{ShapeVector{-2}, kNumberTypeUInt16},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{1, 2}}, {kNumberTypeUInt16});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeUInt32},
         InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeUInt32});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{2, 3}, kNumberTypeUInt64},
         InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 3}}, {kNumberTypeUInt64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{-1, -1}, kNumberTypeBool},
         InferInfoParam{ShapeVector{1, 2}, kNumberTypeBool},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{-1, -1}}, {kNumberTypeBool});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{-2}, kNumberTypeComplex64},
         InferInfoParam{ShapeVector{-1}, kNumberTypeComplex64},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeComplex64});
    generator
      .FeedInputArgs(
        {InferInfoParam{ShapeVector{-2}, kNumberTypeComplex128},
         InferInfoParam{ShapeVector{-1}, kNumberTypeComplex128},
         InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeComplex128});
    return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(InplaceCopy, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops