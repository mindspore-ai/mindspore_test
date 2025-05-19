/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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
namespace  {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  // static
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{4, 3, 3}, kNumberTypeUInt8},
                      InferInfoParam{ShapeVector{3, 3}, kNumberTypeBool},
                      InferInfoParam{ShapeVector{}, kNumberTypeUInt8, CreateScalar<float>(1.0)}})
      .FeedExpectedOutput({{4, 3, 3}}, {kNumberTypeUInt8});
  // dynamic shape
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{4, -1, -1}, kNumberTypeInt16},
                      InferInfoParam{ShapeVector{3, 3}, kNumberTypeBool},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt16, CreateScalar<float>(1.0)}})
      .FeedExpectedOutput({{4, 3, 3}}, {kNumberTypeInt16});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{4, 3, 3}, kNumberTypeInt32},
                      InferInfoParam{ShapeVector{-1, -1}, kNumberTypeBool},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt32, CreateScalar<float>(1.0)}})
      .FeedExpectedOutput({{4, 3, 3}}, {kNumberTypeInt32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{-1, -1}, kNumberTypeBool},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.0)}})
      .FeedExpectedOutput({{-1, -1}}, {kNumberTypeFloat32});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat64},
                      InferInfoParam{ShapeVector{-2}, kNumberTypeBool},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<float>(1.0)}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeFloat64});
  generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeComplex128},
                      InferInfoParam{ShapeVector{1}, kNumberTypeBool},
                      InferInfoParam{ShapeVector{}, kNumberTypeComplex128, CreateScalar<float>(1.0)}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeComplex128});
  return generator.Generate();
}
}  //namespace
INSTANTIATE_TEST_CASE_P(MaskedFill, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
