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
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},                        // input
                      InferInfoParam{ShapeArray{}, TypeIdList{}, CreateScalarList<int64_t>({})},    // dim
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},    // correction
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})   // keepdim
      .FeedExpectedOutput({{}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeFloat32, mindspore::kNone},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4}, kNumberTypeFloat32},
                      InferInfoParam{ShapeArray{}, TypeIdList{}, CreateScalarList<int64_t>({})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
      .FeedExpectedOutput({{1, 1, 1}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4, 5}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({1, 3})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 4}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4, 5}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({1, 3})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
      .FeedExpectedOutput({{2, 1, 4, 1}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4, 5, 6}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({0, 1, 2, 3, 4})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4, 5, 6}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({0, 1, 2, 3, 4})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
      .FeedExpectedOutput({{1, 1, 1, 1, 1}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, 4, 5, 6}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({-2, -4})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, 4, 6}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeArray{}, TypeIdList{}, CreateScalarList<int64_t>({})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, 3, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeArray{}, TypeIdList{}, CreateScalarList<int64_t>({})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
      .FeedExpectedOutput({{1, 1, 1}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{2, -1, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({1})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{2, -1}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1, -1}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({0, 1, 2})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeArray{}, TypeIdList{}, CreateScalarList<int64_t>({})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
      .FeedExpectedOutput({{}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeArray{}, TypeIdList{}, CreateScalarList<int64_t>({})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
    generator
      .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalarList<int64_t>({0})},
                      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
      .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
    
    return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Std, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
