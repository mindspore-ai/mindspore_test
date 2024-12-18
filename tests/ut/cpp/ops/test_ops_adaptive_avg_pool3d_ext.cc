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
    .FeedInputArgs({InferInfoParam{ShapeVector{4, 2, 3, 5}, kNumberTypeFloat16},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                    ValuePtrList{CreateScalar<int64_t>(2), CreateScalar<int64_t>(4), CreateScalar<int64_t>(4)}}})
    .FeedExpectedOutput({{4, 2, 4, 4}}, {kNumberTypeFloat16});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{4, 3, 4, 5, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                    ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(-1), CreateScalar<int64_t>(1)}}})
    .FeedExpectedOutput({{4, 3, 1, 5, 1}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 3, 4, 5}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                    ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(-1), CreateScalar<int64_t>(1)}}})
    .FeedExpectedOutput({{-1, 1, 4, 1}}, {kNumberTypeBFloat16});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, -1, 3, 5}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                    ValuePtrList{CreateScalar<int64_t>(1), CreateScalar<int64_t>(-1), CreateScalar<int64_t>(-1)}}})
    .FeedExpectedOutput({{3, 1, 3, 5}}, {kNumberTypeFloat32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeArray{{}, {}, {}}, kNumberTypeInt64,
                    ValuePtrList{CreateScalar<int64_t>(4), CreateScalar<int64_t>(6), CreateScalar<int64_t>(6)}}})
    .FeedExpectedOutput({{-2}}, {kNumberTypeFloat32});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(AdaptiveAvgPool3DExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
