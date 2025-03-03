/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4}, kNumberTypeFloat16},                        // x
                    InferInfoParam{ShapeVector{3, 5}, kNumberTypeInt32},                          // expert_idx
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},    // active_num
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},    // expert_capacity
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},  // expert_num
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},    // drop_pad_mode
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{15, 4}, {15}, {256}, {256}},
                        {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{3, 5}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{256, 2, 4}, {15}, {256}, {256}},
                        {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{3, 5}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{15, 4}, {15}, {256}, {256}},
                        {kNumberTypeBFloat16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 4}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{3, 5}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{256, 2, 4}, {15}, {256}, {256}},
                        {kNumberTypeBFloat16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 4}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{-1, 5}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{-1, 4}, {-1}, {256}, {256}},
                        {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 4}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{-1, 5}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{256, 2, 4}, {-1}, {256}, {256}},
                        {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32});
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(256)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)}})
    .FeedExpectedOutput({{-2}, {-1}, {-1}, {-1}},
                        {kNumberTypeFloat16, kNumberTypeInt32, kNumberTypeInt32, kNumberTypeInt32});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(MoeInitRoutingV2, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
