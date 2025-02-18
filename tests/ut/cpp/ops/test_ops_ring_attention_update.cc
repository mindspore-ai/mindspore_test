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
    .FeedInputArgs({
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{3, 10, 64}, {10, 8, 3, 8}, {10, 8, 3, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 3, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{10, 3, 64}, {3, 8, 10, 8}, {3, 8, 10, 8}},
                        {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, 10, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{3, 10, 64}, {10, 8, 3, 8}, {10, 8, 3, 8}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, -1, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{10, 3, 64}, {3, 8, 10, 8}, {3, 8, 10, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{3, 10, 64}, {10, 8, 3, 8}, {10, 8, 3, 8}},
                        {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, -1, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 8, -1, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 8, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{10, 3, 64}, {3, 8, 10, 8}, {3, 8, 10, 8}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{10, 8, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{10, 8, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{3, 10, 64}, {10, 8, 3, 8}, {10, 8, 3, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{-1, 8, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{-1, 8, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, 8, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{3, -1, 64}, {-1, 8, 3, 8}, {-1, 8, 3, 8}},
                        {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, -1, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, -1, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, -1, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, -1, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{-1, -1, 64}, {-1, -1, -1, 8}, {-1, -1, -1, 8}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
    })
    .FeedExpectedOutput({{-1, -1, -1}, {-1, -1, -1, 8}, {-1, -1, -1, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{3, 10, 64}, {3, 10, 8}, {3, 10, 8}},
                        {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 3, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{10, 3, 64}, {10, 3, 8}, {10, 3, 8}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, 10, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, 10, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{3, 10, 64}, {3, 10, 8}, {3, 10, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, -1, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{10, 3, 64}, {10, 3, 8}, {10, 3, 8}},
                        {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{10, 3, 64}, {10, 3, 8}, {10, 3, 8}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, -1, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{-1, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{-1, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{10, 3, 64}, {10, 3, 8}, {10,3, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{10, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{10, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 3, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{10, 3, 64}, {10, 3, 8}, {10, 3, 8}},
                        {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, -1, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, -1, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, -1, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, -1, -1}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{10, -1, 64}, {10, -1, 8}, {10, -1, 8}},
                        {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{-1, -1, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{-1, -1, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-1, -1, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{-1, -1, 64}, {-1, -1, 8}, {-1, -1, 8}},
                        {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(4)},
    })
    .FeedExpectedOutput({{-1, -1, -1}, {-1, -1, 8}, {-1, -1, 8}},
                        {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{10, 8, 3, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
    })
    .FeedExpectedOutput({{-1, -1, -1}, {-2}, {-2}}, {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 64}, kNumberTypeBFloat16},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3, 10, 8}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
    })
    .FeedExpectedOutput({{-1, -1, -1}, {-2}, {-2}}, {kNumberTypeBFloat16, kNumberTypeFloat32, kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(RingAttentionUpdate, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
