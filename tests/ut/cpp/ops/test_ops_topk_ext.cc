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
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(-1)},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
    })
    .FeedExpectedOutput({{3}, {3}}, {kNumberTypeFloat16, kNumberTypeInt64});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 8, 5}, kNumberTypeFloat32},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(0)},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
    })
    .FeedExpectedOutput({{3, 8, 5}, {3, 8, 5}}, {kNumberTypeFloat32, kNumberTypeInt64});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 8, 5}, kNumberTypeFloat64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
    })
    .FeedExpectedOutput({{10, 3, 5}, {10, 3, 5}}, {kNumberTypeFloat64, kNumberTypeInt64});
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{10, 8, 5}, kNumberTypeFloat16},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
      InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(true)},
    })
    .FeedExpectedOutput({{10, 8, 3}, {10, 8, 3}}, {kNumberTypeFloat16, kNumberTypeInt64});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(TopkExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
