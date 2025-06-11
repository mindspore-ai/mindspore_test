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
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(7)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{6, 10}, {6}}, {kNumberTypeFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, 10}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{5, 10}, {6}}, {kNumberTypeBFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(6)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{6, -1}, {6}}, {kNumberTypeFloat32, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{5, -1}, {6}}, {kNumberTypeBFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{3, -1}, {6}}, {kNumberTypeFloat32, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{5, -1}, {6}}, {kNumberTypeBFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1000)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{6, -1}, {6}}, {kNumberTypeFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{3, 2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{5, -1}, {6}}, {kNumberTypeBFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{3, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{3, -1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{-1, -1}, {-1}}, {kNumberTypeBFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{3, -1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(123)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{-1, -1}, {-1}}, {kNumberTypeFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{-1, -1}, {-1}}, {kNumberTypeBFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeBFloat16},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{-1, -1}, {-1}}, {kNumberTypeBFloat16, kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(5)},
                    InferInfoParam{ShapeVector{}, kNumberTypeBool, CreateScalar<bool>(false)}})
    .FeedExpectedOutput({{-1, -1}, {-1}}, {kNumberTypeFloat32, kNumberTypeInt32});
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(MoeTokenPermute, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
