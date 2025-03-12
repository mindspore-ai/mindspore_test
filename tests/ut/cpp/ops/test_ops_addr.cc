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
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},                       // input
                    InferInfoParam{ShapeVector{2}, kNumberTypeFloat32},                          // vec1
                    InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},                          // vec2
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},   // beta
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)}})  // alpha
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{3}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{5}, kNumberTypeInt64},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)}})
    .FeedExpectedOutput({{3, 5}}, {kNumberTypeInt64});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{4, 1}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{4}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{2}, kNumberTypeFloat16},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)}})
    .FeedExpectedOutput({{4, 2}}, {kNumberTypeFloat16});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{1, 4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{4}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)}})
    .FeedExpectedOutput({{3, 4}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{5, -1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{5}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeInt32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32, CreateScalar(kValueAny)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt32, CreateScalar(kValueAny)}})
    .FeedExpectedOutput({{5, -1}}, {kNumberTypeInt32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, 2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(2)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(8)}})
    .FeedExpectedOutput({{-1, 2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{-1, -1}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{-2}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)},
                    InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar(kValueAny)}})
    .FeedExpectedOutput({{-1, -1}}, {kNumberTypeFloat32});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Addr, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
