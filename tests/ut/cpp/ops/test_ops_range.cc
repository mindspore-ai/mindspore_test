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

namespace mindspore {
namespace ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(3)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1000000)},
    })
    .FeedExpectedOutput({{2}}, {kNumberTypeInt64});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<double>(1)},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<double>(3)},
      InferInfoParam{ShapeVector{}, kNumberTypeFloat64, CreateScalar<double>(1)},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1000000)},
    })
    .FeedExpectedOutput({{2}}, {kNumberTypeFloat32});

  generator
    .FeedInputArgs({
      InferInfoParam{ShapeVector{}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64},
      InferInfoParam{ShapeVector{}, kNumberTypeInt64, CreateScalar<int64_t>(1000000)},
    })
    .FeedExpectedOutput({{-1}}, {kNumberTypeInt64});

  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(Range, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace ops
}  // namespace mindspore
